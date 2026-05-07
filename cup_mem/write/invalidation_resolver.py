from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..memory.models import SessionChunk, SessionDelta
from ..memory.schema import normalize_bucket_track_pairs
from ..store_layer import ProfileStore
from .invalidation_judge import InvalidationJudgeMixin
from .invalidation_lanes import InvalidationLaneMixin
from .invalidation_merge import InvalidationMergeMixin


class InvalidationResolverMixin(InvalidationLaneMixin, InvalidationMergeMixin, InvalidationJudgeMixin):
    def propose_invalidations(
        self,
        *,
        session_id: str,
        valid_chunks: List[SessionChunk],
        deltas: List[SessionDelta],
        store_view: Optional[ProfileStore] = None,
    ) -> Dict[str, Any]:
        if not valid_chunks or not deltas:
            empty_subset = {"active_items": [], "stale_items": [], "unknown_items": []}
            return {
                "raw_payload": {},
                "profile_subset": empty_subset,
                "profile_subset_summary": self._summarize_profile_subset(empty_subset, requested_pairs=[]),
                "affected_buckets": [],
                "affected_bucket_filter_logs": [],
                "direct_bucket_tracks": [],
                "direct_buckets": [],
                "expanded_invalidation_candidate_pairs": [],
                "invalidation_lane_logs": [],
                "invalidation_proposal_merge_logs": [],
                "invalidation_proposal_merger_raw_payload": {},
                "invalidation_proposal_arbitration_logs": [],
                "invalidation_proposal_filter_logs": [],
                "proposals": [],
            }

        direct_pairs, direct_buckets = self._collect_direct_bucket_tracks(valid_chunks=valid_chunks, deltas=deltas)
        bucket_bridge_result = self.propose_affected_buckets(
            session_id=session_id,
            valid_chunks=valid_chunks,
            deltas=deltas,
            direct_pairs=direct_pairs,
            direct_buckets=direct_buckets,
        )
        candidate_pairs = self._collect_invalidation_candidate_pairs(
            direct_pairs=direct_pairs,
            affected_buckets=bucket_bridge_result["affected_buckets"],
        )
        lane_specs = self._build_invalidation_lane_specs(
            direct_buckets=direct_buckets,
            direct_pairs=direct_pairs,
            affected_buckets=bucket_bridge_result["affected_buckets"],
        )
        store_ref = store_view or self.store
        global_lane_spec = self._build_global_recall_lane_spec(
            session_id=session_id,
            valid_chunks=valid_chunks,
            deltas=deltas,
            store_view=store_ref,
            direct_pairs=direct_pairs,
            direct_buckets=direct_buckets,
            affected_buckets=bucket_bridge_result["affected_buckets"],
        )
        if global_lane_spec is not None:
            lane_specs.append(global_lane_spec)
            candidate_pairs = normalize_bucket_track_pairs(
                list(candidate_pairs) + list(global_lane_spec.get("candidate_pairs", []))
            )
        profile_subset = store_ref.get_profile_subset_before_session(
            candidate_pairs,
            session_index=self._session_index_from_session_id(session_id),
        )
        valid_chunk_ids = {chunk.chunk_id for chunk in valid_chunks}
        valid_delta_ids = {delta.delta_id for delta in deltas}
        lane_results: List[Dict[str, Any]] = []
        if lane_specs:
            max_workers = max(1, min(len(lane_specs), self.thresholds.invalidation_lane_max_workers))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        self._run_invalidation_proposer_lane,
                        session_id=session_id,
                        valid_chunks=valid_chunks,
                        deltas=deltas,
                        lane_spec=lane_spec,
                        store_view=store_ref,
                        valid_chunk_ids=valid_chunk_ids,
                        valid_delta_ids=valid_delta_ids,
                    ): lane_spec
                    for lane_spec in lane_specs
                }
                for future in as_completed(future_map):
                    lane_result = future.result()
                    lane_results.append(lane_result)
            lane_results.sort(key=lambda item: str(item.get("lane_key", "")))

        proposals, proposal_filter_logs, proposal_merge_logs, proposal_merge_metadata = self._merge_lane_proposals(
            lane_results=lane_results,
        )
        proposals, merger_raw_payload, proposal_arbitration_logs = self._select_merged_invalidation_proposals(
            proposals=proposals,
            profile_subset=profile_subset,
            proposal_merge_metadata=proposal_merge_metadata,
            deltas=deltas,
        )

        lane_logs = []
        for lane_result in lane_results:
            lane_logs.append(
                {
                    "lane_key": lane_result.get("lane_key", ""),
                    "lane_type": lane_result.get("lane_type", ""),
                    "lane_focus": lane_result.get("lane_focus", {}),
                    "candidate_pairs": lane_result.get("candidate_pairs", []),
                    "profile_subset_summary": self._summarize_profile_subset(
                        lane_result.get("profile_subset", {"active_items": [], "stale_items": [], "unknown_items": []}),
                        requested_pairs=[
                            (item["bucket"], item["local_track"])
                            for item in lane_result.get("candidate_pairs", [])
                            if item.get("bucket") and item.get("local_track")
                        ],
                    ),
                    "retrieval_summary": lane_result.get("retrieval_summary", {}),
                    "raw_payload": lane_result.get("raw_payload", {}),
                }
            )

        merged_payload = [
            {
                "target_bucket": proposal.target_bucket,
                "target_local_track": proposal.target_local_track,
                "target_item_id_hint": proposal.target_item_id_hint,
                "target_old_value": proposal.target_old_value,
                "trigger_delta_ids": proposal.trigger_delta_ids,
                "evidence_chunk_ids": proposal.evidence_chunk_ids,
                "proposal_type": proposal.proposal_type,
                "reason": proposal.reason,
                "confidence": proposal.confidence,
            }
            for proposal in proposals
        ]
        return {
            "raw_payload": {
                "bucket_bridge_proposer": bucket_bridge_result["raw_payload"],
                "latent_invalidation_proposer_by_lane": [
                    {
                        "lane_key": lane_result.get("lane_key", ""),
                        "lane_type": lane_result.get("lane_type", ""),
                        "lane_focus": lane_result.get("lane_focus", {}),
                        "payload": lane_result.get("raw_payload", {}),
                    }
                    for lane_result in lane_results
                ],
                "latent_invalidation_merged": {
                    "proposal_count": len(merged_payload),
                    "proposals": merged_payload,
                },
                "invalidation_lane_merger": merger_raw_payload,
            },
            "profile_subset": profile_subset,
            "profile_subset_summary": self._summarize_profile_subset(profile_subset, requested_pairs=candidate_pairs),
            "affected_buckets": bucket_bridge_result["affected_buckets"],
            "affected_bucket_filter_logs": bucket_bridge_result.get("filter_logs", []),
            "direct_bucket_tracks": [
                {"bucket": bucket, "local_track": local_track}
                for bucket, local_track in direct_pairs
            ],
            "direct_buckets": direct_buckets,
            "expanded_invalidation_candidate_pairs": [
                {"bucket": bucket, "local_track": local_track}
                for bucket, local_track in candidate_pairs
            ],
            "invalidation_lane_logs": lane_logs,
            "invalidation_proposal_merge_logs": proposal_merge_logs,
            "invalidation_proposal_merger_raw_payload": merger_raw_payload,
            "invalidation_proposal_arbitration_logs": proposal_arbitration_logs,
            "invalidation_proposal_filter_logs": proposal_filter_logs,
            "proposals": proposals,
        }

