from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import InvalidationProposal, SessionChunk, SessionDelta, maybe_dict_list, maybe_float
from ..prompt_lib.templates import BUCKET_BRIDGE_PROPOSER_PROMPT, LATENT_INVALIDATION_PROPOSER_PROMPT
from ..memory.schema import bucket_track_pairs_for_bucket, is_valid_bucket, normalize_bucket_track_pairs
from ..store_layer import ProfileStore


class InvalidationLaneMixin:
    """Affected-bucket discovery and lane-level invalidation proposal generation."""

    def _candidate_group_summary(
        self,
        *,
        candidate_groups: Dict[str, Any],
        ranked_candidate_count: int,
    ) -> Dict[str, Any]:
        return {
            "same_track_active_count": len(candidate_groups["same_track_active"]),
            "same_bucket_active_count": len(candidate_groups["same_bucket_active"]),
            "same_track_stale_count": len(candidate_groups["same_track_stale"]),
            "unknown_current_present": candidate_groups["unknown_current"] is not None,
            "candidate_pool_size_before_topk": len(candidate_groups["same_track_active"]) + len(candidate_groups["same_bucket_active"]),
            "top_k_used": min(
                self.thresholds.candidate_top_k,
                len(candidate_groups["same_track_active"]) + len(candidate_groups["same_bucket_active"]),
            ),
            "ranked_candidate_count": ranked_candidate_count,
        }

    def _collect_direct_bucket_tracks(
        self,
        *,
        valid_chunks: List[SessionChunk],
        deltas: List[SessionDelta],
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        pairs: List[Tuple[str, str]] = []
        direct_buckets: List[str] = []
        seen_buckets = set()
        for delta in deltas:
            pairs.append((delta.bucket, delta.local_track))
            if delta.bucket not in seen_buckets:
                seen_buckets.add(delta.bucket)
                direct_buckets.append(delta.bucket)
        for chunk in valid_chunks:
            for ref in chunk.candidate_bucket_tracks:
                pairs.append((ref.bucket, ref.local_track))
                if ref.bucket not in seen_buckets:
                    seen_buckets.add(ref.bucket)
                    direct_buckets.append(ref.bucket)
        return normalize_bucket_track_pairs(pairs), direct_buckets

    def _build_bucket_bridge_payload(
        self,
        *,
        session_id: str,
        valid_chunks: List[SessionChunk],
        deltas: List[SessionDelta],
        direct_pairs: List[Tuple[str, str]],
        direct_buckets: List[str],
    ) -> str:
        return json.dumps(
            {
                "session_id": session_id,
                "bucket_track_schema": self.schema_text,
                "direct_bucket_tracks": [
                    {"bucket": bucket, "local_track": local_track}
                    for bucket, local_track in direct_pairs
                ],
                "direct_buckets": direct_buckets,
                "valid_chunks": [chunk.to_dict() for chunk in valid_chunks],
                "observed_and_upstream_deltas": [delta.to_dict() for delta in deltas],
            },
            ensure_ascii=False,
            indent=2,
        )

    def _sanitize_affected_buckets_with_logs(
        self,
        payload: Dict[str, Any],
        *,
        direct_buckets: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        direct_bucket_set = set(direct_buckets)
        sanitized: List[Dict[str, Any]] = []
        filter_logs: List[Dict[str, Any]] = []
        seen = set()
        for raw_index, raw in enumerate(maybe_dict_list(payload.get("affected_buckets", [])), start=1):
            bucket = str(raw.get("bucket", "")).strip()
            reason = str(raw.get("reason", "")).strip()
            confidence = max(0.0, min(1.0, maybe_float(raw.get("confidence", 0.0))))
            drop_reasons: List[str] = []
            if not bucket or not is_valid_bucket(bucket):
                drop_reasons.append("invalid_bucket")
            if bucket in direct_bucket_set:
                drop_reasons.append("already_direct_bucket")
            if bucket in seen:
                drop_reasons.append("duplicate_bucket")
            if not reason:
                drop_reasons.append("missing_reason")
            if len(sanitized) >= 3:
                drop_reasons.append("affected_bucket_limit_reached")

            log_record = {
                "raw_affected_bucket_index": raw_index,
                "bucket": bucket,
                "reason": reason,
                "confidence": confidence,
                "accepted": False,
                "drop_reason": drop_reasons[0] if drop_reasons else "",
                "drop_reasons": drop_reasons,
            }

            if drop_reasons:
                filter_logs.append(log_record)
                continue

            seen.add(bucket)
            accepted = {
                "bucket": bucket,
                "reason": reason,
                "confidence": confidence,
            }
            sanitized.append(accepted)
            log_record["accepted"] = True
            log_record["accepted_bucket"] = accepted
            filter_logs.append(log_record)
        return sanitized, filter_logs

    def propose_affected_buckets(
        self,
        *,
        session_id: str,
        valid_chunks: List[SessionChunk],
        deltas: List[SessionDelta],
        direct_pairs: List[Tuple[str, str]],
        direct_buckets: List[str],
    ) -> Dict[str, Any]:
        if not valid_chunks or not deltas:
            return {"raw_payload": {}, "affected_buckets": []}
        payload = self._llm_call_json(
            phase="bucket_bridge_proposer",
            system_prompt=BUCKET_BRIDGE_PROPOSER_PROMPT,
            user_payload=self._build_bucket_bridge_payload(
                session_id=session_id,
                valid_chunks=valid_chunks,
                deltas=deltas,
                direct_pairs=direct_pairs,
                direct_buckets=direct_buckets,
            ),
            session_id=session_id,
        )
        affected_buckets, filter_logs = self._sanitize_affected_buckets_with_logs(payload, direct_buckets=direct_buckets)
        return {
            "raw_payload": payload,
            "affected_buckets": affected_buckets,
            "filter_logs": filter_logs,
        }

    def _collect_invalidation_candidate_pairs(
        self,
        *,
        direct_pairs: List[Tuple[str, str]],
        affected_buckets: List[Dict[str, Any]],
    ) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = list(direct_pairs)
        for proposal in affected_buckets:
            pairs.extend(bucket_track_pairs_for_bucket(proposal["bucket"]))
        return normalize_bucket_track_pairs(pairs)

    def _build_invalidation_lane_specs(
        self,
        *,
        direct_buckets: List[str],
        direct_pairs: List[Tuple[str, str]],
        affected_buckets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        lane_specs: List[Dict[str, Any]] = []
        seen_lane_keys: set[str] = set()

        direct_pairs_by_bucket: Dict[str, List[Tuple[str, str]]] = {}
        for bucket, local_track in direct_pairs:
            direct_pairs_by_bucket.setdefault(bucket, []).append((bucket, local_track))

        for bucket in direct_buckets:
            if not is_valid_bucket(bucket):
                continue
            lane_key = f"direct_bucket::{bucket}"
            if lane_key in seen_lane_keys:
                continue
            seen_lane_keys.add(lane_key)
            lane_specs.append(
                {
                    "lane_key": lane_key,
                    "lane_type": "direct_bucket",
                    "lane_focus": {
                        "bucket": bucket,
                        "local_track": "",
                    },
                    "candidate_pairs": bucket_track_pairs_for_bucket(bucket),
                    "preferred_direct_tracks": [
                        {"bucket": pair_bucket, "local_track": pair_local_track}
                        for pair_bucket, pair_local_track in normalize_bucket_track_pairs(
                            direct_pairs_by_bucket.get(bucket, [])
                        )
                    ],
                    "affected_bucket_hints": [hint for hint in affected_buckets if hint.get("bucket") == bucket],
                }
            )

        for affected_bucket in affected_buckets:
            bucket = str(affected_bucket.get("bucket", "")).strip()
            if not is_valid_bucket(bucket):
                continue
            lane_key = f"affected_bucket::{bucket}"
            if lane_key in seen_lane_keys:
                continue
            seen_lane_keys.add(lane_key)
            lane_specs.append(
                {
                    "lane_key": lane_key,
                    "lane_type": "affected_bucket",
                    "lane_focus": {
                        "bucket": bucket,
                        "local_track": "",
                    },
                    "candidate_pairs": bucket_track_pairs_for_bucket(bucket),
                    "affected_bucket_hints": [affected_bucket],
                }
            )

        return lane_specs

    def _build_global_invalidation_recall_query(
        self,
        *,
        valid_chunks: List[SessionChunk],
        deltas: List[SessionDelta],
        direct_buckets: List[str],
        affected_buckets: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        seen_lines: set[str] = set()

        for delta in deltas:
            text = self.delta_text(delta)
            if text not in seen_lines:
                seen_lines.add(text)
                lines.append(text)

        chunk_lines_added = 0
        for chunk in valid_chunks:
            if chunk.chunk_type not in {"explicit_state", "support_evidence"} and not chunk.change_signal_present:
                continue
            text = " ".join(str(chunk.text or "").split())
            if not text:
                continue
            line = f"chunk: {text}"
            if line in seen_lines:
                continue
            seen_lines.add(line)
            lines.append(line)
            chunk_lines_added += 1
            if chunk_lines_added >= 6:
                break

        if direct_buckets:
            lines.append("direct_buckets: " + ", ".join(dict.fromkeys(direct_buckets)))
        affected_bucket_names = [
            str(item.get("bucket", "")).strip()
            for item in affected_buckets
            if str(item.get("bucket", "")).strip()
        ]
        if affected_bucket_names:
            lines.append("affected_buckets: " + ", ".join(dict.fromkeys(affected_bucket_names)))
        return "\n\n".join(lines)

    def _build_global_recall_lane_spec(
        self,
        *,
        session_id: str,
        valid_chunks: List[SessionChunk],
        deltas: List[SessionDelta],
        store_view: ProfileStore,
        direct_pairs: List[Tuple[str, str]],
        direct_buckets: List[str],
        affected_buckets: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        top_k = max(0, int(self.thresholds.invalidation_global_recall_top_k))
        if top_k <= 0:
            return None

        snapshot = store_view.to_snapshot()
        affected_bucket_names = [
            str(item.get("bucket", "")).strip()
            for item in affected_buckets
            if str(item.get("bucket", "")).strip()
        ]
        excluded_buckets = set(direct_buckets) | set(affected_bucket_names)

        def _collect_active_candidates(skip_buckets: set[str]) -> List[Dict[str, Any]]:
            candidates: List[Dict[str, Any]] = []
            for items in snapshot["active_profile"].values():
                for item in items:
                    bucket = str(item.get("bucket", "")).strip()
                    if skip_buckets and bucket in skip_buckets:
                        continue
                    strength = str(item.get("active_strength", "STRONG")).upper()
                    retrieval_text = self.profile_item_text(item) + f"\nactive_strength: {strength}"
                    candidates.append(
                        {
                            "payload": item,
                            "retrieval_text": retrieval_text,
                        }
                    )
            return candidates

        candidates = _collect_active_candidates(excluded_buckets)
        fallback_to_all_active = False
        if not candidates and excluded_buckets:
            candidates = _collect_active_candidates(set())
            fallback_to_all_active = True
        if not candidates:
            return None

        query_text = self._build_global_invalidation_recall_query(
            valid_chunks=valid_chunks,
            deltas=deltas,
            direct_buckets=direct_buckets,
            affected_buckets=affected_buckets,
        )
        ranked = self.embedding.rank(
            query_text=query_text,
            candidates=candidates,
            text_getter=lambda item: item["retrieval_text"],
            top_k=top_k,
        )
        selected_items = [entry["item"]["payload"] for entry in ranked]
        candidate_pairs = normalize_bucket_track_pairs(
            [
                (
                    str(item.get("bucket", "")).strip(),
                    str(item.get("local_track", "")).strip(),
                )
                for item in selected_items
            ]
        )
        if not candidate_pairs:
            return None

        session_index = self._session_index_from_session_id(session_id)
        profile_subset = store_view.get_exact_profile_subset_before_session(
            candidate_pairs,
            session_index=session_index,
        )
        retrieval_summary = {
            "query_text": query_text,
            "excluded_bucket_names": sorted(excluded_buckets),
            "fallback_to_all_active": fallback_to_all_active,
            "selected_item_ids": [
                str(item.get("item_id", "")).strip()
                for item in selected_items
                if str(item.get("item_id", "")).strip()
            ],
            "selected_pairs": [
                {"bucket": bucket, "local_track": local_track}
                for bucket, local_track in candidate_pairs
            ],
        }
        return {
            "lane_key": "global_recall",
            "lane_type": "global_recall",
            "lane_focus": {
                "bucket": "",
                "local_track": "",
            },
            "candidate_pairs": candidate_pairs,
            "preferred_direct_tracks": [
                {"bucket": bucket, "local_track": local_track}
                for bucket, local_track in normalize_bucket_track_pairs(direct_pairs)
            ],
            "affected_bucket_hints": list(affected_buckets),
            "profile_subset_override": profile_subset,
            "retrieval_summary": retrieval_summary,
        }

    def _build_invalidation_lane_payload(
        self,
        *,
        session_id: str,
        valid_chunks: List[SessionChunk],
        deltas: List[SessionDelta],
        profile_subset: Dict[str, List[Dict[str, Any]]],
        lane_spec: Dict[str, Any],
    ) -> str:
        return json.dumps(
            {
                "session_id": session_id,
                "bucket_track_schema": self.schema_text,
                "lane": {
                    "lane_key": lane_spec.get("lane_key", ""),
                    "lane_type": lane_spec.get("lane_type", ""),
                    "lane_focus": lane_spec.get("lane_focus", {}),
                    "preferred_direct_tracks": lane_spec.get("preferred_direct_tracks", []),
                    "candidate_pairs": [
                        {"bucket": bucket, "local_track": local_track}
                        for bucket, local_track in lane_spec.get("candidate_pairs", [])
                    ],
                },
                "valid_chunks": [chunk.to_dict() for chunk in valid_chunks],
                "observed_and_upstream_deltas": [delta.to_dict() for delta in deltas],
                "dynamic_affected_bucket_hints": lane_spec.get("affected_bucket_hints", []),
                "candidate_active_profile_subset": profile_subset,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _sanitize_lane_proposal_log(
        self,
        *,
        filter_log: Dict[str, Any],
        lane_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        lane_enriched = dict(filter_log)
        lane_enriched["lane_key"] = lane_spec.get("lane_key", "")
        lane_enriched["lane_type"] = lane_spec.get("lane_type", "")
        lane_enriched["lane_focus"] = lane_spec.get("lane_focus", {})
        return lane_enriched

    def _run_invalidation_proposer_lane(
        self,
        *,
        session_id: str,
        valid_chunks: List[SessionChunk],
        deltas: List[SessionDelta],
        lane_spec: Dict[str, Any],
        store_view: ProfileStore,
        valid_chunk_ids: set[str],
        valid_delta_ids: set[str],
    ) -> Dict[str, Any]:
        lane_candidate_pairs = normalize_bucket_track_pairs(lane_spec.get("candidate_pairs", []))
        lane_profile_subset_override = lane_spec.get("profile_subset_override")
        if isinstance(lane_profile_subset_override, dict):
            lane_profile_subset = lane_profile_subset_override
        else:
            lane_profile_subset = store_view.get_profile_subset_before_session(
                lane_candidate_pairs,
                session_index=self._session_index_from_session_id(session_id),
            )
        payload = self._llm_call_json(
            phase="latent_invalidation_proposer",
            system_prompt=LATENT_INVALIDATION_PROPOSER_PROMPT,
            user_payload=self._build_invalidation_lane_payload(
                session_id=session_id,
                valid_chunks=valid_chunks,
                deltas=deltas,
                profile_subset=lane_profile_subset,
                lane_spec=lane_spec,
            ),
            session_id=session_id,
            extra_meta={
                "lane_key": lane_spec.get("lane_key", ""),
                "lane_type": lane_spec.get("lane_type", ""),
            },
            extra_request_kwargs={
                "temperature": self.thresholds.latent_invalidation_temperature,
            },
        )
        valid_item_ids = {item["item_id"] for item in lane_profile_subset["active_items"]}
        proposals: List[InvalidationProposal] = []
        proposal_filter_logs: List[Dict[str, Any]] = []
        for raw_index, raw_proposal in enumerate(maybe_dict_list(payload.get("proposals", [])), start=1):
            proposal, filter_log = self._sanitize_invalidation_proposal_with_log(
                raw_proposal,
                raw_index=raw_index,
                session_id=session_id,
                valid_chunk_ids=valid_chunk_ids,
                valid_delta_ids=valid_delta_ids,
                valid_item_ids=valid_item_ids,
            )
            lane_filter_log = self._sanitize_lane_proposal_log(filter_log=filter_log, lane_spec=lane_spec)
            if proposal is not None:
                proposals.append(proposal)
            proposal_filter_logs.append(lane_filter_log)
        return {
            "lane_key": lane_spec.get("lane_key", ""),
            "lane_type": lane_spec.get("lane_type", ""),
            "lane_focus": lane_spec.get("lane_focus", {}),
            "candidate_pairs": [
                {"bucket": bucket, "local_track": local_track}
                for bucket, local_track in lane_candidate_pairs
            ],
            "retrieval_summary": lane_spec.get("retrieval_summary", {}),
            "profile_subset": lane_profile_subset,
            "raw_payload": payload,
            "proposal_filter_logs": proposal_filter_logs,
            "proposals": proposals,
        }

