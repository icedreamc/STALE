from __future__ import annotations

from typing import Any, Dict, List

from ..memory.models import SessionChunk, SessionDelta, maybe_dict_list


class SessionWriterMixin:
    """Session write orchestration."""

    def process_session(
        self,
        *,
        session: List[Dict[str, Any]],
        session_index: int,
        session_time: str,
    ) -> Dict[str, Any]:
        session_id = f"s_{session_index:03d}"
        user_turns = self.session_to_user_turns(session)
        chunk_result = self.chunk_session(session_id=session_id, user_turns=user_turns)
        chunks: List[SessionChunk] = chunk_result["chunks"]
        chunk_filter_logs: List[Dict[str, Any]] = []
        valid_chunks: List[SessionChunk] = []
        for chunk in chunks:
            chunk_log = self._evaluate_chunk_validity(chunk)
            chunk_filter_logs.append(chunk_log)
            if chunk_log["passed_valid_chunk_filter"]:
                valid_chunks.append(chunk)
        decision_store = self.store.clone()
        delta_result = self.extract_deltas(
            session_id=session_id,
            valid_chunks=valid_chunks,
            store_view=decision_store,
            session_time=session_time,
        )
        update_decision_records: List[Dict[str, Any]] = []
        for delta in delta_result["deltas"]:
            update_decision_records.append(
                {
                    "delta": delta,
                    "judge_result": self.judge_update(delta, store_view=decision_store),
                }
            )
        provisional_semantic_deltas = [
            record["delta"]
            for record in update_decision_records
            if str(record["judge_result"]["decision"].decision or "").strip().upper() != "NO_OP"
        ]

        invalidation_result = self.propose_invalidations(
            session_id=session_id,
            valid_chunks=valid_chunks,
            deltas=provisional_semantic_deltas,
            store_view=decision_store,
        )

        invalidation_decision_records: List[Dict[str, Any]] = []
        for proposal in invalidation_result["proposals"]:
            invalidation_decision_records.append(
                {
                    "proposal": proposal,
                    "judge_result": self.judge_invalidation(proposal, store_view=decision_store),
                }
            )

        same_session_arbitration = self._arbitrate_same_session_actions(
            update_decision_records=update_decision_records,
            invalidation_decision_records=invalidation_decision_records,
        )
        suppressed_records = same_session_arbitration["suppressed_records"]
        commit_store = self.store.clone()
        delta_logs: List[Dict[str, Any]] = []
        accepted_semantic_delta_ids: set[str] = set()
        arbitration_logs: List[Dict[str, Any]] = list(same_session_arbitration["arbitration_logs"])
        for update_index, record in enumerate(update_decision_records):
            delta = record["delta"]
            judge_result = record["judge_result"]
            suppression_record = suppressed_records.get(
                ("update", update_index),
            )
            if suppression_record is not None:
                current_snapshot = commit_store.to_snapshot()
                apply_result = {
                    "decision": judge_result["decision"].to_dict(),
                    "applied_action": "NO_OP_SUPPRESSED_BY_SESSION_ACTION_ARBITRATION",
                    "before": current_snapshot,
                    "after": current_snapshot,
                }
            else:
                apply_result = commit_store.apply_update(delta, judge_result["decision"], timestamp=session_time)
            if self._is_semantic_accept(apply_result["applied_action"]):
                accepted_semantic_delta_ids.add(delta.delta_id)
            delta_logs.append(
                {
                    "delta": self._delta_trace_dict(delta),
                    "candidate_groups": judge_result["candidate_groups"],
                    "candidate_pool_summary": judge_result["candidate_pool_summary"],
                    "ranked_candidates": judge_result["ranked_candidates"],
                    "judge_raw_payload": judge_result["raw_payload"],
                    "decision": judge_result["decision"].to_dict(),
                    "suppression_record": suppression_record,
                    "apply_result": apply_result,
                }
            )
        invalidation_logs: List[Dict[str, Any]] = []
        weak_suppression_logs: List[Dict[str, Any]] = []
        update_conflict_suppression_logs: List[Dict[str, Any]] = []
        invalidated_stale_items_for_linking: List[Dict[str, Any]] = []
        for invalidation_index, record in enumerate(invalidation_decision_records):
            proposal = record["proposal"]
            judge_result = record["judge_result"]
            suppression_record = suppressed_records.get(("invalidation", invalidation_index))
            if suppression_record is not None:
                current_snapshot = commit_store.to_snapshot()
                apply_result = {
                    "decision": judge_result["decision"].to_dict(),
                    "applied_action": "NO_OP_SUPPRESSED_BY_SESSION_ACTION_ARBITRATION",
                    "before": current_snapshot,
                    "after": current_snapshot,
                }
                if suppression_record["winner_source"] == "update":
                    update_conflict_suppression_logs.append(
                        {
                            "proposal_id": proposal.proposal_id,
                            "target_item_id": judge_result["decision"].target_item_id,
                            **suppression_record,
                        }
                    )
                else:
                    weak_suppression_logs.append(
                        {
                            "proposal_id": proposal.proposal_id,
                            "target_item_id": judge_result["decision"].target_item_id,
                            **suppression_record,
                        }
                    )
            else:
                apply_result = commit_store.apply_invalidation(proposal, judge_result["decision"], timestamp=session_time)
            if self._is_semantic_accept(apply_result["applied_action"]):
                accepted_semantic_delta_ids.update(proposal.trigger_delta_ids)
            after_snapshot = apply_result.get("after", {}) if isinstance(apply_result, dict) else {}
            stale_archive = maybe_dict_list(after_snapshot.get("stale_archive", [])) if isinstance(after_snapshot, dict) else []
            target_item_id = str(judge_result["decision"].target_item_id or "").strip()
            if target_item_id:
                matched_stale = next(
                    (item for item in stale_archive if str(item.get("item_id", "")).strip() == target_item_id),
                    None,
                )
                if matched_stale is not None:
                    invalidated_stale_items_for_linking.append(matched_stale)
            invalidation_logs.append(
                {
                    "proposal": proposal.to_dict(),
                    "candidate_groups": judge_result["candidate_groups"],
                    "candidate_pool_summary": judge_result["candidate_pool_summary"],
                    "ranked_candidates": judge_result["ranked_candidates"],
                    "judge_raw_payload": judge_result["raw_payload"],
                    "decision": judge_result["decision"].to_dict(),
                    "suppression_record": suppression_record,
                    "apply_result": apply_result,
                }
            )
        self.store = commit_store
        accepted_semantic_deltas = [
            delta for delta in delta_result["deltas"] if delta.delta_id in accepted_semantic_delta_ids
        ]
        self.delta_store.extend(accepted_semantic_deltas)
        stale_support_result = self.link_stale_support(
            session_id=session_id,
            session_index=session_index,
            deltas=accepted_semantic_deltas,
            affected_buckets=invalidation_result.get("affected_buckets", []),
            invalidated_stale_items=invalidated_stale_items_for_linking,
        )
        session_summary = {
            "chunk_count": len(chunks),
            "valid_chunk_count": len(valid_chunks),
            "dropped_chunk_count": len([log for log in chunk_filter_logs if not log["passed_valid_chunk_filter"]]),
            "raw_delta_count": len(delta_result.get("delta_filter_logs", [])),
            "accepted_delta_count": len(delta_result["deltas"]),
            "dropped_delta_count": len([log for log in delta_result.get("delta_filter_logs", []) if not log.get("accepted")]),
            "provisional_semantic_delta_count": len(provisional_semantic_deltas),
            "affected_bucket_count": len(invalidation_result.get("affected_buckets", [])),
            "raw_invalidation_proposal_count": len(invalidation_result.get("invalidation_proposal_filter_logs", [])),
            "accepted_invalidation_proposal_count": len(invalidation_result["proposals"]),
            "dropped_invalidation_proposal_count": len(
                [log for log in invalidation_result.get("invalidation_proposal_filter_logs", []) if not log.get("accepted")]
            ),
            "applied_invalidation_count": len(invalidation_logs),
            "accepted_semantic_delta_count": len(accepted_semantic_deltas),
            "weak_suppression_count": len(weak_suppression_logs),
            "update_arbitration_claimed_target_count": len(
                [
                    record
                    for record in update_decision_records
                    if str(record["judge_result"]["decision"].target_item_id or "").strip()
                ]
            ),
            "update_conflict_suppression_count": len(update_conflict_suppression_logs),
            "same_session_action_arbitration_count": len(arbitration_logs),
            "same_session_action_suppression_count": len(suppressed_records),
            "stale_support_link_added_count": stale_support_result["summary"]["added_count"],
            "stale_support_link_deduped_count": stale_support_result["summary"]["deduped_count"],
        }
        return {
            "session_id": session_id,
            "session_index": session_index,
            "session_time": session_time,
            "user_turns": user_turns,
            "chunker_raw_payload": chunk_result["raw_payload"],
            "chunks": [self._chunk_trace_dict(chunk) for chunk in chunks],
            "chunk_filter_logs": chunk_filter_logs,
            "valid_chunks": [self._chunk_trace_dict(chunk) for chunk in valid_chunks],
            "delta_extractor_raw_payload": delta_result["raw_payload"],
            "delta_profile_subset": delta_result["profile_subset"],
            "delta_profile_subset_summary": delta_result.get("profile_subset_summary"),
            "delta_filter_logs": delta_result.get("delta_filter_logs", []),
            "provisional_semantic_delta_ids": [delta.delta_id for delta in provisional_semantic_deltas],
            "delta_logs": delta_logs,
            "latent_invalidation_proposer_raw_payload": invalidation_result["raw_payload"],
            "latent_invalidation_profile_subset": invalidation_result["profile_subset"],
            "latent_invalidation_profile_subset_summary": invalidation_result.get("profile_subset_summary"),
            "affected_buckets": invalidation_result["affected_buckets"],
            "affected_bucket_filter_logs": invalidation_result.get("affected_bucket_filter_logs", []),
            "direct_bucket_tracks": invalidation_result.get("direct_bucket_tracks", []),
            "direct_buckets": invalidation_result.get("direct_buckets", []),
            "expanded_invalidation_candidate_pairs": invalidation_result.get("expanded_invalidation_candidate_pairs", []),
            "invalidation_lane_logs": invalidation_result.get("invalidation_lane_logs", []),
            "invalidation_proposal_merge_logs": invalidation_result.get("invalidation_proposal_merge_logs", []),
            "invalidation_proposal_merger_raw_payload": invalidation_result.get("invalidation_proposal_merger_raw_payload", {}),
            "invalidation_proposal_arbitration_logs": invalidation_result.get("invalidation_proposal_arbitration_logs", []),
            "invalidation_proposal_filter_logs": invalidation_result.get("invalidation_proposal_filter_logs", []),
            "accepted_semantic_delta_ids": [delta.delta_id for delta in accepted_semantic_deltas],
            "invalidation_logs": invalidation_logs,
            "arbitration_logs": arbitration_logs + update_conflict_suppression_logs,
            "weak_suppression_logs": weak_suppression_logs,
            "stale_support_logs": stale_support_result["link_logs"],
            "stale_support_summary": stale_support_result["summary"],
            "session_summary": session_summary,
            "profile_snapshot_after_session": self.store.to_snapshot(),
        }


    def write_session(self, *, session: List[Dict[str, Any]], session_index: int, session_time: str) -> Dict[str, Any]:
        return self.process_session(session=session, session_index=session_index, session_time=session_time)
