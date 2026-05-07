from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..memory.models import InvalidationProposal, SessionDelta, maybe_float, maybe_str_list
from ..memory.schema import is_valid_bucket_track


class InvalidationMergeMixin:
    """Merge and deterministic selection of invalidation proposals from lanes."""

    def _proposal_merge_key(self, proposal: InvalidationProposal) -> Tuple[Any, ...]:
        target_hint = str(proposal.target_item_id_hint or "").strip()
        if target_hint:
            return ("target_item_id_hint", target_hint, proposal.target_bucket, proposal.target_local_track)
        return (
            "target_old_value",
            proposal.target_bucket,
            proposal.target_local_track,
            self._normalize_basis_text(str(proposal.target_old_value or "").strip()),
        )

    @staticmethod
    def _merge_unique_preserve_order(values: Iterable[str]) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        return merged

    @classmethod
    def _merge_reason_snippets(cls, reasons: Iterable[str], *, max_items: int = 3) -> str:
        merged = cls._merge_unique_preserve_order(reasons)
        if not merged:
            return ""
        return " | ".join(merged[:max_items])

    def _merge_lane_proposals(
        self,
        *,
        lane_results: List[Dict[str, Any]],
    ) -> Tuple[
        List[InvalidationProposal],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        Dict[str, Dict[str, Any]],
    ]:
        merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        merge_logs: List[Dict[str, Any]] = []
        proposal_filter_logs: List[Dict[str, Any]] = []

        for lane_result in lane_results:
            proposal_filter_logs.extend(lane_result.get("proposal_filter_logs", []))
            lane_key = str(lane_result.get("lane_key", "")).strip()
            lane_type = str(lane_result.get("lane_type", "")).strip()
            for proposal in lane_result.get("proposals", []):
                merge_key = self._proposal_merge_key(proposal)
                existing = merged.get(merge_key)
                if existing is None:
                    merged[merge_key] = {
                        "proposal": proposal,
                        "lane_keys": [lane_key],
                        "lane_types": [lane_type],
                        "reasons": [proposal.reason],
                    }
                    merge_logs.append(
                        {
                            "merge_key": list(merge_key),
                            "action": "accepted_first",
                            "selected_lane_key": lane_key,
                            "selected_lane_type": lane_type,
                            "proposal": proposal.to_dict(),
                        }
                    )
                    continue

                existing_proposal = existing["proposal"]
                merged_trigger_delta_ids = self._merge_unique_preserve_order(
                    list(existing_proposal.trigger_delta_ids) + list(proposal.trigger_delta_ids)
                )
                merged_evidence_chunk_ids = self._merge_unique_preserve_order(
                    list(existing_proposal.evidence_chunk_ids) + list(proposal.evidence_chunk_ids)
                )
                merged_reasons = list(existing.get("reasons", [])) + [proposal.reason]
                merged_reason_text = self._merge_reason_snippets(merged_reasons)
                merged_confidence = max(existing_proposal.confidence, proposal.confidence)
                merged_target_item_id_hint = existing_proposal.target_item_id_hint or proposal.target_item_id_hint
                existing_old_value = str(existing_proposal.target_old_value or "").strip()
                incoming_old_value = str(proposal.target_old_value or "").strip()
                if self._is_placeholder_target_old_value(existing_old_value) and not self._is_placeholder_target_old_value(incoming_old_value):
                    merged_target_old_value = incoming_old_value
                elif existing_old_value:
                    merged_target_old_value = existing_old_value
                else:
                    merged_target_old_value = incoming_old_value

                existing["proposal"] = InvalidationProposal(
                    proposal_id=existing_proposal.proposal_id,
                    session_id=existing_proposal.session_id,
                    target_bucket=existing_proposal.target_bucket,
                    target_local_track=existing_proposal.target_local_track,
                    trigger_delta_ids=merged_trigger_delta_ids,
                    evidence_chunk_ids=merged_evidence_chunk_ids,
                    reason=merged_reason_text,
                    confidence=merged_confidence,
                    proposal_type=existing_proposal.proposal_type,
                    target_item_id_hint=merged_target_item_id_hint,
                    target_old_value=merged_target_old_value,
                )
                existing["lane_keys"].append(lane_key)
                existing["lane_types"].append(lane_type)
                existing["reasons"] = self._merge_unique_preserve_order(merged_reasons)

                merge_logs.append(
                    {
                        "merge_key": list(merge_key),
                        "action": "merged_duplicate_target",
                        "selected_lane_key": existing["lane_keys"][0],
                        "selected_lane_type": existing["lane_types"][0],
                        "existing_proposal": existing_proposal.to_dict(),
                        "incoming_proposal": proposal.to_dict(),
                        "merged_trigger_delta_ids": merged_trigger_delta_ids,
                        "merged_evidence_chunk_ids": merged_evidence_chunk_ids,
                        "merged_reason": merged_reason_text,
                    }
                )

        merged_proposals = [entry["proposal"] for entry in merged.values()]
        merged_proposals.sort(
            key=lambda proposal: (
                proposal.target_bucket,
                proposal.target_local_track,
                -proposal.confidence,
                proposal.proposal_id,
            )
        )
        proposal_merge_metadata = {
            entry["proposal"].proposal_id: {
                "lane_keys": list(dict.fromkeys(entry["lane_keys"])),
                "lane_types": list(dict.fromkeys(entry["lane_types"])),
                "reason_snippets": list(entry.get("reasons", [])),
            }
            for entry in merged.values()
        }
        return merged_proposals, proposal_filter_logs, merge_logs, proposal_merge_metadata

    @staticmethod
    def _is_placeholder_target_old_value(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
        return normalized in {"", "unknown", "n/a", "na", "none", "not available"}

    def _proposal_old_value_match_score(
        self,
        *,
        proposal: InvalidationProposal,
        active_by_track: Dict[Tuple[str, str], List[Dict[str, Any]]],
        active_by_bucket: Dict[str, List[Dict[str, Any]]],
    ) -> int:
        target_old_value = str(proposal.target_old_value or "").strip()
        if self._is_placeholder_target_old_value(target_old_value):
            return 0
        normalized_target = self._normalize_basis_text(target_old_value)
        if not normalized_target:
            return 0

        same_track_items = active_by_track.get((proposal.target_bucket, proposal.target_local_track), [])
        for item in same_track_items:
            normalized_value = self._normalize_basis_text(str(item.get("value", "")).strip())
            if normalized_value and (
                normalized_target == normalized_value
                or normalized_target in normalized_value
                or normalized_value in normalized_target
            ):
                return 2

        for item in active_by_bucket.get(proposal.target_bucket, []):
            normalized_value = self._normalize_basis_text(str(item.get("value", "")).strip())
            if normalized_value and (
                normalized_target == normalized_value
                or normalized_target in normalized_value
                or normalized_value in normalized_target
            ):
                return 1
        return 0

    def _proposal_exact_active_target_hit(
        self,
        *,
        proposal: InvalidationProposal,
        active_by_id: Dict[str, Dict[str, Any]],
    ) -> int:
        target_item_id = str(proposal.target_item_id_hint or "").strip()
        if not target_item_id:
            return 0
        item = active_by_id.get(target_item_id)
        if item is None:
            return 0
        return int(
            str(item.get("bucket", "")).strip() == proposal.target_bucket
            and str(item.get("local_track", "")).strip() == proposal.target_local_track
        )

    def _proposal_self_track_reassessment_penalty(
        self,
        *,
        proposal: InvalidationProposal,
        delta_by_id: Dict[str, SessionDelta],
        active_by_track: Dict[Tuple[str, str], List[Dict[str, Any]]],
    ) -> int:
        target_pair = (proposal.target_bucket, proposal.target_local_track)
        trigger_pairs = {
            (delta_by_id[delta_id].bucket, delta_by_id[delta_id].local_track)
            for delta_id in proposal.trigger_delta_ids
            if delta_id in delta_by_id
        }
        if target_pair not in trigger_pairs:
            return 0
        if active_by_track.get(target_pair):
            return 0
        if str(proposal.target_item_id_hint or "").strip():
            return 0
        return 1

    def _select_merged_invalidation_proposals(
        self,
        *,
        proposals: List[InvalidationProposal],
        profile_subset: Dict[str, List[Dict[str, Any]]],
        proposal_merge_metadata: Dict[str, Dict[str, Any]],
        deltas: List[SessionDelta],
    ) -> Tuple[List[InvalidationProposal], Dict[str, Any], List[Dict[str, Any]]]:
        if not proposals:
            return [], {"mode": "deterministic_reducer", "selected_proposal_ids": [], "proposal_scores": []}, []

        active_items = list(profile_subset.get("active_items", []))
        active_by_id = {
            str(item.get("item_id", "")).strip(): item
            for item in active_items
            if str(item.get("item_id", "")).strip()
        }
        active_by_track: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        active_by_bucket: Dict[str, List[Dict[str, Any]]] = {}
        for item in active_items:
            bucket = str(item.get("bucket", "")).strip()
            local_track = str(item.get("local_track", "")).strip()
            active_by_track.setdefault((bucket, local_track), []).append(item)
            active_by_bucket.setdefault(bucket, []).append(item)
        delta_by_id = {delta.delta_id: delta for delta in deltas}

        scored_entries: List[Dict[str, Any]] = []
        for proposal in proposals:
            merge_meta = proposal_merge_metadata.get(proposal.proposal_id, {})
            lane_keys = list(dict.fromkeys(maybe_str_list(merge_meta.get("lane_keys", []))))
            lane_types = list(dict.fromkeys(maybe_str_list(merge_meta.get("lane_types", []))))
            exact_active_target_hit = self._proposal_exact_active_target_hit(
                proposal=proposal,
                active_by_id=active_by_id,
            )
            old_value_match_score = self._proposal_old_value_match_score(
                proposal=proposal,
                active_by_track=active_by_track,
                active_by_bucket=active_by_bucket,
            )
            has_old_track = int(bool(active_by_track.get((proposal.target_bucket, proposal.target_local_track))))
            has_old_bucket = int(bool(active_by_bucket.get(proposal.target_bucket)))
            self_track_penalty = self._proposal_self_track_reassessment_penalty(
                proposal=proposal,
                delta_by_id=delta_by_id,
                active_by_track=active_by_track,
            )
            concrete_old_value = int(not self._is_placeholder_target_old_value(proposal.target_old_value))
            corroboration_count = len(lane_keys)
            lane_type_diversity = len(lane_types)
            global_recall_present = int("global_recall" in lane_types)
            direct_bucket_present = int("direct_bucket" in lane_types)

            selection_key = (
                exact_active_target_hit,
                old_value_match_score,
                concrete_old_value,
                corroboration_count,
                lane_type_diversity,
                global_recall_present,
                direct_bucket_present,
                proposal.confidence,
                1 - self_track_penalty,
                has_old_track,
                has_old_bucket,
            )
            scored_entries.append(
                {
                    "proposal": proposal,
                    "lane_keys": lane_keys,
                    "lane_types": lane_types,
                    "selection_key": selection_key,
                    "score_breakdown": {
                        "exact_active_target_hit": exact_active_target_hit,
                        "old_value_match_score": old_value_match_score,
                        "has_old_track": has_old_track,
                        "has_old_bucket": has_old_bucket,
                        "not_self_track_reassessment": 1 - self_track_penalty,
                        "concrete_old_value": concrete_old_value,
                        "corroboration_count": corroboration_count,
                        "lane_type_diversity": lane_type_diversity,
                        "global_recall_present": global_recall_present,
                        "direct_bucket_present": direct_bucket_present,
                        "confidence": proposal.confidence,
                    },
                }
            )

        scored_entries.sort(
            key=lambda item: (
                item["selection_key"],
                item["proposal"].proposal_id,
            ),
            reverse=True,
        )
        kept: List[Dict[str, Any]] = []
        for entry in scored_entries:
            if len(kept) >= self.thresholds.invalidation_merge_max_keep:
                break
            kept.append(entry)

        selected_ids = [entry["proposal"].proposal_id for entry in kept]
        final_proposals = [entry["proposal"] for entry in kept]
        arbitration_logs: List[Dict[str, Any]] = [
            {
                "proposal_id": entry["proposal"].proposal_id,
                "action": "kept" if entry["proposal"].proposal_id in selected_ids else "dropped_by_deterministic_reducer",
                "selection_key": list(entry["selection_key"]),
                "score_breakdown": entry["score_breakdown"],
                "lane_keys": entry["lane_keys"],
                "lane_types": entry["lane_types"],
            }
            for entry in scored_entries
        ]
        payload = {
            "mode": "deterministic_reducer",
            "selected_proposal_ids": selected_ids,
            "proposal_scores": [
                {
                    "proposal_id": entry["proposal"].proposal_id,
                    "selection_key": list(entry["selection_key"]),
                    "score_breakdown": entry["score_breakdown"],
                    "lane_keys": entry["lane_keys"],
                    "lane_types": entry["lane_types"],
                    "target_bucket": entry["proposal"].target_bucket,
                    "target_local_track": entry["proposal"].target_local_track,
                    "target_item_id_hint": entry["proposal"].target_item_id_hint,
                    "target_old_value": entry["proposal"].target_old_value,
                }
                for entry in scored_entries
            ],
        }
        return final_proposals, payload, arbitration_logs

    def _sanitize_invalidation_proposal(
        self,
        raw: Dict[str, Any],
        *,
        session_id: str,
        valid_chunk_ids: set[str],
        valid_delta_ids: set[str],
        valid_item_ids: set[str],
    ) -> Optional[InvalidationProposal]:
        target_bucket = str(raw.get("target_bucket", "")).strip()
        target_local_track = str(raw.get("target_local_track", "")).strip()
        if not is_valid_bucket_track(target_bucket, target_local_track):
            return None
        trigger_delta_ids = self._resolve_reference_ids(
            maybe_str_list(raw.get("trigger_delta_ids", [])),
            valid_delta_ids,
            prefix="d",
        )
        evidence_chunk_ids = self._resolve_reference_ids(
            maybe_str_list(raw.get("evidence_chunk_ids", [])),
            valid_chunk_ids,
            prefix="c",
        )
        reason = str(raw.get("reason", "")).strip()
        target_item_id_hint = str(raw.get("target_item_id_hint", "")).strip()
        if target_item_id_hint and target_item_id_hint not in valid_item_ids:
            target_item_id_hint = ""
        proposal_type = str(raw.get("proposal_type", "INVALIDATE_OR_REASSESS")).strip().upper() or "INVALIDATE_OR_REASSESS"
        if proposal_type not in {"INVALIDATE_OR_REASSESS"}:
            proposal_type = "INVALIDATE_OR_REASSESS"
        if not trigger_delta_ids or not evidence_chunk_ids or not reason:
            return None
        return InvalidationProposal(
            proposal_id=self._next_proposal_id(),
            session_id=session_id,
            target_bucket=target_bucket,
            target_local_track=target_local_track,
            trigger_delta_ids=trigger_delta_ids,
            evidence_chunk_ids=evidence_chunk_ids,
            reason=reason,
            confidence=max(0.0, min(1.0, maybe_float(raw.get("confidence", 0.0)))),
            proposal_type=proposal_type,
            target_item_id_hint=target_item_id_hint,
            target_old_value=str(raw.get("target_old_value", "")).strip(),
        )

    def _sanitize_invalidation_proposal_with_log(
        self,
        raw: Dict[str, Any],
        *,
        raw_index: int,
        session_id: str,
        valid_chunk_ids: set[str],
        valid_delta_ids: set[str],
        valid_item_ids: set[str],
    ) -> Tuple[Optional[InvalidationProposal], Dict[str, Any]]:
        target_bucket = str(raw.get("target_bucket", "")).strip()
        target_local_track = str(raw.get("target_local_track", "")).strip()
        trigger_delta_ids = self._resolve_reference_ids(
            maybe_str_list(raw.get("trigger_delta_ids", [])),
            valid_delta_ids,
            prefix="d",
        )
        evidence_chunk_ids = self._resolve_reference_ids(
            maybe_str_list(raw.get("evidence_chunk_ids", [])),
            valid_chunk_ids,
            prefix="c",
        )
        reason = str(raw.get("reason", "")).strip()
        confidence = max(0.0, min(1.0, maybe_float(raw.get("confidence", 0.0))))
        target_item_id_hint = str(raw.get("target_item_id_hint", "")).strip()
        warnings: List[str] = []
        if target_item_id_hint and target_item_id_hint not in valid_item_ids:
            warnings.append("invalid_target_item_id_hint_cleared")
            target_item_id_hint = ""

        drop_reasons: List[str] = []
        if not is_valid_bucket_track(target_bucket, target_local_track):
            drop_reasons.append("invalid_target_bucket_track")
        if not trigger_delta_ids:
            drop_reasons.append("missing_valid_trigger_delta_ids")
        if not evidence_chunk_ids:
            drop_reasons.append("missing_valid_evidence_chunk_ids")
        if not reason:
            drop_reasons.append("missing_reason")

        log_record: Dict[str, Any] = {
            "raw_proposal_index": raw_index,
            "target_bucket": target_bucket,
            "target_local_track": target_local_track,
            "trigger_delta_ids": trigger_delta_ids,
            "evidence_chunk_ids": evidence_chunk_ids,
            "confidence": confidence,
            "target_item_id_hint": target_item_id_hint,
            "target_old_value": str(raw.get("target_old_value", "")).strip(),
            "accepted": False,
            "drop_reason": drop_reasons[0] if drop_reasons else "",
            "drop_reasons": drop_reasons,
            "warnings": warnings,
        }

        if drop_reasons:
            return None, log_record

        proposal = self._sanitize_invalidation_proposal(
            raw,
            session_id=session_id,
            valid_chunk_ids=valid_chunk_ids,
            valid_delta_ids=valid_delta_ids,
            valid_item_ids=valid_item_ids,
        )
        if proposal is None:
            log_record["drop_reason"] = "sanitize_invalidation_proposal_failed"
            log_record["drop_reasons"] = ["sanitize_invalidation_proposal_failed"]
            return None, log_record
        log_record["accepted"] = True
        log_record["proposal_id"] = proposal.proposal_id
        return proposal, log_record

