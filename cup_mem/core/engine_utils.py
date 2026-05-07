from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..memory.models import ConstraintSet, PremiseVerdict, QueryParse, maybe_str_list
from ..memory.schema import normalize_bucket_track_pairs


class EngineUtilityMixin:
    """Shared helper methods for write/query mixins."""

    def _debug_trace_enabled(self) -> bool:
        trace_config = getattr(self, "trace_config", None)
        return bool(getattr(trace_config, "enable_debug_trace", False))

    def _attach_debug_trace(
        self,
        payload: Dict[str, Any],
        *,
        section: str,
        fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._debug_trace_enabled():
            return payload
        filtered = {
            key: value
            for key, value in fields.items()
            if value not in ("", None, [], {})
        }
        if not filtered:
            return payload
        debug_trace = payload.setdefault("debug_trace", {})
        section_payload = debug_trace.setdefault(section, {})
        section_payload.update(filtered)
        return payload

    def _move_aux_fields_to_debug_trace(
        self,
        payload: Dict[str, Any],
        *,
        section: str,
        field_names: Iterable[str],
    ) -> Dict[str, Any]:
        sanitized = dict(payload)
        moved: Dict[str, Any] = {}
        for field_name in field_names:
            if field_name in sanitized:
                moved[field_name] = sanitized.pop(field_name)
        return self._attach_debug_trace(sanitized, section=section, fields=moved)

    def _chunk_trace_dict(self, chunk: Any) -> Dict[str, Any]:
        chunk_dict = chunk.to_dict() if hasattr(chunk, "to_dict") else dict(chunk)
        return self._move_aux_fields_to_debug_trace(
            chunk_dict,
            section="write",
            field_names=["change_signal_summary"],
        )

    def _delta_trace_dict(self, delta: Any) -> Dict[str, Any]:
        delta_dict = delta.to_dict() if hasattr(delta, "to_dict") else dict(delta)
        return self._move_aux_fields_to_debug_trace(
            delta_dict,
            section="write",
            field_names=["op_hint"],
        )

    @staticmethod
    def _preview_text(text: str, *, limit: int = 160) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)] + "..."

    def _infer_session_ids_from_payload(self, payload: Dict[str, Any]) -> List[str]:
        session_ids: List[str] = []
        direct_session_id = str(payload.get("session_id", "")).strip()
        if direct_session_id:
            session_ids.append(direct_session_id)

        chunk_session_map = {chunk.chunk_id: chunk.session_id for chunk in self.chunk_bank}
        delta_session_map = {delta.delta_id: delta.session_id for delta in self.delta_store}

        for chunk_id in maybe_str_list(payload.get("evidence_chunk_ids", [])):
            session_id = chunk_session_map.get(chunk_id)
            if session_id and session_id not in session_ids:
                session_ids.append(session_id)
        for delta_id in maybe_str_list(payload.get("source_delta_ids", [])):
            session_id = delta_session_map.get(delta_id)
            if session_id and session_id not in session_ids:
                session_ids.append(session_id)
        return session_ids

    @staticmethod
    def _resolve_reference_ids(raw_ids: List[str], valid_ids: Iterable[str], *, prefix: str) -> List[str]:
        valid_ids = set(valid_ids)
        numeric_to_valid: Dict[int, str] = {}
        for valid_id in valid_ids:
            match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", str(valid_id))
            if match:
                numeric_to_valid[int(match.group(1))] = str(valid_id)

        resolved: List[str] = []
        seen: set[str] = set()
        for raw_id in raw_ids:
            raw_text = str(raw_id or "").strip()
            if not raw_text:
                continue
            resolved_id = raw_text if raw_text in valid_ids else ""
            if not resolved_id:
                match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", raw_text)
                if match:
                    resolved_id = numeric_to_valid.get(int(match.group(1)), "")
            if resolved_id and resolved_id not in seen:
                seen.add(resolved_id)
                resolved.append(resolved_id)
        return resolved

    def _summarize_profile_subset(
        self,
        subset: Dict[str, List[Dict[str, Any]]],
        *,
        requested_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        normalized_requested = normalize_bucket_track_pairs(requested_pairs or [])
        requested_set = set(normalized_requested)

        def _summarize_group(items: List[Dict[str, Any]], *, group_name: str) -> Dict[str, Any]:
            track_counter: Counter[Tuple[str, str]] = Counter()
            requested_count = 0
            extra_same_bucket_count = 0
            for item in items:
                pair = (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
                track_counter[pair] += 1
                if requested_set:
                    if pair in requested_set:
                        requested_count += 1
                    else:
                        extra_same_bucket_count += 1
            tracks = [
                {
                    "bucket": bucket,
                    "local_track": local_track,
                    "count": count,
                    "requested_track": (bucket, local_track) in requested_set if requested_set else None,
                }
                for (bucket, local_track), count in sorted(track_counter.items())
            ]
            return {
                f"{group_name}_count": len(items),
                f"{group_name}_requested_track_item_count": requested_count if requested_set else None,
                f"{group_name}_extra_same_bucket_item_count": extra_same_bucket_count if requested_set else None,
                f"{group_name}_by_track": tracks,
            }

        summary: Dict[str, Any] = {
            "requested_track_pairs": [
                {"bucket": bucket, "local_track": local_track}
                for bucket, local_track in normalized_requested
            ],
            "requested_track_pair_count": len(normalized_requested),
        }
        summary.update(_summarize_group(subset.get("active_items", []), group_name="active"))
        summary.update(_summarize_group(subset.get("stale_items", []), group_name="stale"))

        unknown_items = subset.get("unknown_items", [])
        unknown_tracks = [
            {
                "bucket": str(item.get("bucket", "")).strip(),
                "local_track": str(item.get("local_track", "")).strip(),
                "requested_track": (
                    (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip()) in requested_set
                    if requested_set
                    else None
                ),
            }
            for item in unknown_items
        ]
        summary["unknown_count"] = len(unknown_items)
        summary["unknown_by_track"] = unknown_tracks
        return summary

    def _build_answer_input_summary(
        self,
        *,
        parsed: QueryParse,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
        constraints: Optional[ConstraintSet],
        basis_recovery_hint: Optional[Dict[str, Any]] = None,
        basis_recovery_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        peripheral_context = relevant_context.get("peripheral_context", {})
        return {
            "intent_type": parsed.intent_type,
            "target_bucket_tracks": [ref.to_dict() for ref in parsed.target_bucket_tracks],
            "old_premise_focus": parsed.old_premise_focus,
            "basis_recovery_focus": parsed.basis_recovery_focus,
            "decision_basis_focus": parsed.decision_basis_focus,
            "requested_action": parsed.requested_action,
            "action_goal": parsed.action_goal,
            "action_object": parsed.action_object,
            "action_options": parsed.action_options,
            "hidden_old_setup": parsed.hidden_old_setup,
            "secondary_surface_details": parsed.secondary_surface_details,
            "query_assumptions": parsed.query_assumptions,
            "option_comparison_requires_viability_check_first": parsed.option_comparison_requires_viability_check_first,
            "presuppositions": parsed.presuppositions,
            "query_readout_mode": relevant_context.get("query_readout_mode", ""),
            "premise_conflict_subset_summary": relevant_context.get("premise_conflict_subset_summary", {}),
            "primary_hit_count": len(relevant_context.get("topk_primary_hits", [])),
            "primary_hit_ids": [
                f"{item.get('source', '')}:{item.get('id', '')}"
                for item in relevant_context.get("topk_primary_hits", [])
            ],
            "verdict_status": verdict.status,
            "verdict_confidence": verdict.confidence,
            "old_premise_safe": verdict.old_premise_safe,
            "supporting_active_item_ids": verdict.supporting_active_item_ids,
            "supporting_active_item_values": verdict.supporting_active_item_values,
            "outdated_item_ids": verdict.outdated_item_ids,
            "outdated_item_values": verdict.outdated_item_values,
            "usable_current_basis_item_ids": verdict.usable_current_basis_item_ids,
            "usable_current_basis_item_values": verdict.usable_current_basis_item_values,
            "decisive_basis_item_ids": list(relevant_context.get("decisive_basis_item_ids", []) or []),
            "secondary_basis_item_ids": list(relevant_context.get("secondary_basis_item_ids", []) or []),
            "blocked_old_basis_item_ids": verdict.blocked_old_basis_item_ids,
            "blocked_old_basis_item_values": verdict.blocked_old_basis_item_values,
            "historical_but_overridden_item_ids": verdict.historical_but_overridden_item_ids,
            "historical_but_overridden_item_values": verdict.historical_but_overridden_item_values,
            "unknown_tracks": verdict.unknown_tracks,
            "primary_hit_statuses": [
                {
                    "source": item.get("source", ""),
                    "source_detail": item.get("source_detail", ""),
                    "id": item.get("id", ""),
                    "bucket": item.get("bucket", ""),
                    "local_track": item.get("local_track", ""),
                }
                for item in relevant_context.get("topk_primary_hits", [])
            ],
            "hit_bundle_count": len(relevant_context.get("topk_hit_bundles", [])),
            "collapsed_related_hit_count": sum(
                len(bundle.get("collapsed_related_hits", []))
                for bundle in relevant_context.get("topk_hit_bundles", [])
            ),
            "verdict_conditioned_hit_bundle_counts": {
                key: len(value)
                for key, value in relevant_context.get("verdict_conditioned_hit_bundles", {}).items()
                if isinstance(value, list)
            },
            "decisive_basis_bundle_count": len(relevant_context.get("decisive_basis_bundles", []) or []),
            "secondary_basis_bundle_count": len(relevant_context.get("secondary_basis_bundles", []) or []),
            "action_readiness": relevant_context.get("action_readiness", {}),
            "action_decision_mode": relevant_context.get("action_decision_mode", {}),
            "decisive_gate_bundle_count": len(relevant_context.get("decisive_gate_bundles", []) or []),
            "blocker_bundle_count": len(relevant_context.get("blocker_bundles", []) or []),
            "supporting_basis_bundle_count": len(relevant_context.get("supporting_basis_bundles", []) or []),
            "peripheral_context_counts": {
                "mixed_query_topk_results": len(peripheral_context.get("mixed_query_topk_results", [])),
            },
            "basis_recovery_hint": basis_recovery_hint or {},
            "basis_recovery_result_summary": {
                "needs_basis_recovery": bool((basis_recovery_result or {}).get("needs_basis_recovery", False)),
                "recovered_basis_item_ids": list((basis_recovery_result or {}).get("recovered_basis_item_ids", [])),
                "recovered_basis_fact_count": len((basis_recovery_result or {}).get("recovered_basis_facts", [])),
                "basis_confidence": float((basis_recovery_result or {}).get("basis_confidence", 0.0) or 0.0),
                "structured_support_bundle_count": len((basis_recovery_result or {}).get("structured_support_bundles", [])),
                "stale_unknown_hit_bundle_count": len((basis_recovery_result or {}).get("stale_unknown_hit_bundles", [])),
                "global_item_hit_bundle_count": len((basis_recovery_result or {}).get("global_item_hit_bundles", [])),
                "retrieval_branches": sorted(((basis_recovery_result or {}).get("retrieval_branch_hits", {}) or {}).keys()),
            },
            "constraint_counts": {
                "hard_positive_constraints": len(constraints.hard_positive_constraints) if constraints is not None else 0,
                "hard_negative_constraints": len(constraints.hard_negative_constraints) if constraints is not None else 0,
                "unresolved_variables": len(constraints.unresolved_variables) if constraints is not None else 0,
            },
        }

    @staticmethod
    def _unique_preserve_order(values: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for value in values:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _hit_item_id(hit: Dict[str, Any]) -> str:
        payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
        item_id = str(payload.get("item_id", "")).strip()
        if item_id:
            return item_id
        source = str(hit.get("source", "")).strip()
        if source in {"active", "stale"}:
            return str(hit.get("id", "")).strip()
        return ""

    @staticmethod
    def _hit_item_value(hit: Dict[str, Any]) -> str:
        payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
        source = str(hit.get("source", "")).strip()
        if source == "chain" and isinstance(payload, dict):
            linking_delta = payload.get("linking_delta") if isinstance(payload.get("linking_delta"), dict) else {}
            return str(linking_delta.get("proposed_value") or payload.get("value") or "").strip()
        if source in {"delta", "recoverable_delta"} and isinstance(payload, dict):
            return str(payload.get("proposed_value") or payload.get("value") or "").strip()
        if source == "chunk" and isinstance(payload, dict):
            return str(payload.get("text") or payload.get("value") or "").strip()
        if source == "conflict_bundle" and isinstance(payload, dict):
            return str(payload.get("text_preview") or "").strip()
        if source in {"active", "stale", "unknown"} and isinstance(payload, dict):
            return str(payload.get("value") or payload.get("reason") or "").strip()
        return str(hit.get("text_preview", "")).strip()

    @staticmethod
    def _hit_track_pair(hit: Dict[str, Any]) -> Tuple[str, str]:
        return (
            str(hit.get("bucket", "")).strip(),
            str(hit.get("local_track", "")).strip(),
        )
