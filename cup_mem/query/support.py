from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import (
    ActionReadinessVerdict,
    BucketTrackRef,
    ConstraintSet,
    PremiseVerdict,
    QueryParse,
    build_answer_result,
    build_constraint_set,
    build_premise_verdict,
    build_query_parse,
    maybe_str_list,
)
from ..prompt_lib.templates import ANSWER_COMPOSER_PROMPT, CONSTRAINT_BUILDER_PROMPT, PREMISE_VERIFIER_PROMPT, QUERY_PARSER_PROMPT
from ..memory.schema import is_valid_bucket_track


class QuerySupportMixin:
    def _bundle_grounding_text(self, bundle: Dict[str, Any]) -> str:
        primary_hit = bundle.get("primary_hit", {}) if isinstance(bundle, dict) else {}
        payload = primary_hit.get("payload", {}) if isinstance(primary_hit, dict) else {}
        source = str(primary_hit.get("source", "")).strip()
        if source == "chain" and isinstance(payload, dict):
            linking_delta = payload.get("linking_delta") if isinstance(payload.get("linking_delta"), dict) else {}
            return str(
                linking_delta.get("proposed_value")
                or payload.get("value")
                or ""
            ).strip()
        if source in {"delta", "recoverable_delta"} and isinstance(payload, dict):
            return str(payload.get("proposed_value") or payload.get("value") or "").strip()
        if source == "chunk" and isinstance(payload, dict):
            return str(payload.get("text") or payload.get("value") or "").strip()
        if source == "conflict_bundle" and isinstance(payload, dict):
            return self._conflict_bundle_fact_from_payload(payload)
        if source in {"active", "stale", "unknown"} and isinstance(payload, dict):
            return str(payload.get("value") or payload.get("reason") or "").strip()
        return str(primary_hit.get("text_preview", "")).strip()

    def _build_item_value_lookup(
        self,
        *,
        relevant_context: Dict[str, Any],
        basis_recovery_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        lookup: Dict[str, str] = {}

        def _add_bundle(bundle: Dict[str, Any]) -> None:
            if not isinstance(bundle, dict):
                return
            primary_hit = bundle.get("primary_hit", {})
            item_id = self._hit_item_id(primary_hit)
            value = self._bundle_grounding_text(bundle)
            if item_id and value and item_id not in lookup:
                lookup[item_id] = value

        for bundle in relevant_context.get("topk_hit_bundles", []) or []:
            _add_bundle(bundle)
        conditioned = relevant_context.get("verdict_conditioned_hit_bundles", {}) or {}
        for bundles in conditioned.values():
            if isinstance(bundles, list):
                for bundle in bundles:
                    _add_bundle(bundle)
        for bundle in (basis_recovery_result or {}).get("recovered_current_basis_bundles", []) or []:
            _add_bundle(bundle)
        for key in [
            "chain_basis_bundles",
            "conflict_bundle_bundles",
            "stale_unknown_hit_bundles",
            "global_item_hit_bundles",
        ]:
            for bundle in (basis_recovery_result or {}).get(key, []) or []:
                _add_bundle(bundle)
        return lookup

    def _attach_verdict_value_views(
        self,
        *,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
        basis_recovery_result: Optional[Dict[str, Any]] = None,
    ) -> PremiseVerdict:
        lookup = self._build_item_value_lookup(
            relevant_context=relevant_context,
            basis_recovery_result=basis_recovery_result,
        )

        def _values_from_ids(ids: List[str]) -> List[str]:
            values: List[str] = []
            for item_id in self._unique_preserve_order(ids):
                value = lookup.get(item_id, "")
                if value and value not in values:
                    values.append(value)
            return values

        verdict.supporting_active_item_values = _values_from_ids(verdict.supporting_active_item_ids)
        verdict.outdated_item_values = _values_from_ids(verdict.outdated_item_ids)
        verdict.usable_current_basis_item_values = _values_from_ids(verdict.usable_current_basis_item_ids)
        verdict.blocked_old_basis_item_values = _values_from_ids(verdict.blocked_old_basis_item_ids)
        verdict.historical_but_overridden_item_values = _values_from_ids(verdict.historical_but_overridden_item_ids)
        return verdict

    def _bundle_current_frontload_key(
        self,
        bundle: Dict[str, Any],
        *,
        role: str = "",
    ) -> Tuple[int, int, int, int, int]:
        primary_hit = bundle.get("primary_hit", {}) if isinstance(bundle, dict) else {}
        payload = primary_hit.get("payload", {}) if isinstance(primary_hit, dict) else {}
        source = str(primary_hit.get("source", "")).strip()
        source_detail = str(primary_hit.get("source_detail", "")).strip()
        basis_recovery_source = str(bundle.get("basis_recovery_source", "")).strip()
        text = self._bundle_grounding_text(bundle)

        explicit_current_rank = 5
        if source == "chain" and isinstance(payload, dict):
            linking_delta = payload.get("linking_delta") if isinstance(payload.get("linking_delta"), dict) else {}
            if str(linking_delta.get("proposed_value", "")).strip():
                explicit_current_rank = 0
            elif text:
                explicit_current_rank = 2
        elif source in {"delta", "recoverable_delta"} and isinstance(payload, dict):
            explicit_current_rank = 0 if str(payload.get("proposed_value") or payload.get("value") or "").strip() else 4
        elif source == "conflict_bundle" and text:
            explicit_current_rank = 0
        elif source in {"chunk", "unknown"} and text:
            explicit_current_rank = 1
        elif source == "active" and text:
            explicit_current_rank = 2
        elif source == "stale" and text:
            explicit_current_rank = 3
        elif text:
            explicit_current_rank = 4

        role_rank = {
            "blocker": 0,
            "decisive_basis": 1,
            "chain": 2,
            "conflict_bundle": 2,
            "supporting_basis": 3,
            "secondary_basis": 3,
            "stale_unknown": 4,
            "unknown_track": 4,
        }.get(role, 6)

        locality_rank = 1
        if basis_recovery_source in {"conflict_track", "requested_track", "unknown_track"}:
            locality_rank = 0
        elif basis_recovery_source == "same_bucket":
            locality_rank = 2

        successor_rank = 0 if source_detail == "stale_successor_link" else 1
        empty_text_rank = 0 if text else 1
        return (
            explicit_current_rank,
            role_rank,
            locality_rank,
            successor_rank,
            empty_text_rank,
        )

    def _frontload_bundles(
        self,
        bundles: List[Dict[str, Any]],
        *,
        role: str = "",
    ) -> List[Dict[str, Any]]:
        ranked: List[Tuple[Tuple[int, int, int, int, int], int, Dict[str, Any]]] = []
        for idx, bundle in enumerate(bundles):
            if not isinstance(bundle, dict):
                continue
            ranked.append((self._bundle_current_frontload_key(bundle, role=role), idx, bundle))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [bundle for _, _, bundle in ranked]

    @staticmethod
    def _build_action_parsed_query_payload(parsed: QueryParse) -> Dict[str, Any]:
        return {
            "intent_type": parsed.intent_type,
            "queried_proposition": parsed.queried_proposition,
            "old_premise_focus": parsed.old_premise_focus,
            "requested_action": parsed.requested_action,
            "action_goal": parsed.action_goal,
            "action_object": parsed.action_object,
            "action_options": list(parsed.action_options or []),
            "hidden_old_setup": parsed.hidden_old_setup,
            "option_comparison_requires_viability_check_first": bool(
                parsed.option_comparison_requires_viability_check_first
            ),
        }

    @staticmethod
    def _build_constraint_parsed_query_payload(parsed: QueryParse) -> Dict[str, Any]:
        return {
            "intent_type": parsed.intent_type,
            "queried_proposition": parsed.queried_proposition,
            "requested_action": parsed.requested_action,
            "action_goal": parsed.action_goal,
            "action_object": parsed.action_object,
            "action_options": list(parsed.action_options or []),
            "option_comparison_requires_viability_check_first": bool(
                parsed.option_comparison_requires_viability_check_first
            ),
        }

    @staticmethod
    def _build_verifier_parsed_query_payload(parsed: QueryParse) -> Dict[str, Any]:
        return {
            "intent_type": parsed.intent_type,
            "target_bucket_tracks": [ref.to_dict() for ref in parsed.target_bucket_tracks],
            "queried_proposition": parsed.queried_proposition,
            "old_premise_focus": parsed.old_premise_focus,
            "basis_recovery_focus": parsed.basis_recovery_focus,
            "decision_basis_focus": parsed.decision_basis_focus,
            "requested_action": parsed.requested_action,
            "hidden_old_setup": parsed.hidden_old_setup,
            "query_assumptions": list(parsed.query_assumptions or []),
            "presuppositions": list(parsed.presuppositions or []),
            "option_comparison_requires_viability_check_first": bool(
                parsed.option_comparison_requires_viability_check_first
            ),
        }

    @staticmethod
    def _build_answer_parsed_query_payload(parsed: QueryParse) -> Dict[str, Any]:
        return {
            "intent_type": parsed.intent_type,
            "queried_proposition": parsed.queried_proposition,
            "old_premise_focus": parsed.old_premise_focus,
            "hidden_old_setup": parsed.hidden_old_setup,
            "requested_action": parsed.requested_action,
            "action_object": parsed.action_object,
            "action_options": list(parsed.action_options or []),
            "option_comparison_requires_viability_check_first": bool(
                parsed.option_comparison_requires_viability_check_first
            ),
        }

    @staticmethod
    def _build_answer_action_readiness_payload(
        action_readiness: Optional[ActionReadinessVerdict],
    ) -> Optional[Dict[str, Any]]:
        if action_readiness is None:
            return None
        payload = action_readiness.to_dict()
        payload["rejected_nearby_bundle_ids"] = []
        return payload

    def _build_canonical_answer_grounding(
        self,
        *,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
        action_readiness: Optional[ActionReadinessVerdict] = None,
    ) -> Dict[str, Any]:
        def _bundle_brief(bundle: Dict[str, Any]) -> Dict[str, Any]:
            primary_hit = bundle.get("primary_hit", {}) if isinstance(bundle, dict) else {}
            return {
                "hit_ref": self._hit_ref(primary_hit),
                "status_label": self._primary_hit_status_label(primary_hit),
                "basis_recovery_source": bundle.get("basis_recovery_source", ""),
                "text": self._bundle_grounding_text(bundle),
            }

        decisive_source_bundles = self._unique_bundle_list(
            list(relevant_context.get("decisive_gate_bundles", []) or [])
            + list(relevant_context.get("decisive_basis_bundles", []) or [])
        )
        decisive_basis = [
            _bundle_brief(bundle)
            for bundle in self._frontload_bundles(
                decisive_source_bundles,
                role="decisive_basis",
            )[:5]
            if isinstance(bundle, dict)
        ]
        supporting_source_bundles = self._unique_bundle_list(
            list(relevant_context.get("supporting_basis_bundles", []) or [])
            + list(relevant_context.get("secondary_basis_bundles", []) or [])
        )
        supporting_basis = [
            _bundle_brief(bundle)
            for bundle in self._frontload_bundles(
                supporting_source_bundles,
                role="supporting_basis",
            )[:3]
            if isinstance(bundle, dict)
        ]
        blocker_basis = [
            _bundle_brief(bundle)
            for bundle in self._frontload_bundles(
                list(relevant_context.get("blocker_bundles", []) or []),
                role="blocker",
            )[:2]
            if isinstance(bundle, dict)
        ]

        summary = {
            "blocker_basis": blocker_basis,
            "decisive_basis": decisive_basis,
            "supporting_basis": supporting_basis,
            "corrected_current_facts": list(
                (action_readiness.corrected_current_facts if action_readiness is not None else verdict.corrected_current_facts)
                or []
            )[:4],
            "old_premise_safe": bool(verdict.old_premise_safe),
            "premise_status": verdict.status,
            "action_status": action_readiness.status if action_readiness is not None else "",
        }
        blocked_reason_summary = self._unique_preserve_order(
            list((action_readiness.corrected_current_facts if action_readiness is not None else []) or [])
        )[:3]
        return self._attach_debug_trace(
            summary,
            section="query",
            fields={"blocked_reason_summary": blocked_reason_summary},
        )

    def _unique_bundle_list(self, bundles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        for bundle in bundles:
            if not isinstance(bundle, dict):
                continue
            primary_hit = bundle.get("primary_hit", {}) if isinstance(bundle, dict) else {}
            sig = (
                str(primary_hit.get("source", "")).strip(),
                str(primary_hit.get("id", "")).strip(),
            )
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(bundle)
        return unique

    def _canonicalize_query_parse(self, parsed: QueryParse, *, query_text: str) -> QueryParse:
        valid_targets = [
            BucketTrackRef(bucket=ref.bucket, local_track=ref.local_track)
            for ref in parsed.target_bucket_tracks
            if is_valid_bucket_track(ref.bucket, ref.local_track)
        ]
        parsed.target_bucket_tracks = valid_targets

        def _normalize_track_dicts(items: List[Dict[str, Any]], *, text_key: str = "text") -> List[Dict[str, Any]]:
            normalized: List[Dict[str, Any]] = []
            for item in items:
                bucket = str(item.get("bucket", "")).strip()
                local_track = str(item.get("local_track", "")).strip()
                text_value = str(item.get(text_key, item.get("text", item.get("focus", "")))).strip()
                if text_value and is_valid_bucket_track(bucket, local_track):
                    normalized.append(
                        {
                            text_key: text_value,
                            "bucket": bucket,
                            "local_track": local_track,
                        }
                    )
            return normalized

        parsed.presuppositions = _normalize_track_dicts(parsed.presuppositions, text_key="text")
        parsed.query_assumptions = _normalize_track_dicts(parsed.query_assumptions, text_key="text")

        if parsed.intent_type not in {"VALIDITY_CHECK", "PREMISE_LADEN_QA", "TASK_PLANNING"}:
            parsed.intent_type = "PREMISE_LADEN_QA"

        for attr in [
            "queried_proposition",
            "old_premise_focus",
            "basis_recovery_focus",
            "decision_basis_focus",
            "requested_action",
            "action_goal",
            "action_object",
            "hidden_old_setup",
        ]:
            setattr(parsed, attr, " ".join(str(getattr(parsed, attr, "") or "").split()))

        parsed.action_options = self._unique_preserve_order(
            " ".join(str(item).split()) for item in parsed.action_options if str(item).strip()
        )
        parsed.secondary_surface_details = self._unique_preserve_order(
            " ".join(str(item).split()) for item in parsed.secondary_surface_details if str(item).strip()
        )
        parsed.option_comparison_requires_viability_check_first = bool(parsed.option_comparison_requires_viability_check_first)

        if not parsed.requested_action and parsed.intent_type == "TASK_PLANNING":
            parsed.requested_action = parsed.queried_proposition or query_text
        if not parsed.action_goal and parsed.intent_type == "TASK_PLANNING":
            parsed.action_goal = parsed.requested_action
        if not parsed.hidden_old_setup:
            parsed.hidden_old_setup = parsed.old_premise_focus
        if not parsed.basis_recovery_focus:
            parsed.basis_recovery_focus = (
                parsed.decision_basis_focus
                or parsed.requested_action
                or parsed.queried_proposition
                or query_text
            )
        return parsed

    def _parse_query(self, query_text: str, *, query_label: str = "") -> QueryParse:
        payload = self._llm_call_json(
            phase="query_parser",
            system_prompt=QUERY_PARSER_PROMPT,
            user_payload=json.dumps(
                {
                    "bucket_track_schema": self.schema_text,
                    "query": query_text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            query_label=query_label,
        )
        parsed = build_query_parse(payload)
        return self._canonicalize_query_parse(parsed, query_text=query_text)

    def _action_decision_mode_info(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        verdict: PremiseVerdict,
        basis_recovery_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Decide whether action gating should run."""
        basis_recovery_hint = basis_recovery_hint or {}
        reasons: List[str] = []
        if parsed.intent_type == "TASK_PLANNING":
            return {
                "enabled": True,
                "mode": "task_planning",
                "reasons": ["intent_task_planning"],
            }

        has_action_fields = bool(
            str(parsed.requested_action or "").strip()
            or str(parsed.action_goal or "").strip()
            or parsed.action_options
            or parsed.option_comparison_requires_viability_check_first
        )
        has_gate_fields = bool(
            str(parsed.basis_recovery_focus or "").strip()
            or str(parsed.hidden_old_setup or "").strip()
        )
        if has_action_fields:
            reasons.append("has_action_fields")
        if has_gate_fields:
            reasons.append("has_gate_fields")

        text = " ".join(str(query_text or "").lower().split())
        action_decision_pattern = re.compile(
            r"\b("
            r"should\s+(?:i|we)\b|"
            r"go\s+ahead|"
            r"accept\b|join\b|sign\b|book\b|buy\b|subscribe\b|get\s+it\b|"
            r"proceed\b|take\b|use\b|choose\b|pick\b|"
            r"reasonable\b|worth\s+it\b|advisable\b|good\s+idea\b|"
            r"thinking\s+of\s+(?:getting|booking|buying|accepting|joining|signing|using)"
            r")"
        )
        lexical_action_decision = bool(action_decision_pattern.search(text))
        if lexical_action_decision:
            reasons.append("lexical_action_decision")
        if parsed.option_comparison_requires_viability_check_first:
            reasons.append("option_viability_first")
        if parsed.action_options:
            reasons.append("has_action_options")

        unsafe_or_gap = (
            verdict.status in {"OUTDATED", "UNRESOLVED"}
            or not verdict.old_premise_safe
            or bool(basis_recovery_hint.get("needs_basis_recovery"))
            or str(basis_recovery_hint.get("basis_gap_reason", "")).strip()
            in {"no_usable_basis", "usable_too_generic", "not_action_relevant", "current_basis_incomplete"}
        )
        if unsafe_or_gap:
            reasons.append("premise_unsafe_or_basis_gap")

        strong_action_signal = bool(
            parsed.intent_type == "TASK_PLANNING"
            or parsed.option_comparison_requires_viability_check_first
            or parsed.action_options
        )
        lexical_action_signal = bool(
            lexical_action_decision
            and (str(parsed.requested_action or "").strip() or str(parsed.action_goal or "").strip())
        )
        if strong_action_signal:
            reasons.append("strong_action_signal")
        if lexical_action_signal:
            reasons.append("lexical_action_signal")

        enabled = bool(
            has_action_fields
            and has_gate_fields
            and unsafe_or_gap
            and (strong_action_signal or lexical_action_signal)
        )
        return {
            "enabled": enabled,
            "mode": "action_decision_limited" if enabled else "off",
            "reasons": reasons,
        }

    @staticmethod
    def _set_action_decision_mode(parsed: QueryParse, enabled: bool) -> None:
        setattr(parsed, "_action_decision_limited_mode", bool(enabled and parsed.intent_type != "TASK_PLANNING"))

    def _collect_relevant_state_context(self, parsed: QueryParse, *, query_text: str) -> Dict[str, Any]:
        raw_topk_primary_hits = self._retrieve_query_topk(parsed=parsed, query_text=query_text)
        raw_topk_status_context = self._build_topk_status_context(raw_topk_primary_hits)
        raw_topk_chain_context = self._build_topk_chain_context(raw_topk_primary_hits)
        topk_hit_bundles, topk_primary_hits = self._build_topk_hit_bundles(
            topk_primary_hits=raw_topk_primary_hits,
            topk_status_context=raw_topk_status_context,
            topk_chain_context=raw_topk_chain_context,
        )

        strong_active_items = self._dedupe_context_items(
            [hit.get("payload", {}) for hit in topk_primary_hits if str(hit.get("source_detail", "")).strip() == "strong_active"],
            key_fn=lambda item: str(item.get("item_id", "")).strip(),
        )
        weak_active_items = self._dedupe_context_items(
            [hit.get("payload", {}) for hit in topk_primary_hits if str(hit.get("source_detail", "")).strip() == "weak_active"],
            key_fn=lambda item: str(item.get("item_id", "")).strip(),
        )
        stale_items = self._dedupe_context_items(
            [hit.get("payload", {}) for hit in topk_primary_hits if str(hit.get("source", "")).strip() == "stale"],
            key_fn=lambda item: str(item.get("item_id", "")).strip(),
        )
        unknown_items = self._dedupe_context_items(
            [hit.get("payload", {}) for hit in topk_primary_hits if str(hit.get("source", "")).strip() == "unknown"],
            key_fn=lambda item: (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip()),
        )

        exclude_refs = {
            (str(hit.get("source", "")).strip(), str(hit.get("id", "")).strip())
            for hit in topk_primary_hits
            if str(hit.get("source", "")).strip() and str(hit.get("id", "")).strip()
        }
        mixed_query_topk_results = self._retrieve_query_peripheral_topk(
            parsed=parsed,
            query_text=query_text,
            exclude_refs=exclude_refs,
            source_allowlist={"stale", "unknown"},
        )
        premise_conflict_subset = self._collect_premise_conflict_subset(
            parsed=parsed,
            query_text=query_text,
            relevant_context={
                "topk_primary_hits": topk_primary_hits,
            },
        )

        return {
            "query_readout_mode": "premise_centered_primary_hits",
            "query_topk_results": topk_primary_hits,
            "topk_primary_hits": topk_primary_hits,
            "topk_hit_bundles": topk_hit_bundles,
            "premise_conflict_subset": premise_conflict_subset["subset"],
            "premise_conflict_subset_seed_tracks": premise_conflict_subset["seed_tracks"],
            "premise_conflict_subset_seed_track_logs": premise_conflict_subset["seed_track_logs"],
            "premise_conflict_subset_summary": premise_conflict_subset["subset_summary"],
            "active_items": strong_active_items + weak_active_items,
            "strong_active_items": strong_active_items,
            "weak_active_items": weak_active_items,
            "stale_items": stale_items,
            "unknown_items": unknown_items,
            "peripheral_context": {
                "mixed_query_topk_results": mixed_query_topk_results,
            },
            "query_topk_summary": {
                "query_top_k": self.thresholds.query_top_k,
                "retrieved_count": len(topk_primary_hits),
                "retrieved_by_source": dict(Counter(item.get("source", "") for item in topk_primary_hits)),
                "peripheral_count": len(mixed_query_topk_results),
            },
        }

    def _fallback_retrieve(self, query_text: str) -> List[Dict[str, Any]]:
        if self.thresholds.fallback_top_k <= 0:
            return []
        candidates: List[Dict[str, Any]] = []
        snapshot = self.store.to_snapshot()
        for item in snapshot["stale_archive"]:
            candidates.append(
                {
                    "source": "stale",
                    "id": item["item_id"],
                    "text": self.profile_item_text(item),
                    "payload": item,
                }
            )
        for item in snapshot["unknown_current"]:
            candidates.append(
                {
                    "source": "unknown",
                    "id": f"{item.get('bucket', '')}/{item.get('local_track', '')}",
                    "text": self.unknown_item_text(item),
                    "payload": item,
                }
            )
        if not candidates:
            return []
        ranked = self.embedding.rank(
            query_text=query_text,
            candidates=candidates,
            text_getter=lambda item: item["text"],
            top_k=self.thresholds.fallback_top_k,
        )
        return [
            {
                "rank_index": rank_index,
                "score": entry["score"],
                "source": entry["item"]["source"],
                "id": entry["item"]["id"],
                "payload": entry["item"]["payload"],
                "bucket": entry["item"]["payload"].get("bucket", ""),
                "local_track": entry["item"]["payload"].get("local_track", ""),
                "session_ids": self._infer_session_ids_from_payload(entry["item"]["payload"]),
                "text_preview": self._preview_text(entry["item"]["text"]),
            }
            for rank_index, entry in enumerate(ranked, start=1)
        ]

    def _verify_premise(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        relevant_context: Dict[str, Any],
        extra_evidence: Optional[List[Dict[str, Any]]] = None,
        second_pass_mode: str = "",
        query_label: str = "",
    ) -> Dict[str, Any]:
        allowed_active_item_ids = {
            self._hit_item_id(hit)
            for hit in relevant_context.get("topk_primary_hits", [])
            if str(hit.get("source", "")).strip() == "active" and self._hit_item_id(hit)
        }
        allowed_hit_item_ids = {
            self._hit_item_id(hit)
            for hit in relevant_context.get("topk_primary_hits", [])
            if self._hit_item_id(hit)
        }
        allowed_unknown_tracks = {
            self._hit_track_pair(hit)
            for hit in relevant_context.get("topk_primary_hits", [])
            if str(hit.get("source", "")).strip() == "unknown"
            and is_valid_bucket_track(*self._hit_track_pair(hit))
        }
        conflict_subset = relevant_context.get("premise_conflict_subset", {}) or {}
        for item in conflict_subset.get("active_items", []) if isinstance(conflict_subset.get("active_items", []), list) else []:
            item_id = str(item.get("item_id", "")).strip()
            if item_id:
                allowed_active_item_ids.add(item_id)
                allowed_hit_item_ids.add(item_id)
        for item in conflict_subset.get("stale_items", []) if isinstance(conflict_subset.get("stale_items", []), list) else []:
            item_id = str(item.get("item_id", "")).strip()
            if item_id:
                allowed_hit_item_ids.add(item_id)
        for item in conflict_subset.get("unknown_items", []) if isinstance(conflict_subset.get("unknown_items", []), list) else []:
            pair = (
                str(item.get("bucket", "")).strip(),
                str(item.get("local_track", "")).strip(),
            )
            if is_valid_bucket_track(*pair):
                allowed_unknown_tracks.add(pair)
        payload = self._llm_call_json(
            phase="premise_verifier_with_evidence" if extra_evidence else "premise_verifier",
            system_prompt=PREMISE_VERIFIER_PROMPT,
            user_payload=json.dumps(
                {
                    "parsed_query": self._build_verifier_parsed_query_payload(parsed),
                    "query": query_text,
                    "query_readout_mode": relevant_context.get("query_readout_mode", ""),
                    "topk_hit_bundles": relevant_context.get("topk_hit_bundles", []),
                    "premise_conflict_subset": relevant_context.get("premise_conflict_subset", {}),
                    "peripheral_context": relevant_context.get("peripheral_context", {}) if extra_evidence else {},
                    "retrieved_evidence": extra_evidence or [],
                    "second_pass_mode": second_pass_mode if extra_evidence else "",
                },
                ensure_ascii=False,
                indent=2,
            ),
            query_label=query_label,
        )
        verdict = build_premise_verdict(payload)
        verdict = self._sanitize_verdict_payload(
            verdict=verdict,
            payload=payload,
            allowed_active_item_ids=allowed_active_item_ids,
            allowed_hit_item_ids=allowed_hit_item_ids,
            allowed_unknown_tracks=allowed_unknown_tracks,
        )
        verdict = self._normalize_premise_status(verdict=verdict)
        verdict = self._resolve_basis_permissions(verdict=verdict)
        verdict = self._complete_supported_basis_if_needed(verdict=verdict)
        verdict = self._attach_verdict_value_views(
            verdict=verdict,
            relevant_context=relevant_context,
            basis_recovery_result=None,
        )
        return {
            "raw_payload": payload,
            "verdict": verdict,
        }

    def _build_constraints(
        self,
        *,
        query_text: str,
        parsed: QueryParse,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
        action_readiness: Optional[ActionReadinessVerdict] = None,
        query_label: str = "",
    ) -> Dict[str, Any]:
        canonical_answer_grounding = self._build_canonical_answer_grounding(
            verdict=verdict,
            relevant_context=relevant_context,
            action_readiness=action_readiness,
        )
        payload = self._llm_call_json(
            phase="constraint_builder",
            system_prompt=CONSTRAINT_BUILDER_PROMPT,
            user_payload=json.dumps(
                {
                    "query": query_text,
                    "parsed_query": self._build_constraint_parsed_query_payload(parsed),
                    "canonical_answer_grounding": canonical_answer_grounding,
                    "premise_verdict": verdict.to_dict(),
                    "action_readiness_verdict": self._build_answer_action_readiness_payload(action_readiness),
                    "query_assumptions": parsed.query_assumptions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            query_label=query_label,
        )
        constraints = build_constraint_set(payload)
        return {
            "raw_payload": payload,
            "constraints": constraints,
        }

    def _compose_answer(
        self,
        *,
        query_text: str,
        parsed: QueryParse,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
        constraints: Optional[ConstraintSet],
        action_readiness: Optional[ActionReadinessVerdict] = None,
        query_label: str = "",
    ) -> Dict[str, Any]:
        canonical_answer_grounding = self._build_canonical_answer_grounding(
            verdict=verdict,
            relevant_context=relevant_context,
            action_readiness=action_readiness,
        )
        payload = self._llm_call_json(
            phase="answer_composer",
            system_prompt=ANSWER_COMPOSER_PROMPT,
            user_payload=json.dumps(
                {
                    "query": query_text,
                    "parsed_query": self._build_answer_parsed_query_payload(parsed),
                    "canonical_answer_grounding": canonical_answer_grounding,
                    "premise_verdict": verdict.to_dict(),
                    "action_readiness_verdict": self._build_answer_action_readiness_payload(action_readiness),
                    "constraint_set": constraints.to_dict() if constraints is not None else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            query_label=query_label,
        )
        answer = build_answer_result(payload)
        return {
            "raw_payload": payload,
            "answer": answer,
        }
