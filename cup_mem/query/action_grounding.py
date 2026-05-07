from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import ActionReadinessVerdict, PremiseVerdict, QueryParse, build_action_readiness_verdict
from ..prompt_lib.templates import ACTION_GROUNDING_RESOLVER_PROMPT
from ..memory.schema import is_valid_bucket_track


class ActionGroundingMixin:
    @staticmethod
    def _bundle_ref_id(bundle: Dict[str, Any]) -> str:
        primary_hit = bundle.get("primary_hit", {}) if isinstance(bundle, dict) else {}
        source = str(primary_hit.get("source", "")).strip()
        hit_id = str(primary_hit.get("id", "")).strip()
        if source and hit_id:
            return f"{source}:{hit_id}"
        return str(bundle.get("bundle_id", "")).strip()

    def _bundle_text_for_action(self, bundle: Dict[str, Any]) -> str:
        primary_hit = bundle.get("primary_hit", {}) if isinstance(bundle, dict) else {}
        payload = primary_hit.get("payload", {}) if isinstance(primary_hit, dict) else {}
        source = str(primary_hit.get("source", "")).strip()
        if source == "chain" and isinstance(payload, dict):
            linking_delta = payload.get("linking_delta") if isinstance(payload.get("linking_delta"), dict) else {}
            return str(linking_delta.get("proposed_value") or payload.get("value") or "").strip()
        if source == "unknown" and isinstance(payload, dict):
            return self.unknown_item_text(payload)
        if source in {"active", "stale"} and isinstance(payload, dict):
            return self.profile_item_text(payload)
        if source == "delta" and isinstance(payload, dict):
            return (
                f"bucket: {payload.get('bucket', '')}\n"
                f"local_track: {payload.get('local_track', '')}\n"
                f"proposed_value: {payload.get('proposed_value', '')}"
            )
        if source == "recoverable_delta" and isinstance(payload, dict):
            return (
                f"bucket: {payload.get('bucket', '')}\n"
                f"local_track: {payload.get('local_track', '')}\n"
                f"proposed_value: {payload.get('proposed_value', payload.get('value', ''))}\n"
                f"source_detail: {primary_hit.get('source_detail', '')}"
            )
        if source == "chunk" and isinstance(payload, dict):
            return str(payload.get("text", "")).strip()
        if source == "conflict_bundle" and isinstance(payload, dict):
            return self._conflict_bundle_fact_from_payload(payload)
        return str(payload.get("value") or primary_hit.get("text_preview") or "").strip() if isinstance(payload, dict) else ""

    def _prepare_action_candidate_bundles(
        self,
        *,
        relevant_context: Dict[str, Any],
        max_bundles: int = 10,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        conditioned = relevant_context.get("verdict_conditioned_hit_bundles", {}) or {}
        candidate_sources = [
            ("decisive_basis", relevant_context.get("decisive_basis_bundles", []) or []),
            ("secondary_basis", relevant_context.get("secondary_basis_bundles", []) or []),
            ("blocker", relevant_context.get("blocker_bundles", []) or []),
            ("chain", conditioned.get("chain_basis_bundles", []) or []),
            ("conflict_bundle", conditioned.get("conflict_bundle_bundles", []) or []),
            ("stale_unknown", conditioned.get("stale_unknown_conflict_bundles", []) or []),
            ("unknown_track", conditioned.get("unknown_track_bundles", []) or []),
        ]
        prepared: List[Dict[str, Any]] = []
        lookup: Dict[str, Dict[str, Any]] = {}
        seen: set[str] = set()
        for role, bundles in candidate_sources:
            for bundle in bundles:
                if not isinstance(bundle, dict):
                    continue
                ref_id = self._bundle_ref_id(bundle)
                if not ref_id or ref_id in seen:
                    continue
                seen.add(ref_id)
                primary_hit = bundle.get("primary_hit", {})
                entry = {
                    "bundle_id": ref_id,
                    "candidate_role": role,
                    "basis_recovery_source": bundle.get("basis_recovery_source", ""),
                    "hit_ref": self._hit_ref(primary_hit),
                    "status_label": self._primary_hit_status_label(primary_hit),
                    "text": self._bundle_text_for_action(bundle),
                    "chain_view": bundle.get("chain_view", {}),
                }
                prepared.append(entry)
                lookup[ref_id] = bundle
        prepared.sort(
            key=lambda entry: self._bundle_current_frontload_key(
                lookup.get(entry["bundle_id"], {}),
                role=str(entry.get("candidate_role", "")).strip(),
            )
        )
        return prepared[:max_bundles], lookup

    def _blocked_action_grounding_refs(
        self,
        *,
        verdict: PremiseVerdict,
        bundle_lookup: Dict[str, Dict[str, Any]],
    ) -> Tuple[set[str], List[str]]:
        blocked_item_ids = set(
            self._unique_preserve_order(
                list(verdict.blocked_old_basis_item_ids or [])
                + list(verdict.historical_but_overridden_item_ids or [])
            )
        )
        blocked_bundle_ids: set[str] = set()
        blocked_fact_texts: List[str] = []
        for bundle_id, bundle in bundle_lookup.items():
            primary_hit = bundle.get("primary_hit", {}) if isinstance(bundle, dict) else {}
            item_id = self._hit_item_id(primary_hit)
            if item_id and item_id in blocked_item_ids:
                blocked_bundle_ids.add(bundle_id)
                text = self._bundle_text_for_action(bundle)
                if text:
                    blocked_fact_texts.append(text)
        return blocked_bundle_ids, blocked_fact_texts

    def _sanitize_action_readiness(
        self,
        *,
        readiness: ActionReadinessVerdict,
        valid_bundle_ids: set[str],
        blocked_bundle_ids: Optional[set[str]] = None,
        blocked_fact_texts: Optional[List[str]] = None,
        premise_verdict: Optional[PremiseVerdict] = None,
    ) -> ActionReadinessVerdict:
        if readiness.status not in {"PROCEED", "PROCEED_WITH_ADJUSTMENT", "BLOCK_AND_REFRAME", "ASK_MIN_CLARIFICATION"}:
            readiness.status = "ASK_MIN_CLARIFICATION"
        readiness.confidence = max(0.0, min(1.0, readiness.confidence))
        blocked_bundle_ids = blocked_bundle_ids or set()

        def _filter(ids: List[str]) -> List[str]:
            return [
                item
                for item in self._unique_preserve_order(ids)
                if item in valid_bundle_ids and item not in blocked_bundle_ids
            ]

        readiness.decisive_gate_bundle_ids = _filter(readiness.decisive_gate_bundle_ids)
        readiness.blocker_bundle_ids = _filter(readiness.blocker_bundle_ids)
        readiness.supporting_basis_bundle_ids = _filter(readiness.supporting_basis_bundle_ids)
        readiness.rejected_nearby_bundle_ids = _filter(readiness.rejected_nearby_bundle_ids)
        readiness.decisive_gate_bundle_ids = readiness.decisive_gate_bundle_ids[:2]
        readiness.blocker_bundle_ids = readiness.blocker_bundle_ids[:2]
        readiness.supporting_basis_bundle_ids = readiness.supporting_basis_bundle_ids[:1]
        blocked_norms = [
            " ".join(str(text or "").lower().split())
            for text in (blocked_fact_texts or [])
            if str(text or "").strip()
        ]
        safe_facts: List[str] = []
        for fact in self._unique_preserve_order(readiness.corrected_current_facts):
            norm = " ".join(str(fact or "").lower().split())
            if not norm:
                continue
            if any(norm == blocked or (blocked and blocked in norm) or (norm and norm in blocked) for blocked in blocked_norms):
                continue
            safe_facts.append(fact)
        readiness.corrected_current_facts = safe_facts[:4]

        if (
            readiness.status in {"PROCEED", "PROCEED_WITH_ADJUSTMENT"}
            and not readiness.decisive_gate_bundle_ids
            and not readiness.blocker_bundle_ids
            and not readiness.supporting_basis_bundle_ids
            and not readiness.corrected_current_facts
        ):
            premise_unsafe = premise_verdict is not None and (
                premise_verdict.status == "OUTDATED" or not premise_verdict.old_premise_safe
            )
            readiness.status = "BLOCK_AND_REFRAME" if premise_unsafe else "ASK_MIN_CLARIFICATION"
            readiness.confidence = min(readiness.confidence, 0.5)
            readiness.reasoning_summary = (
                readiness.reasoning_summary
                or "No non-blocked current-memory basis remains after removing rejected old-premise grounding."
            )
        return readiness

    def _augment_action_readiness_basis(
        self,
        *,
        readiness: ActionReadinessVerdict,
        candidate_bundles: List[Dict[str, Any]],
    ) -> ActionReadinessVerdict:
        if not candidate_bundles:
            return readiness

        candidate_by_id = {
            str(bundle.get("bundle_id", "")).strip(): bundle
            for bundle in candidate_bundles
            if str(bundle.get("bundle_id", "")).strip()
        }
        decisive_ids = self._unique_preserve_order(readiness.decisive_gate_bundle_ids)
        blocker_ids = set(self._unique_preserve_order(readiness.blocker_bundle_ids))
        supporting_ids = self._unique_preserve_order(readiness.supporting_basis_bundle_ids)
        rejected_ids = set(self._unique_preserve_order(readiness.rejected_nearby_bundle_ids))
        selected_ids = set(decisive_ids) | blocker_ids | set(supporting_ids) | rejected_ids

        def _pair(bundle_id: str) -> Tuple[str, str]:
            bundle = candidate_by_id.get(bundle_id, {})
            hit_ref = bundle.get("hit_ref", {}) if isinstance(bundle, dict) else {}
            return (
                str(hit_ref.get("bucket", "")).strip(),
                str(hit_ref.get("local_track", "")).strip(),
            )

        decisive_pairs = {
            pair
            for pair in (_pair(bundle_id) for bundle_id in decisive_ids)
            if is_valid_bucket_track(*pair)
        }

        if len(decisive_ids) <= 1:
            supplemental_ids: List[str] = []
            for bundle in candidate_bundles:
                bundle_id = str(bundle.get("bundle_id", "")).strip()
                if not bundle_id or bundle_id in selected_ids:
                    continue
                role = str(bundle.get("candidate_role", "")).strip()
                if role not in {"decisive_basis", "secondary_basis", "chain", "conflict_bundle", "stale_unknown", "unknown_track"}:
                    continue
                pair = _pair(bundle_id)
                if decisive_pairs and is_valid_bucket_track(*pair) and pair not in decisive_pairs:
                    text = str(bundle.get("text", "")).strip()
                    status_label = str(bundle.get("status_label", "")).strip().lower()
                    if not text and not status_label:
                        continue
                supplemental_ids.append(bundle_id)
                if len(supplemental_ids) >= 2:
                    break
            if supplemental_ids:
                readiness.supporting_basis_bundle_ids = self._unique_preserve_order(
                    supporting_ids + supplemental_ids
                )

        return readiness

    def _judge_action_readiness(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
        basis_recovery_result: Optional[Dict[str, Any]] = None,
        query_label: str = "",
    ) -> Dict[str, Any]:
        candidate_bundles, bundle_lookup = self._prepare_action_candidate_bundles(
            relevant_context=relevant_context,
        )
        if not candidate_bundles:
            readiness = ActionReadinessVerdict(
                status="ASK_MIN_CLARIFICATION",
                confidence=0.2,
                reasoning_summary="No bounded current-memory candidate was available to ground the requested action.",
            )
            return {"raw_payload": readiness.to_dict(), "action_readiness": readiness, "candidate_bundles": [], "bundle_lookup": {}}
        payload = self._llm_call_json(
            phase="action_grounding_resolver",
            system_prompt=ACTION_GROUNDING_RESOLVER_PROMPT,
            user_payload=json.dumps(
                {
                    "query": query_text,
                    "parsed_query": self._build_action_parsed_query_payload(parsed),
                    "premise_verdict": verdict.to_dict(),
                    "candidate_bundles": candidate_bundles,
                },
                ensure_ascii=False,
                indent=2,
            ),
            query_label=query_label,
        )
        readiness = build_action_readiness_verdict(payload)
        blocked_bundle_ids, blocked_fact_texts = self._blocked_action_grounding_refs(
            verdict=verdict,
            bundle_lookup=bundle_lookup,
        )
        readiness = self._sanitize_action_readiness(
            readiness=readiness,
            valid_bundle_ids=set(bundle_lookup.keys()),
            blocked_bundle_ids=blocked_bundle_ids,
            blocked_fact_texts=blocked_fact_texts,
            premise_verdict=verdict,
        )
        readiness = self._augment_action_readiness_basis(
            readiness=readiness,
            candidate_bundles=candidate_bundles,
        )
        return {
            "raw_payload": payload,
            "action_readiness": readiness,
            "candidate_bundles": candidate_bundles,
            "bundle_lookup": bundle_lookup,
        }

    def _apply_action_readiness(
        self,
        *,
        readiness: ActionReadinessVerdict,
        action_result: Dict[str, Any],
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
    ) -> PremiseVerdict:
        bundle_lookup = action_result.get("bundle_lookup", {}) if isinstance(action_result, dict) else {}

        def _bundles_for(ids: List[str]) -> List[Dict[str, Any]]:
            return [bundle_lookup[item_id] for item_id in ids if item_id in bundle_lookup]

        decisive_bundles = _bundles_for(readiness.decisive_gate_bundle_ids)
        blocker_bundles = _bundles_for(readiness.blocker_bundle_ids)
        supporting_bundles = _bundles_for(readiness.supporting_basis_bundle_ids)
        rejected_bundles = _bundles_for(readiness.rejected_nearby_bundle_ids)

        relevant_context["action_readiness"] = readiness.to_dict()
        relevant_context["action_readiness_candidate_bundles"] = action_result.get("candidate_bundles", [])
        relevant_context["decisive_gate_bundles"] = decisive_bundles
        relevant_context["blocker_bundles"] = blocker_bundles
        relevant_context["supporting_basis_bundles"] = supporting_bundles
        relevant_context["rejected_nearby_bundles"] = rejected_bundles

        active_ids: List[str] = []
        for bundle in decisive_bundles + supporting_bundles:
            primary_hit = bundle.get("primary_hit", {})
            item_id = self._hit_item_id(primary_hit)
            if item_id and str(primary_hit.get("source", "")).strip() == "active" and item_id not in verdict.blocked_old_basis_item_ids:
                active_ids.append(item_id)
        if active_ids:
            verdict.usable_current_basis_item_ids = self._unique_preserve_order(
                verdict.usable_current_basis_item_ids + active_ids
            )
        if readiness.corrected_current_facts:
            verdict.corrected_current_facts = self._unique_preserve_order(
                readiness.corrected_current_facts + verdict.corrected_current_facts
            )[:4]
        return verdict
