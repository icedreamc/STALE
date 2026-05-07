from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import PremiseVerdict, QueryParse, maybe_bool, maybe_str_list
from ..memory.schema import is_valid_bucket_track


class VerifierBasisMixin:
    def _sanitize_verdict_payload(
        self,
        *,
        verdict: PremiseVerdict,
        payload: Dict[str, Any],
        allowed_active_item_ids: set[str],
        allowed_hit_item_ids: set[str],
        allowed_unknown_tracks: set[Tuple[str, str]],
    ) -> PremiseVerdict:
        if verdict.status not in {"SUPPORTED", "OUTDATED", "UNRESOLVED"}:
            verdict.status = "UNRESOLVED"
        verdict.confidence = max(0.0, min(1.0, verdict.confidence))
        verdict.old_premise_safe = maybe_bool(payload.get("old_premise_safe", verdict.old_premise_safe), verdict.old_premise_safe)
        verdict.supporting_active_item_ids = [
            item_id
            for item_id in self._unique_preserve_order(verdict.supporting_active_item_ids)
            if item_id in allowed_active_item_ids
        ]
        verdict.supporting_active_item_values = self._unique_preserve_order(verdict.supporting_active_item_values)
        verdict.outdated_item_ids = [
            item_id
            for item_id in self._unique_preserve_order(verdict.outdated_item_ids)
            if item_id in allowed_hit_item_ids
        ]
        verdict.outdated_item_values = self._unique_preserve_order(verdict.outdated_item_values)
        verdict.usable_current_basis_item_ids = [
            item_id
            for item_id in self._unique_preserve_order(verdict.usable_current_basis_item_ids)
            if item_id in allowed_active_item_ids
        ]
        verdict.usable_current_basis_item_values = self._unique_preserve_order(verdict.usable_current_basis_item_values)
        verdict.blocked_old_basis_item_ids = [
            item_id
            for item_id in self._unique_preserve_order(verdict.blocked_old_basis_item_ids)
            if item_id in allowed_hit_item_ids
        ]
        verdict.blocked_old_basis_item_values = self._unique_preserve_order(verdict.blocked_old_basis_item_values)
        verdict.historical_but_overridden_item_ids = [
            item_id
            for item_id in self._unique_preserve_order(verdict.historical_but_overridden_item_ids)
            if item_id in allowed_hit_item_ids
        ]
        verdict.historical_but_overridden_item_values = self._unique_preserve_order(verdict.historical_but_overridden_item_values)
        verdict.unknown_tracks = [
            item
            for item in verdict.unknown_tracks
            if (
                str(item.get("bucket", "")).strip(),
                str(item.get("local_track", "")).strip(),
            ) in allowed_unknown_tracks
        ]
        return verdict

    def _normalize_premise_status(self, *, verdict: PremiseVerdict) -> PremiseVerdict:
        if verdict.status == "OUTDATED":
            verdict.old_premise_safe = False
        if verdict.status == "UNRESOLVED" and verdict.unknown_tracks:
            verdict.old_premise_safe = False
        if (
            verdict.status == "OUTDATED"
            and not verdict.outdated_item_ids
            and not verdict.blocked_old_basis_item_ids
            and not verdict.historical_but_overridden_item_ids
            and not verdict.unknown_tracks
            and not verdict.old_premise_safe
        ):
            verdict.status = "UNRESOLVED"
            verdict.confidence = min(verdict.confidence, 0.65)
            if verdict.reasoning_summary:
                verdict.reasoning_summary += " The store reading suggests the old premise may be unsafe, but exact premise-level failure is not yet grounded strongly enough for an outdated ruling."
            else:
                verdict.reasoning_summary = "The store reading suggests the old premise may be unsafe, but exact premise-level failure is not yet grounded strongly enough for an outdated ruling."
        if (
            verdict.status == "UNRESOLVED"
            and verdict.outdated_item_ids
            and not verdict.supporting_active_item_ids
            and not verdict.old_premise_safe
        ):
            verdict.status = "OUTDATED"
            verdict.confidence = max(verdict.confidence, 0.75)
            if verdict.reasoning_summary:
                verdict.reasoning_summary += " Relevant stale/invalidation evidence exists, so the old premise should be treated as outdated rather than merely unresolved."
            else:
                verdict.reasoning_summary = "Relevant stale/invalidation evidence exists, so the old premise should be treated as outdated rather than merely unresolved."
        return verdict

    def _resolve_basis_permissions(self, *, verdict: PremiseVerdict) -> PremiseVerdict:
        """Separate negative old-premise evidence from current basis."""
        if verdict.status == "OUTDATED":
            blocked_seed = self._unique_preserve_order(
                verdict.blocked_old_basis_item_ids + verdict.outdated_item_ids
            )
            blocked_set = set(blocked_seed)
            current_replacement_seed = [
                item_id
                for item_id in self._unique_preserve_order(
                    verdict.usable_current_basis_item_ids + verdict.supporting_active_item_ids
                )
                if item_id not in blocked_set
            ]
            verdict.blocked_old_basis_item_ids = blocked_seed
            verdict.usable_current_basis_item_ids = current_replacement_seed
            verdict.historical_but_overridden_item_ids = self._unique_preserve_order(
                [
                    item_id
                    for item_id in (
                        verdict.historical_but_overridden_item_ids + verdict.blocked_old_basis_item_ids
                    )
                    if item_id not in set(verdict.usable_current_basis_item_ids)
                ]
            )
        else:
            blocked_set = set(verdict.blocked_old_basis_item_ids)
            verdict.usable_current_basis_item_ids = [
                item_id for item_id in verdict.usable_current_basis_item_ids if item_id not in blocked_set
            ]
        if verdict.status == "UNRESOLVED" and not verdict.old_premise_safe and verdict.outdated_item_ids:
            verdict.status = "OUTDATED"
            verdict.blocked_old_basis_item_ids = self._unique_preserve_order(
                verdict.blocked_old_basis_item_ids + verdict.outdated_item_ids
            )
            verdict.historical_but_overridden_item_ids = self._unique_preserve_order(
                verdict.historical_but_overridden_item_ids + verdict.blocked_old_basis_item_ids
            )
        return verdict

    def _complete_supported_basis_if_needed(self, *, verdict: PremiseVerdict) -> PremiseVerdict:
        if verdict.status == "SUPPORTED" and not verdict.usable_current_basis_item_ids:
            verdict.usable_current_basis_item_ids = list(verdict.supporting_active_item_ids)
        return verdict

    def _derive_corrected_current_facts(
        self,
        *,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
        basis_recovery_result: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        facts: List[str] = []
        basis_recovery_result = basis_recovery_result or {}

        decisive_item_ids = set(
            self._unique_preserve_order(relevant_context.get("decisive_basis_item_ids", []) or [])
        )
        decisive_bundle_sigs = {
            (str(bundle.get("primary_hit", {}).get("source", "")).strip(), str(bundle.get("primary_hit", {}).get("id", "")).strip())
            for bundle in (relevant_context.get("decisive_basis_bundles", []) or [])
            if isinstance(bundle, dict)
        }

        bundles = (relevant_context.get("verdict_conditioned_hit_bundles", {}) or {}).get("usable_current_basis_bundles", [])
        blocked_ids = set(verdict.blocked_old_basis_item_ids)
        for bundle in bundles:
            primary_hit = bundle.get("primary_hit", {})
            item_id = self._hit_item_id(primary_hit)
            if item_id and item_id in blocked_ids:
                continue
            sig = (str(primary_hit.get("source", "")).strip(), str(primary_hit.get("id", "")).strip())
            if decisive_item_ids and item_id not in decisive_item_ids:
                continue
            if decisive_bundle_sigs and sig not in decisive_bundle_sigs:
                continue
            payload = primary_hit.get("payload", {}) if isinstance(primary_hit, dict) else {}
            value = str(payload.get("value", "")).strip()
            if value and value not in facts:
                facts.append(value)

        if not facts and decisive_bundle_sigs:
            for bundle in relevant_context.get("decisive_basis_bundles", []) or []:
                primary_hit = bundle.get("primary_hit", {})
                sig = (str(primary_hit.get("source", "")).strip(), str(primary_hit.get("id", "")).strip())
                if sig not in decisive_bundle_sigs:
                    continue
                payload = primary_hit.get("payload", {}) if isinstance(primary_hit, dict) else {}
                source = str(primary_hit.get("source", "")).strip()
                if source == "chain":
                    linking_delta = payload.get("linking_delta") if isinstance(payload, dict) else None
                    if isinstance(linking_delta, dict):
                        value = str(linking_delta.get("proposed_value", "")).strip()
                    else:
                        value = ""
                    if value and value not in facts:
                        facts.append(value)
                elif source in {"stale", "unknown"}:
                    value = str(payload.get("value") or payload.get("reason") or "").strip()
                    if value and value not in facts:
                        facts.append(value)
                elif source == "conflict_bundle":
                    value = self._conflict_bundle_fact_from_payload(payload)
                    if value and value not in facts:
                        facts.append(value)
                elif source in {"delta", "recoverable_delta"}:
                    value = str(payload.get("proposed_value") or payload.get("value") or "").strip()
                    if value and value not in facts:
                        facts.append(value)
                elif source == "chunk":
                    value = str(payload.get("text") or payload.get("value") or "").strip()
                    if value and value not in facts:
                        facts.append(value)

        if not facts and not decisive_item_ids and not decisive_bundle_sigs:
            for fact in maybe_str_list(basis_recovery_result.get("recovered_basis_facts", [])):
                if fact not in facts:
                    facts.append(fact)

        if not facts:
            for key in ["chain_basis_facts", "conflict_bundle_facts"]:
                for fact in maybe_str_list(basis_recovery_result.get(key, [])):
                    if fact not in facts:
                        facts.append(fact)

        if not facts and decisive_item_ids:
            for bundle in bundles:
                primary_hit = bundle.get("primary_hit", {})
                item_id = self._hit_item_id(primary_hit)
                if not item_id or item_id not in decisive_item_ids or item_id in blocked_ids:
                    continue
                payload = primary_hit.get("payload", {}) if isinstance(primary_hit, dict) else {}
                value = str(payload.get("value", "")).strip()
                if value and value not in facts:
                    facts.append(value)

        if not facts and verdict.status == "UNRESOLVED":
            for item in verdict.unknown_tracks:
                bucket = str(item.get("bucket", "")).strip()
                local_track = str(item.get("local_track", "")).strip()
                if is_valid_bucket_track(bucket, local_track):
                    caution = f"Current state is unresolved for {bucket}/{local_track}."
                    if caution not in facts:
                        facts.append(caution)

        return self._unique_preserve_order(facts[:4])

    def _resolve_decisive_basis(
        self,
        *,
        parsed: QueryParse,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
        basis_recovery_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        basis_recovery_result = basis_recovery_result or {}
        conditioned = relevant_context.get("verdict_conditioned_hit_bundles", {}) or {}
        usable_bundles = [
            bundle
            for bundle in conditioned.get("usable_current_basis_bundles", [])
            if isinstance(bundle, dict)
        ]
        recovery_context_bundles: List[Dict[str, Any]] = []
        include_conflict_context = verdict.status in {"OUTDATED", "UNRESOLVED"} or not verdict.old_premise_safe
        recovery_keys = [
            "recovered_current_basis_bundles",
            "structured_support_bundles",
            "chain_basis_bundles",
            "conflict_bundle_bundles",
        ]
        if include_conflict_context:
            recovery_keys.extend(["stale_unknown_conflict_bundles", "unknown_track_bundles"])
        for key in recovery_keys:
            recovery_context_bundles.extend(
                bundle for bundle in conditioned.get(key, []) if isinstance(bundle, dict)
            )

        candidate_bundles: List[Dict[str, Any]] = []
        seen_candidate_sigs: set[Tuple[str, str]] = set()
        for bundle in usable_bundles + recovery_context_bundles:
            primary_hit = bundle.get("primary_hit", {})
            sig = (str(primary_hit.get("source", "")).strip(), str(primary_hit.get("id", "")).strip())
            if sig in seen_candidate_sigs:
                continue
            seen_candidate_sigs.add(sig)
            candidate_bundles.append(bundle)
        if not candidate_bundles:
            return {
                "decisive_basis_item_ids": [],
                "secondary_basis_item_ids": [],
                "decisive_basis_bundles": [],
                "secondary_basis_bundles": [],
            }

        blocked_item_ids = set(
            self._unique_preserve_order(
                list(verdict.blocked_old_basis_item_ids or [])
                + list(verdict.historical_but_overridden_item_ids or [])
            )
        )

        target_track_pairs = {
            (ref.bucket, ref.local_track)
            for ref in parsed.target_bucket_tracks
            if is_valid_bucket_track(ref.bucket, ref.local_track)
        }
        presupposition_pairs = {
            (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
            for item in parsed.presuppositions
            if is_valid_bucket_track(str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
        }
        conflict_pairs = {
            (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
            for item in (basis_recovery_result.get("conflict_tracks", []) or [])
            if is_valid_bucket_track(str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
        }
        conflict_pairs.update(
            {
                (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
                for item in verdict.unknown_tracks
                if is_valid_bucket_track(str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
            }
        )
        recovered_item_ids = set(self._unique_preserve_order(basis_recovery_result.get("recovered_basis_item_ids", [])))
        candidate_bundle_sigs = {
            (
                str(bundle.get("primary_hit", {}).get("source", "")).strip(),
                str(bundle.get("primary_hit", {}).get("id", "")).strip(),
            )
            for bundle in (basis_recovery_result.get("recovered_current_basis_bundles", []) or [])
            if isinstance(bundle, dict)
        }
        surface_details = [
            str(item).strip().lower()
            for item in (parsed.secondary_surface_details or [])
            if str(item).strip()
        ]

        def _bundle_text(bundle: Dict[str, Any]) -> str:
            primary_hit = bundle.get("primary_hit", {})
            payload = primary_hit.get("payload", {}) if isinstance(primary_hit, dict) else {}
            source = str(primary_hit.get("source", "")).strip()
            if source == "chain" and isinstance(payload, dict):
                linking_delta = payload.get("linking_delta") if isinstance(payload.get("linking_delta"), dict) else {}
                return str(linking_delta.get("proposed_value") or payload.get("value") or "").strip()
            if source in {"recoverable_delta", "delta"} and isinstance(payload, dict):
                return str(payload.get("proposed_value") or payload.get("value") or "").strip()
            if source == "chunk" and isinstance(payload, dict):
                return str(payload.get("text") or payload.get("value") or "").strip()
            if source == "conflict_bundle" and isinstance(payload, dict):
                return self._conflict_bundle_fact_from_payload(payload)
            if source in {"active", "stale", "unknown"} and isinstance(payload, dict):
                return str(payload.get("value") or payload.get("reason") or "").strip()
            return str(primary_hit.get("text_preview", "")).strip()

        def _surface_penalty(bundle: Dict[str, Any], *, pair: Tuple[str, str]) -> int:
            text = _bundle_text(bundle).lower()
            if not text:
                return 0
            if pair in target_track_pairs or pair in conflict_pairs:
                return 0
            return 1 if any(detail in text for detail in surface_details) else 0

        def _has_chain_replacement_value(bundle: Dict[str, Any]) -> bool:
            primary_hit = bundle.get("primary_hit", {})
            if str(primary_hit.get("source", "")).strip() != "chain":
                return False
            payload = primary_hit.get("payload", {}) if isinstance(primary_hit, dict) else {}
            if not isinstance(payload, dict):
                return False
            linking_delta = payload.get("linking_delta") if isinstance(payload.get("linking_delta"), dict) else {}
            return bool(str(linking_delta.get("proposed_value", "")).strip())

        def _has_explicit_current_value(bundle: Dict[str, Any]) -> bool:
            primary_hit = bundle.get("primary_hit", {})
            source = str(primary_hit.get("source", "")).strip()
            payload = primary_hit.get("payload", {}) if isinstance(primary_hit, dict) else {}
            if source in {"recoverable_delta", "delta"} and isinstance(payload, dict):
                return bool(str(payload.get("proposed_value") or payload.get("value") or "").strip())
            if source == "chunk" and isinstance(payload, dict):
                return bool(str(payload.get("text") or payload.get("value") or "").strip())
            if source == "conflict_bundle" and isinstance(payload, dict):
                return bool(self._conflict_bundle_fact_from_payload(payload))
            return False

        def _is_decisive_candidate(bundle: Dict[str, Any], *, pair: Tuple[str, str]) -> bool:
            primary_hit = bundle.get("primary_hit", {})
            source = str(primary_hit.get("source", "")).strip()
            source_detail = str(primary_hit.get("source_detail", "")).strip()
            recovery_source = str(bundle.get("basis_recovery_source", "")).strip()
            if recovery_source == "same_bucket":
                return False
            if source == "stale" and pair not in conflict_pairs:
                return False
            if source_detail == "stale_support_link":
                return False
            if _surface_penalty(bundle, pair=pair):
                return False
            if pair in target_track_pairs or pair in conflict_pairs:
                return True
            if _has_chain_replacement_value(bundle) or _has_explicit_current_value(bundle):
                return True
            sig = (
                str(primary_hit.get("source", "")).strip(),
                str(primary_hit.get("id", "")).strip(),
            )
            if sig in candidate_bundle_sigs:
                return True
            item_id = self._hit_item_id(primary_hit)
            if source == "active" and item_id and item_id in recovered_item_ids:
                return True
            return False

        def _bundle_rank(bundle: Dict[str, Any]) -> Tuple[Any, ...]:
            primary_hit = bundle.get("primary_hit", {})
            pair = self._hit_track_pair(primary_hit)
            item_id = self._hit_item_id(primary_hit)
            source = str(primary_hit.get("source", "")).strip()
            source_detail = str(primary_hit.get("source_detail", "")).strip()
            recovery_source = str(bundle.get("basis_recovery_source", "")).strip()
            recovery_priority = {
                "conflict_bundle": 6,
                "stale_unknown_conflict": 3,
                "global_item_value": 3,
                "topk": 1,
                "same_bucket": 0,
            }.get(recovery_source, 2)
            status_priority = {
                "stale_successor_link": 4,
                "strong_active": 3,
                "weak_active": 2,
                "unknown_current": 0,
                "track_conflict_bundle": 2,
                "stale": 1,
                "stale_support_link": -1,
            }.get(source_detail or source, 0)
            surface_penalty = _surface_penalty(bundle, pair=pair)
            return (
                recovery_priority,
                1 if _has_chain_replacement_value(bundle) else 0,
                1 if _has_explicit_current_value(bundle) else 0,
                1 if pair in target_track_pairs else 0,
                1 if pair in conflict_pairs else 0,
                1 if (
                    str(primary_hit.get("source", "")).strip(),
                    str(primary_hit.get("id", "")).strip(),
                ) in candidate_bundle_sigs else 0,
                1 if item_id and item_id in recovered_item_ids else 0,
                1 if pair in presupposition_pairs else 0,
                status_priority,
                -surface_penalty,
                float(primary_hit.get("score", 0.0) or 0.0),
                -int(primary_hit.get("rank_index", 10_000) or 10_000),
            )

        ranked_bundles = sorted(candidate_bundles, key=_bundle_rank, reverse=True)
        decisive_bundles: List[Dict[str, Any]] = []
        secondary_bundles: List[Dict[str, Any]] = []
        seen_pairs: set[Tuple[str, str]] = set()
        seen_item_ids: set[str] = set()

        for bundle in ranked_bundles:
            primary_hit = bundle.get("primary_hit", {})
            item_id = self._hit_item_id(primary_hit)
            pair = self._hit_track_pair(primary_hit)
            if item_id and item_id in blocked_item_ids:
                secondary_bundles.append(bundle)
                continue
            if item_id and item_id in seen_item_ids:
                secondary_bundles.append(bundle)
                continue
            if is_valid_bucket_track(*pair) and pair in seen_pairs:
                secondary_bundles.append(bundle)
                continue
            if not _is_decisive_candidate(bundle, pair=pair):
                secondary_bundles.append(bundle)
                continue
            decisive_bundles.append(bundle)
            if item_id:
                seen_item_ids.add(item_id)
            if is_valid_bucket_track(*pair):
                seen_pairs.add(pair)
            if len(decisive_bundles) >= 5:
                break

        if not decisive_bundles:
            for bundle in ranked_bundles[:5]:
                primary_hit = bundle.get("primary_hit", {})
                item_id = self._hit_item_id(primary_hit)
                pair = self._hit_track_pair(primary_hit)
                if item_id and item_id in blocked_item_ids:
                    continue
                if item_id and item_id in seen_item_ids:
                    continue
                if is_valid_bucket_track(*pair) and pair in seen_pairs:
                    continue
                decisive_bundles.append(bundle)
                if item_id:
                    seen_item_ids.add(item_id)
                if is_valid_bucket_track(*pair):
                    seen_pairs.add(pair)
                if len(decisive_bundles) >= 5:
                    break

        for bundle in ranked_bundles:
            if bundle in decisive_bundles:
                continue
            secondary_bundles.append(bundle)

        decisive_ids = self._unique_preserve_order(
            [
                self._hit_item_id(bundle.get("primary_hit", {}))
                for bundle in decisive_bundles
                if self._hit_item_id(bundle.get("primary_hit", {}))
            ]
        )
        secondary_ids = self._unique_preserve_order(
            [
                self._hit_item_id(bundle.get("primary_hit", {}))
                for bundle in secondary_bundles
                if self._hit_item_id(bundle.get("primary_hit", {}))
                and self._hit_item_id(bundle.get("primary_hit", {})) not in decisive_ids
            ]
        )
        return {
            "decisive_basis_item_ids": decisive_ids,
            "secondary_basis_item_ids": secondary_ids,
            "decisive_basis_bundles": decisive_bundles,
            "secondary_basis_bundles": secondary_bundles,
        }
