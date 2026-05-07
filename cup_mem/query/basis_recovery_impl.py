from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import PremiseVerdict, QueryParse, build_premise_verdict, maybe_dict_list, maybe_str_list
from ..prompt_lib.templates import QUERY_LANE_FILTER_PROMPT
from ..memory.schema import is_valid_bucket_track


class BasisRecoveryMixin:
    def _build_basis_retrieval_query(self, *, parsed: QueryParse, query_text: str) -> str:
        return self._join_query_parts(
            query_text,
            parsed.queried_proposition,
        )

    @staticmethod
    def _join_query_parts(*parts: str) -> str:
        lines: List[str] = []
        for part in parts:
            normalized = " ".join(str(part or "").split())
            if normalized and normalized not in lines:
                lines.append(normalized)
        return "\n".join(lines)

    def _build_basis_query_variants(self, *, parsed: QueryParse, query_text: str) -> Dict[str, str]:
        return {
            "query_plus_queried_proposition": self._join_query_parts(query_text, parsed.queried_proposition),
        }

    def _lane_hit_ref(self, hit: Dict[str, Any]) -> str:
        return f"{str(hit.get('source', '')).strip()}:{str(hit.get('id', '')).strip()}"

    def _build_lane_filter_premise_context(self, verdict: Optional[PremiseVerdict]) -> Dict[str, Any]:
        if verdict is None:
            return {}
        return {
            "premise_status": verdict.status,
            "old_premise_safe": bool(verdict.old_premise_safe),
            "outdated_item_ids": list(verdict.outdated_item_ids or []),
            "outdated_item_values": list(verdict.outdated_item_values or []),
            "blocked_old_basis_item_ids": list(verdict.blocked_old_basis_item_ids or []),
            "blocked_old_basis_item_values": list(verdict.blocked_old_basis_item_values or []),
        }

    def _filter_lane_hits_with_llm(
        self,
        *,
        lane_name: str,
        parsed: QueryParse,
        query_text: str,
        hits: List[Dict[str, Any]],
        max_hits: int,
        premise_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not hits:
            return [], {"lane_name": lane_name, "skipped": True, "reason": "no_hits"}
        candidate_payload: List[Dict[str, Any]] = []
        hit_by_ref: Dict[str, Dict[str, Any]] = {}
        for hit in hits:
            ref = self._lane_hit_ref(hit)
            if not ref or ref in hit_by_ref:
                continue
            hit_by_ref[ref] = hit
            candidate_payload.append(
                {
                    "ref": ref,
                    "score": hit.get("score"),
                    "source": hit.get("source"),
                    "source_detail": hit.get("source_detail"),
                    "text": hit.get("text_preview"),
                }
            )
        payload = json.dumps(
            {
                "lane_name": lane_name,
                "query_text": query_text,
                "parsed_query": self._build_lane_filter_parsed_query_payload(parsed),
                "premise_context": premise_context or {},
                "candidates": candidate_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
        raw = self._llm_call_json(
            phase="query_lane_filter",
            system_prompt=QUERY_LANE_FILTER_PROMPT,
            user_payload=payload,
            extra_meta={"lane_name": lane_name},
        )
        kept_refs: List[str] = []
        for ref in maybe_str_list(raw.get("kept_refs", [])):
            normalized = str(ref or "").strip()
            if normalized in hit_by_ref and normalized not in kept_refs:
                kept_refs.append(normalized)
        if not kept_refs:
            return [], {
                "lane_name": lane_name,
                "raw_payload": raw,
                "kept_refs": [],
                "candidate_count": len(candidate_payload),
            }
        kept_hits = [hit_by_ref[ref] for ref in kept_refs][: max(1, max_hits)]
        if self._debug_trace_enabled():
            for rank_index, hit in enumerate(kept_hits, start=1):
                self._attach_debug_trace(
                    hit,
                    section="lane_filter",
                    fields={"rank_index": rank_index},
                )
        return kept_hits, {
            "lane_name": lane_name,
            "raw_payload": self._move_aux_fields_to_debug_trace(
                raw,
                section="lane_filter",
                field_names=["useful_facts", "rejected_refs", "reasoning_summary"],
            ),
            "kept_refs": kept_refs[: max(1, max_hits)],
            "candidate_count": len(candidate_payload),
        }

    def _rank_state_candidates(
        self,
        *,
        query_text: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not str(query_text or "").strip() or not candidates:
            return []
        ranked = self.embedding.rank(
            query_text=query_text,
            candidates=candidates,
            text_getter=lambda item: item["retrieval_text"],
            top_k=min(max(1, top_k), len(candidates)),
        )
        hits: List[Dict[str, Any]] = []
        for rank_index, entry in enumerate(ranked, start=1):
            candidate = entry["item"]
            hits.append(
                {
                    "rank_index": rank_index,
                    "score": entry["score"],
                    "source": candidate["source"],
                    "source_detail": candidate.get("source_detail", candidate["source"]),
                    "id": candidate["id"],
                    "payload": candidate["payload"],
                    "bucket": candidate.get("bucket", ""),
                    "local_track": candidate.get("local_track", ""),
                    "session_ids": self._infer_session_ids_from_payload(candidate["payload"]),
                    "text_preview": self._preview_text(candidate["retrieval_text"]),
                }
            )
        return hits

    def _build_state_candidate_from_item(
        self,
        item: Dict[str, Any],
        *,
        source: str,
        source_detail: str,
        retrieval_text: str,
    ) -> Dict[str, Any]:
        return {
            "source": source,
            "source_detail": source_detail,
            "id": str(item.get("item_id", "")).strip() if source in {"active", "stale"} else f"{item.get('bucket', '')}/{item.get('local_track', '')}",
            "payload": item,
            "bucket": str(item.get("bucket", "")).strip(),
            "local_track": str(item.get("local_track", "")).strip(),
            "retrieval_text": retrieval_text,
        }






    @staticmethod
    def _conflict_bundle_fact_from_payload(payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        facts: List[str] = []
        for wrapped in maybe_dict_list(payload.get("conflict_links", payload.get("links", []))):
            linking_delta = wrapped.get("linking_delta", {}) if isinstance(wrapped, dict) else {}
            if isinstance(linking_delta, dict):
                value = str(linking_delta.get("proposed_value", "")).strip()
                if value:
                    facts.append(value)
        for item in maybe_dict_list(payload.get("conflict_unknown_items", payload.get("unknown_items", []))):
            reason = str(item.get("reason", "")).strip()
            if reason:
                facts.append(reason)
        for item in maybe_dict_list(payload.get("conflict_active_items", [])):
            value = str(item.get("value", "")).strip()
            if value:
                facts.append(value)
        if facts:
            return " | ".join(dict.fromkeys(facts[:3]))
        return ""

    def _stale_unknown_candidates(self) -> List[Dict[str, Any]]:
        snapshot = self.store.to_snapshot()
        candidates: List[Dict[str, Any]] = []
        for item in snapshot.get("stale_archive", []):
            retrieval_text = self.profile_item_text(item)
            candidates.append(
                self._build_state_candidate_from_item(
                    item,
                    source="stale",
                    source_detail="stale",
                    retrieval_text=retrieval_text,
                )
            )
        for item in snapshot.get("unknown_current", []):
            candidates.append(
                self._build_state_candidate_from_item(
                    item,
                    source="unknown",
                    source_detail="unknown_current",
                    retrieval_text=self.unknown_item_text(item),
                )
            )
        return candidates


    def _candidate_to_hit(self, candidate: Dict[str, Any], *, score: float = 0.0, rank_index: int = 1) -> Dict[str, Any]:
        payload = candidate["payload"]
        return {
            "rank_index": rank_index,
            "score": score,
            "source": candidate["source"],
            "source_detail": candidate.get("source_detail", candidate["source"]),
            "id": candidate["id"],
            "payload": payload,
            "bucket": candidate.get("bucket", ""),
            "local_track": candidate.get("local_track", ""),
            "session_ids": self._infer_session_ids_from_payload(payload),
            "text_preview": self._preview_text(candidate["retrieval_text"]),
        }

    def _global_item_value_candidates(self) -> List[Dict[str, Any]]:
        snapshot = self.store.to_snapshot()
        candidates: List[Dict[str, Any]] = []
        for items in snapshot.get("active_profile", {}).values():
            for item in items:
                strength = str(item.get("active_strength", "STRONG")).upper()
                candidates.append(
                    self._build_state_candidate_from_item(
                        item,
                        source="active",
                        source_detail="weak_active" if strength == "WEAK" else "strong_active",
                        retrieval_text=f"value: {item.get('value', '')}",
                    )
                )
        for item in snapshot.get("stale_archive", []):
            candidates.append(
                self._build_state_candidate_from_item(
                    item,
                    source="stale",
                    source_detail="stale",
                    retrieval_text=f"value: {item.get('value', '')}",
                )
            )
        for item in snapshot.get("unknown_current", []):
            candidates.append(
                self._build_state_candidate_from_item(
                    item,
                    source="unknown",
                    source_detail="unknown_current",
                    retrieval_text=f"reason: {item.get('reason', '')}",
                )
            )
        return candidates

    def _rank_global_item_value_hits(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        top_k: int,
        premise_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        candidates = self._global_item_value_candidates()
        if not candidates:
            return [], {}
        basis_query = self._build_basis_retrieval_query(parsed=parsed, query_text=query_text)
        hits = self._rank_state_candidates(
            query_text=basis_query,
            candidates=candidates,
            top_k=max(1, top_k),
        )
        enriched_hits: List[Dict[str, Any]] = []
        for hit in hits:
            enriched = dict(hit)
            enriched["basis_recovery_source"] = "global_item_value"
            enriched["basis_recovery_query_variant"] = "query_plus_queried_proposition"
            enriched_hits.append(enriched)
        filtered_hits, filter_log = self._filter_lane_hits_with_llm(
            lane_name="global_item_value",
            parsed=parsed,
            query_text=basis_query,
            hits=enriched_hits,
            max_hits=top_k,
            premise_context=premise_context,
        )
        for rank_index, hit in enumerate(filtered_hits, start=1):
            hit["rank_index"] = rank_index
        return filtered_hits, {
            "global_item_value::query_plus_queried_proposition": enriched_hits,
            "global_item_value::llm_filter": filtered_hits,
            "global_item_value::llm_filter_log": [filter_log],
        }

    def _rank_stale_unknown_basis_hits(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        top_k: int,
        premise_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        candidates = self._stale_unknown_candidates()
        if not candidates:
            return [], {}
        query_variants = self._build_basis_query_variants(parsed=parsed, query_text=query_text)
        variant_order = [
            "query_plus_queried_proposition",
        ]
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        branch_hits: Dict[str, List[Dict[str, Any]]] = {}
        for variant_name in variant_order:
            ranked_hits = self._rank_state_candidates(
                query_text=query_variants.get(variant_name, ""),
                candidates=candidates,
                top_k=max(1, top_k),
            )
            branch_hits[f"stale_unknown::{variant_name}"] = []
            for hit in ranked_hits:
                enriched = dict(hit)
                enriched["basis_recovery_source"] = "stale_unknown_conflict"
                enriched["basis_recovery_query_variant"] = variant_name
                branch_hits[f"stale_unknown::{variant_name}"].append(enriched)
                sig = (str(hit.get("source", "")).strip(), str(hit.get("id", "")).strip())
                existing = merged.get(sig)
                if existing is None or float(enriched.get("score", 0.0) or 0.0) > float(existing.get("score", 0.0) or 0.0):
                    merged[sig] = enriched
        hits = sorted(
            merged.values(),
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                1 if str(item.get("source", "")).strip() == "stale" else 0,
                -1 if str(item.get("source", "")).strip() == "unknown" else 0,
                -int(item.get("rank_index", 10_000) or 10_000),
            ),
            reverse=True,
        )[: max(1, top_k)]
        filtered_hits, filter_log = self._filter_lane_hits_with_llm(
            lane_name="stale_unknown",
            parsed=parsed,
            query_text=self._build_basis_retrieval_query(parsed=parsed, query_text=query_text),
            hits=hits,
            max_hits=top_k,
            premise_context=premise_context,
        )
        branch_hits["stale_unknown::llm_filter"] = filtered_hits
        branch_hits["stale_unknown::llm_filter_log"] = [filter_log]
        hits = filtered_hits
        for rank_index, hit in enumerate(hits, start=1):
            hit["rank_index"] = rank_index
        return hits, branch_hits

    def _build_bundles_from_hits(self, hits: List[Dict[str, Any]], *, basis_recovery_source: str) -> List[Dict[str, Any]]:
        if not hits:
            return []
        status_context = self._build_topk_status_context(hits)
        chain_context = self._build_topk_chain_context(hits)
        bundles, _ = self._build_topk_hit_bundles(
            topk_primary_hits=hits,
            topk_status_context=status_context,
            topk_chain_context=chain_context,
        )
        for bundle in bundles:
            bundle["basis_recovery_source"] = basis_recovery_source
        return bundles


    def _select_basis_recovery_items(
        self,
        *,
        parsed: QueryParse,
        bundles: List[Dict[str, Any]],
        blocked_item_ids: set[str],
        max_items: int = 5,
        allow_chain_facts: bool = False,
    ) -> Dict[str, Any]:
        recovered_ids: List[str] = []
        recovered_facts: List[str] = []
        chain_facts: List[str] = []
        conflict_bundle_facts: List[str] = []
        candidate_bundles: List[Dict[str, Any]] = []
        unknown_bundles: List[Dict[str, Any]] = []
        chain_bundles: List[Dict[str, Any]] = []
        conflict_bundle_bundles: List[Dict[str, Any]] = []
        structured_support_bundles: List[Dict[str, Any]] = []
        seen_support_sigs: set[Tuple[str, str]] = set()
        seen_candidate_sigs: set[Tuple[str, str]] = set()
        seen_candidate_pairs: set[Tuple[str, str]] = set()

        def _bundle_current_fact(bundle: Dict[str, Any]) -> str:
            primary_hit = bundle.get("primary_hit", {})
            payload = primary_hit.get("payload", {}) if isinstance(primary_hit, dict) else {}
            source = str(primary_hit.get("source", "")).strip()
            if source == "chain" and isinstance(payload, dict):
                linking_delta = payload.get("linking_delta") if isinstance(payload.get("linking_delta"), dict) else {}
                return str(linking_delta.get("proposed_value") or payload.get("value") or "").strip()
            if source in {"delta", "recoverable_delta"} and isinstance(payload, dict):
                return str(payload.get("proposed_value") or payload.get("value") or "").strip()
            if source == "chunk" and isinstance(payload, dict):
                return str(payload.get("text") or payload.get("value") or "").strip()
            if source in {"active", "stale", "unknown"} and isinstance(payload, dict):
                return str(payload.get("value") or payload.get("reason") or "").strip()
            if source == "conflict_bundle" and isinstance(payload, dict):
                return self._conflict_bundle_fact_from_payload(payload)
            return str(primary_hit.get("text_preview", "")).strip()

        def _recovery_rank(bundle: Dict[str, Any]) -> Tuple[Any, ...]:
            primary_hit = bundle.get("primary_hit", {})
            source_detail = str(primary_hit.get("source_detail", "")).strip()
            source = str(primary_hit.get("source", "")).strip()
            recovery_source = str(bundle.get("basis_recovery_source", "")).strip()
            recovery_priority = {
                "conflict_bundle": 5,
                "stale_unknown_conflict": 4,
                "global_item_value": 3,
                "topk": 1,
                "same_bucket": 0,
            }.get(recovery_source, 2)
            status_priority = {
                "strong_active": 4,
                "weak_active": 3,
                "unknown_current": 1,
                "track_conflict_bundle": 1,
                "stale_successor_link": 2,
                "stale_support_link": 1,
                "stale": 2,
            }.get(source_detail or source, 0)
            return (
                recovery_priority,
                status_priority,
                float(primary_hit.get("score", 0.0) or 0.0),
                -int(primary_hit.get("rank_index", 10_000) or 10_000),
            )

        def _maybe_add_structured_support(bundle: Dict[str, Any]) -> None:
            if len(structured_support_bundles) >= max(5, max_items + 2):
                return
            primary_hit = bundle.get("primary_hit", {})
            sig = (
                str(primary_hit.get("source", "")).strip(),
                str(primary_hit.get("id", "")).strip(),
            )
            if not sig[0] or not sig[1] or sig in seen_support_sigs:
                return
            seen_support_sigs.add(sig)
            structured_support_bundles.append(bundle)

        bundles = sorted(bundles, key=_recovery_rank, reverse=True)
        for bundle in bundles:
            primary_hit = bundle.get("primary_hit", {})
            source = str(primary_hit.get("source", "")).strip()
            source_detail = str(primary_hit.get("source_detail", "")).strip()
            item_id = self._hit_item_id(primary_hit)
            pair = self._hit_track_pair(primary_hit)
            sig = (
                str(primary_hit.get("source", "")).strip(),
                str(primary_hit.get("id", "")).strip(),
            )
            if source == "unknown":
                unknown_bundles.append(bundle)
                _maybe_add_structured_support(bundle)
            elif source == "chain":
                chain_bundles.append(bundle)
                if allow_chain_facts and source_detail == "stale_successor_link":
                    fact = _bundle_current_fact(bundle)
                    if fact:
                        chain_facts.append(fact)
                _maybe_add_structured_support(bundle)
            elif source == "conflict_bundle" or str(bundle.get("basis_recovery_source", "")).strip() == "conflict_bundle":
                conflict_bundle_bundles.append(bundle)
                fact = _bundle_current_fact(bundle)
                if fact:
                    conflict_bundle_facts.append(fact)
                _maybe_add_structured_support(bundle)
            elif source == "stale":
                _maybe_add_structured_support(bundle)
            elif source == "active":
                _maybe_add_structured_support(bundle)
                if item_id and item_id not in blocked_item_ids and item_id not in recovered_ids:
                    recovered_ids.append(item_id)
                    value = str(primary_hit.get("payload", {}).get("value", "")).strip()
                    if value:
                        recovered_facts.append(value)

            if not sig[0] or not sig[1] or sig in seen_candidate_sigs:
                continue
            if source == "active" and (not item_id or item_id in blocked_item_ids):
                continue
            if is_valid_bucket_track(*pair) and pair in seen_candidate_pairs:
                continue
            seen_candidate_sigs.add(sig)
            if is_valid_bucket_track(*pair):
                seen_candidate_pairs.add(pair)
            candidate_bundles.append(bundle)
            if len(candidate_bundles) >= max_items:
                break
        confidence = 0.0
        if candidate_bundles:
            confidence = max(
                float(bundle.get("primary_hit", {}).get("payload", {}).get("confidence", 0.0) or 0.0)
                for bundle in candidate_bundles
            )
        elif unknown_bundles:
            confidence = 0.25
        elif chain_bundles:
            confidence = 0.35
        elif conflict_bundle_bundles:
            confidence = 0.35
        return {
            "recovered_basis_item_ids": recovered_ids,
            "recovered_basis_facts": self._unique_preserve_order(recovered_facts),
            "chain_basis_facts": self._unique_preserve_order(chain_facts),
            "conflict_bundle_facts": self._unique_preserve_order(conflict_bundle_facts),
            "basis_confidence": max(0.0, min(1.0, confidence)),
            "selected_bundles": candidate_bundles,
            "unknown_bundles": unknown_bundles,
            "chain_bundles": chain_bundles,
            "structured_support_bundles": structured_support_bundles,
        }

    def _run_basis_recovery(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        verdict: PremiseVerdict,
        basis_recovery_hint: Dict[str, Any],
        relevant_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not basis_recovery_hint.get("needs_basis_recovery"):
            return {
                "needs_basis_recovery": False,
                "basis_query": "",
                "recovered_basis_item_ids": [],
                "recovered_basis_facts": [],
                "basis_confidence": 0.0,
                "recovered_current_basis_bundles": [],
                "unknown_track_bundles": [],
                "structured_support_bundles": [],
                "chain_basis_facts": [],
                "conflict_bundle_facts": [],
                "retrieval_branch_hits": {},
            }

        basis_query = self._build_basis_retrieval_query(parsed=parsed, query_text=query_text)
        blocked_item_ids = set(verdict.blocked_old_basis_item_ids)

        stale_unknown_top_k = max(1, min(5, self.thresholds.query_top_k))
        global_item_top_k = max(1, min(10, max(self.thresholds.query_top_k, self.thresholds.fallback_top_k)))

        lane_filter_premise_context = self._build_lane_filter_premise_context(verdict)

        stale_unknown_hits, retrieval_branch_hits = self._rank_stale_unknown_basis_hits(
            parsed=parsed,
            query_text=query_text,
            top_k=stale_unknown_top_k,
            premise_context=lane_filter_premise_context,
        )
        stale_unknown_hit_bundles = self._build_bundles_from_hits(
            stale_unknown_hits,
            basis_recovery_source="stale_unknown_conflict",
        )

        global_item_hits, global_item_branch_hits = self._rank_global_item_value_hits(
            parsed=parsed,
            query_text=query_text,
            top_k=global_item_top_k,
            premise_context=lane_filter_premise_context,
        )
        retrieval_branch_hits.update(global_item_branch_hits)
        global_item_hit_bundles = self._build_bundles_from_hits(
            global_item_hits,
            basis_recovery_source="global_item_value",
        )

        base_candidate_bundles = (
            stale_unknown_hit_bundles
            + global_item_hit_bundles
        )
        selection = self._select_basis_recovery_items(
            parsed=parsed,
            bundles=base_candidate_bundles,
            blocked_item_ids=blocked_item_ids,
            allow_chain_facts=verdict.status in {"OUTDATED", "UNRESOLVED"} or not verdict.old_premise_safe,
        )
        return {
            "needs_basis_recovery": True,
            "basis_query": basis_query,
            "recovered_basis_item_ids": selection["recovered_basis_item_ids"],
            "recovered_basis_facts": selection["recovered_basis_facts"],
            "chain_basis_facts": selection.get("chain_basis_facts", []),
            "conflict_bundle_facts": selection.get("conflict_bundle_facts", []),
            "basis_confidence": selection["basis_confidence"],
            "recovered_current_basis_bundles": selection["selected_bundles"],
            "structured_support_bundles": selection.get("structured_support_bundles", []),
            "unknown_track_bundles": selection["unknown_bundles"],
            "chain_basis_bundles": selection["chain_bundles"],
            "stale_unknown_hit_bundles": stale_unknown_hit_bundles,
            "global_item_hit_bundles": global_item_hit_bundles,
            "retrieval_branch_hits": retrieval_branch_hits,
        }

    def _merge_basis_recovery_into_verdict(
        self,
        *,
        verdict: PremiseVerdict,
        basis_recovery_result: Dict[str, Any],
    ) -> PremiseVerdict:
        blocked_set = set(verdict.blocked_old_basis_item_ids)
        recovered_ids = [
            item_id
            for item_id in self._unique_preserve_order(basis_recovery_result.get("recovered_basis_item_ids", []))
            if item_id not in blocked_set
        ]
        verdict.usable_current_basis_item_ids = self._unique_preserve_order(
            verdict.usable_current_basis_item_ids + recovered_ids
        )
        return verdict


    def _augment_verdict_conditioned_hit_bundles(
        self,
        *,
        verdict_conditioned_hit_bundles: Dict[str, List[Dict[str, Any]]],
        basis_recovery_result: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        augmented = {
            key: list(value) if isinstance(value, list) else value
            for key, value in verdict_conditioned_hit_bundles.items()
        }

        def _bundle_sig(bundle: Dict[str, Any]) -> Tuple[str, str]:
            primary_hit = bundle.get("primary_hit", {})
            return (str(primary_hit.get("source", "")).strip(), str(primary_hit.get("id", "")).strip())

        recovered_bundles = [
            bundle
            for bundle in basis_recovery_result.get("recovered_current_basis_bundles", [])
            if isinstance(bundle, dict)
        ]
        unknown_bundles = [
            bundle
            for bundle in basis_recovery_result.get("unknown_track_bundles", [])
            if isinstance(bundle, dict)
        ]
        structured_support_bundles = [
            bundle
            for bundle in basis_recovery_result.get("structured_support_bundles", [])
            if isinstance(bundle, dict)
        ]
        existing_usable = augmented.setdefault("usable_current_basis_bundles", [])
        seen_usable = {_bundle_sig(bundle) for bundle in existing_usable}
        for bundle in recovered_bundles:
            sig = _bundle_sig(bundle)
            if sig not in seen_usable:
                existing_usable.append(bundle)
                seen_usable.add(sig)

        existing_unknown = augmented.setdefault("unknown_track_bundles", [])
        seen_unknown = {_bundle_sig(bundle) for bundle in existing_unknown}
        for bundle in unknown_bundles:
            sig = _bundle_sig(bundle)
            if sig not in seen_unknown:
                existing_unknown.append(bundle)
                seen_unknown.add(sig)

        augmented["recovered_current_basis_bundles"] = recovered_bundles
        augmented["structured_support_bundles"] = structured_support_bundles
        return augmented

    def _refresh_verdict_basis_views(
        self,
        *,
        verdict: PremiseVerdict,
        parsed: QueryParse,
        relevant_context: Dict[str, Any],
        basis_recovery_result: Optional[Dict[str, Any]] = None,
    ) -> PremiseVerdict:
        relevant_context["verdict_conditioned_hit_bundles"] = self._build_verdict_conditioned_hit_bundles(
            verdict=verdict,
            relevant_context=relevant_context,
        )
        relevant_context["verdict_conditioned_hit_bundles"] = self._augment_verdict_conditioned_hit_bundles(
            verdict_conditioned_hit_bundles=relevant_context["verdict_conditioned_hit_bundles"],
            basis_recovery_result=basis_recovery_result or {},
        )
        decisive_basis = self._resolve_decisive_basis(
            parsed=parsed,
            verdict=verdict,
            relevant_context=relevant_context,
            basis_recovery_result=basis_recovery_result,
        )
        relevant_context["decisive_basis_item_ids"] = decisive_basis["decisive_basis_item_ids"]
        relevant_context["secondary_basis_item_ids"] = decisive_basis["secondary_basis_item_ids"]
        relevant_context["decisive_basis_bundles"] = decisive_basis["decisive_basis_bundles"]
        relevant_context["secondary_basis_bundles"] = decisive_basis["secondary_basis_bundles"]
        verdict.corrected_current_facts = self._derive_corrected_current_facts(
            verdict=verdict,
            relevant_context=relevant_context,
            basis_recovery_result=basis_recovery_result,
        )
        return verdict
