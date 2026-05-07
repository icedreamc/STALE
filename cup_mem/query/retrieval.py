from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List

from ..memory.models import (
    PremiseVerdict,
)
from .readout import QueryReadoutMixin
from .basis_recovery_impl import BasisRecoveryMixin
from .track_hints import QueryTrackHintMixin


class QueryRetrievalMixin(QueryReadoutMixin, QueryTrackHintMixin, BasisRecoveryMixin):
    @staticmethod
    def _dedupe_context_items(items: List[Dict[str, Any]], *, key_fn: Callable[[Dict[str, Any]], Any]) -> List[Dict[str, Any]]:
        deduped: Dict[Any, Dict[str, Any]] = {}
        for item in items:
            deduped[key_fn(item)] = item
        return list(deduped.values())

    def _summarize_query_readout_context(self, relevant_context: Dict[str, Any]) -> Dict[str, Any]:
        peripheral_context = relevant_context.get("peripheral_context", {})
        return {
            "query_readout_mode": relevant_context.get("query_readout_mode", ""),
            "primary_hit_count": len(relevant_context.get("topk_primary_hits", [])),
            "primary_hits_by_source": dict(Counter(hit.get("source", "") for hit in relevant_context.get("topk_primary_hits", []))),
            "hit_bundle_count": len(relevant_context.get("topk_hit_bundles", [])),
            "collapsed_related_hit_count": sum(
                len(bundle.get("collapsed_related_hits", []))
                for bundle in relevant_context.get("topk_hit_bundles", [])
            ),
            "peripheral_context_counts": {
                "mixed_query_topk_results": len(peripheral_context.get("mixed_query_topk_results", [])),
            },
            "premise_conflict_subset_requested_track_count": len(relevant_context.get("premise_conflict_subset_seed_tracks", [])),
            "premise_conflict_subset_summary": relevant_context.get("premise_conflict_subset_summary", {}),
        }

    def _should_run_second_pass(
        self,
        *,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
    ) -> List[str]:
        reasons: List[str] = []
        primary_hits = relevant_context.get("topk_primary_hits", [])
        if not primary_hits:
            reasons.append("primary_topk_empty")
        if float(verdict.confidence) < self.thresholds.verifier_fallback_threshold:
            reasons.append(
                f"low_initial_confidence:{float(verdict.confidence):.2f}<"
                f"{self.thresholds.verifier_fallback_threshold:.2f}"
            )
        if verdict.status == "UNRESOLVED":
            reasons.append("initial_unresolved")
        has_strong = any(str(hit.get("source_detail", "")).strip() == "strong_active" for hit in primary_hits)
        has_stale_or_unknown = any(str(hit.get("source", "")).strip() in {"stale", "unknown"} for hit in primary_hits)
        if has_strong and has_stale_or_unknown:
            reasons.append("core_conflict_strong_vs_stale_or_unknown")
        if has_stale_or_unknown and verdict.status in {"OUTDATED", "UNRESOLVED"}:
            reasons.append("premise_conflict_line_present")
        if has_stale_or_unknown and not verdict.old_premise_safe:
            reasons.append("unsafe_premise_with_conflict_line")
        deduped_reasons: List[str] = []
        for reason in reasons:
            if reason not in deduped_reasons:
                deduped_reasons.append(reason)
        return deduped_reasons
