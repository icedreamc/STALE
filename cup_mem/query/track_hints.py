from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import PremiseVerdict, QueryParse
from ..memory.schema import is_valid_bucket_track, normalize_bucket_track_pairs


class QueryTrackHintMixin:
    def _find_track_pair_for_item_id(self, item_id: str) -> Optional[Tuple[str, str]]:
        normalized = str(item_id or "").strip()
        if not normalized:
            return None
        snapshot = self.store.to_snapshot()
        for items in snapshot.get("active_profile", {}).values():
            for item in items:
                if str(item.get("item_id", "")).strip() == normalized:
                    return (
                        str(item.get("bucket", "")).strip(),
                        str(item.get("local_track", "")).strip(),
                    )
        for item in snapshot.get("stale_archive", []):
            if str(item.get("item_id", "")).strip() == normalized:
                return (
                    str(item.get("bucket", "")).strip(),
                    str(item.get("local_track", "")).strip(),
                )
        return None

    def _build_basis_track_query(self, *, parsed: QueryParse, query_text: str) -> str:
        lines: List[str] = []
        for text in [
            query_text,
            parsed.queried_proposition,
            parsed.basis_recovery_focus,
            parsed.old_premise_focus,
            parsed.requested_action,
            parsed.action_goal,
            parsed.action_object,
            parsed.decision_basis_focus,
        ]:
            normalized = " ".join(str(text or "").split())
            if normalized and normalized not in lines:
                lines.append(normalized)
        track_refs = [
            f"{ref.bucket}/{ref.local_track}"
            for ref in parsed.target_bucket_tracks
            if is_valid_bucket_track(ref.bucket, ref.local_track)
        ]
        if track_refs:
            lines.append("candidate_tracks: " + ", ".join(track_refs))
        return "\n".join(lines)

    def _build_premise_track_query(self, *, parsed: QueryParse, query_text: str) -> str:
        lines: List[str] = []
        for text in [parsed.old_premise_focus, parsed.queried_proposition, query_text]:
            normalized = " ".join(str(text or "").split())
            if normalized and normalized not in lines:
                lines.append(normalized)
        track_refs = [
            f"{ref.bucket}/{ref.local_track}"
            for ref in parsed.target_bucket_tracks
            if is_valid_bucket_track(ref.bucket, ref.local_track)
        ]
        for presupposition in parsed.presuppositions:
            bucket = str(presupposition.get("bucket", "")).strip()
            local_track = str(presupposition.get("local_track", "")).strip()
            if is_valid_bucket_track(bucket, local_track):
                ref_text = f"{bucket}/{local_track}"
                if ref_text not in track_refs:
                    track_refs.append(ref_text)
        if track_refs:
            lines.append("candidate_tracks: " + ", ".join(track_refs))
        return "\n".join(lines)

    def _rank_tracks_for_premise_focus(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        exclude_pairs: Optional[set[Tuple[str, str]]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        exclude_pairs = exclude_pairs or set()
        premise_query = self._build_premise_track_query(parsed=parsed, query_text=query_text)
        if not premise_query.strip():
            return []

        snapshot = self.store.to_snapshot()
        track_texts: Dict[Tuple[str, str], Dict[str, List[str]]] = {}

        def _ensure_track(pair: Tuple[str, str]) -> Dict[str, List[str]]:
            return track_texts.setdefault(pair, {"active": [], "stale": [], "unknown": []})

        for items in snapshot.get("active_profile", {}).values():
            for item in items:
                pair = (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
                if not is_valid_bucket_track(*pair) or pair in exclude_pairs:
                    continue
                _ensure_track(pair)["active"].append(str(item.get("value", "")).strip())

        for item in snapshot.get("stale_archive", []):
            pair = (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
            if not is_valid_bucket_track(*pair) or pair in exclude_pairs:
                continue
            _ensure_track(pair)["stale"].append(str(item.get("value", "")).strip())

        for item in snapshot.get("unknown_current", []):
            pair = (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
            if not is_valid_bucket_track(*pair) or pair in exclude_pairs:
                continue
            _ensure_track(pair)["unknown"].append(str(item.get("reason", "")).strip())

        candidates: List[Dict[str, Any]] = []
        for (bucket, local_track), grouped in track_texts.items():
            lines = [f"track: {bucket}/{local_track}"]
            if grouped["active"]:
                lines.append("active: " + " | ".join(text for text in grouped["active"][:2] if text))
            if grouped["stale"]:
                lines.append("stale: " + " | ".join(text for text in grouped["stale"][:2] if text))
            if grouped["unknown"]:
                lines.append("unknown: " + " | ".join(text for text in grouped["unknown"][:1] if text))
            candidates.append(
                {
                    "bucket": bucket,
                    "local_track": local_track,
                    "retrieval_text": "\n".join(lines),
                }
            )

        if not candidates:
            return []

        ranked = self.embedding.rank(
            query_text=premise_query,
            candidates=candidates,
            text_getter=lambda item: item["retrieval_text"],
            top_k=min(max(1, top_k), len(candidates)),
        )
        results: List[Dict[str, Any]] = []
        for entry in ranked:
            item = entry["item"]
            results.append(
                {
                    "bucket": item["bucket"],
                    "local_track": item["local_track"],
                    "source": "premise_focus_rank",
                    "score": entry["score"],
                }
            )
        return results

    def _rank_tracks_for_decision_basis_focus(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        exclude_pairs: Optional[set[Tuple[str, str]]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        exclude_pairs = exclude_pairs or set()
        basis_query = self._build_basis_track_query(parsed=parsed, query_text=query_text)
        if not basis_query.strip():
            return []

        snapshot = self.store.to_snapshot()
        track_texts: Dict[Tuple[str, str], Dict[str, List[str]]] = {}

        def _ensure_track(pair: Tuple[str, str]) -> Dict[str, List[str]]:
            return track_texts.setdefault(pair, {"active": [], "stale": [], "unknown": []})

        for items in snapshot.get("active_profile", {}).values():
            for item in items:
                pair = (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
                if not is_valid_bucket_track(*pair) or pair in exclude_pairs:
                    continue
                _ensure_track(pair)["active"].append(str(item.get("value", "")).strip())

        for item in snapshot.get("stale_archive", []):
            pair = (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
            if not is_valid_bucket_track(*pair) or pair in exclude_pairs:
                continue
            _ensure_track(pair)["stale"].append(str(item.get("value", "")).strip())

        for item in snapshot.get("unknown_current", []):
            pair = (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
            if not is_valid_bucket_track(*pair) or pair in exclude_pairs:
                continue
            _ensure_track(pair)["unknown"].append(str(item.get("reason", "")).strip())

        candidates: List[Dict[str, Any]] = []
        for (bucket, local_track), grouped in track_texts.items():
            lines = [f"track: {bucket}/{local_track}"]
            if grouped["active"]:
                lines.append("active: " + " | ".join(text for text in grouped["active"][:2] if text))
            if grouped["stale"]:
                lines.append("stale: " + " | ".join(text for text in grouped["stale"][:2] if text))
            if grouped["unknown"]:
                lines.append("unknown: " + " | ".join(text for text in grouped["unknown"][:1] if text))
            candidates.append(
                {
                    "bucket": bucket,
                    "local_track": local_track,
                    "retrieval_text": "\n".join(lines),
                }
            )

        if not candidates:
            return []

        ranked = self.embedding.rank(
            query_text=basis_query,
            candidates=candidates,
            text_getter=lambda item: item["retrieval_text"],
            top_k=min(max(1, top_k), len(candidates)),
        )
        results: List[Dict[str, Any]] = []
        for entry in ranked:
            item = entry["item"]
            results.append(
                {
                    "bucket": item["bucket"],
                    "local_track": item["local_track"],
                    "source": "decision_basis_focus_rank",
                    "score": entry["score"],
                }
            )
        return results

    def _build_basis_recovery_hint(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        usable_item_ids = set(self._unique_preserve_order(verdict.usable_current_basis_item_ids))
        blocked_item_ids = set(self._unique_preserve_order(verdict.blocked_old_basis_item_ids))
        visible_usable_item_ids: set[str] = set()
        usable_has_strong = False

        for bundle in relevant_context.get("topk_hit_bundles", []):
            primary_hit = bundle.get("primary_hit", {})
            item_id = self._hit_item_id(primary_hit)
            if not item_id or item_id not in usable_item_ids or item_id in blocked_item_ids:
                continue
            visible_usable_item_ids.add(item_id)
            if str(primary_hit.get("source_detail", "")).strip() == "strong_active":
                usable_has_strong = True

        conflict_subset = relevant_context.get("premise_conflict_subset", {}) or {}
        for item in conflict_subset.get("active_items", []) if isinstance(conflict_subset.get("active_items", []), list) else []:
            item_id = str(item.get("item_id", "")).strip()
            if not item_id or item_id not in usable_item_ids or item_id in blocked_item_ids:
                continue
            visible_usable_item_ids.add(item_id)
            if str(item.get("active_strength", "STRONG")).upper() == "STRONG":
                usable_has_strong = True

        needs_basis_recovery = False
        basis_gap_reason = ""
        if verdict.status == "OUTDATED" or not verdict.old_premise_safe:
            needs_basis_recovery = True
            basis_gap_reason = "no_usable_basis" if not usable_item_ids else "current_basis_incomplete"
        if not usable_item_ids:
            needs_basis_recovery = True
            basis_gap_reason = basis_gap_reason or "no_usable_basis"
        elif parsed.intent_type == "TASK_PLANNING":
            if not visible_usable_item_ids:
                needs_basis_recovery = True
                basis_gap_reason = basis_gap_reason or "no_usable_basis"
            elif not usable_has_strong:
                needs_basis_recovery = True
                basis_gap_reason = basis_gap_reason or "usable_too_generic"

        conflict_tracks: List[Dict[str, Any]] = []
        seen_conflict_pairs: set[Tuple[str, str]] = set()

        def _add_conflict_pair(pair: Tuple[str, str], *, source: str) -> None:
            if not is_valid_bucket_track(*pair) or pair in seen_conflict_pairs:
                return
            seen_conflict_pairs.add(pair)
            conflict_tracks.append(
                {
                    "bucket": pair[0],
                    "local_track": pair[1],
                    "source": source,
                }
            )

        for item_id in verdict.outdated_item_ids:
            pair = self._find_track_pair_for_item_id(item_id)
            if pair is not None:
                _add_conflict_pair(pair, source="outdated_item_same_track")

        for item in verdict.unknown_tracks:
            pair = (
                str(item.get("bucket", "")).strip(),
                str(item.get("local_track", "")).strip(),
            )
            _add_conflict_pair(pair, source="unknown_track")

        for hit in relevant_context.get("topk_primary_hits", []):
            source = str(hit.get("source", "")).strip()
            if source not in {"stale", "unknown"}:
                continue
            _add_conflict_pair(self._hit_track_pair(hit), source=f"primary_{source}_track")

        candidate_basis_tracks: List[Dict[str, Any]] = []
        seen_candidate_pairs: set[Tuple[str, str]] = set()

        def _add_candidate_pair(pair: Tuple[str, str], *, source: str, score: Optional[float] = None) -> None:
            if not is_valid_bucket_track(*pair) or pair in seen_candidate_pairs:
                return
            seen_candidate_pairs.add(pair)
            entry: Dict[str, Any] = {
                "bucket": pair[0],
                "local_track": pair[1],
                "source": source,
            }
            if score is not None:
                entry["score"] = score
            candidate_basis_tracks.append(entry)

        for item in conflict_tracks:
            _add_candidate_pair(
                (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip()),
                source=str(item.get("source", "conflict_track")).strip(),
            )

        for ref in parsed.target_bucket_tracks:
            _add_candidate_pair((ref.bucket, ref.local_track), source="target_bucket_track")

        for item in parsed.presuppositions:
            _add_candidate_pair(
                (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip()),
                source="presupposition_track",
            )

        focus_ranked_tracks = self._rank_tracks_for_decision_basis_focus(
            parsed=parsed,
            query_text=query_text,
            exclude_pairs=seen_candidate_pairs,
            top_k=2,
        )
        for item in focus_ranked_tracks:
            _add_candidate_pair(
                (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip()),
                source=str(item.get("source", "decision_basis_focus_rank")).strip(),
                score=float(item.get("score", 0.0) or 0.0),
            )

        return {
            "needs_basis_recovery": needs_basis_recovery,
            "candidate_basis_tracks": candidate_basis_tracks,
            "conflict_tracks": conflict_tracks,
            "basis_gap_reason": basis_gap_reason or ("no_usable_basis" if needs_basis_recovery else ""),
        }

    def _collect_premise_conflict_subset(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        relevant_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        seed_tracks: List[Tuple[str, str]] = []
        seed_track_logs: List[Dict[str, Any]] = []
        seen_pairs: set[Tuple[str, str]] = set()

        def _add_pair(pair: Tuple[str, str], *, source: str) -> None:
            if not is_valid_bucket_track(*pair) or pair in seen_pairs:
                return
            seen_pairs.add(pair)
            seed_tracks.append(pair)
            seed_track_logs.append(
                {
                    "bucket": pair[0],
                    "local_track": pair[1],
                    "source": source,
                }
            )

        for ref in parsed.target_bucket_tracks:
            _add_pair((ref.bucket, ref.local_track), source="target_bucket_track")

        for item in parsed.presuppositions:
            _add_pair(
                (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip()),
                source="presupposition_track",
            )

        premise_ranked_tracks = self._rank_tracks_for_premise_focus(
            parsed=parsed,
            query_text=query_text,
            exclude_pairs=seen_pairs,
            top_k=3,
        )
        for item in premise_ranked_tracks:
            _add_pair(
                (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip()),
                source=str(item.get("source", "premise_focus_rank")).strip(),
            )

        conflict_seed_count = 0
        for hit in relevant_context.get("topk_primary_hits", []):
            source = str(hit.get("source", "")).strip()
            source_detail = str(hit.get("source_detail", "")).strip()
            if source not in {"stale", "unknown"} and source_detail != "weak_active":
                continue
            _add_pair(self._hit_track_pair(hit), source=f"primary_conflict_hit:{source_detail or source}")
            conflict_seed_count += 1
            if conflict_seed_count >= 2:
                break

        seed_tracks = normalize_bucket_track_pairs(seed_tracks[:6])
        subset = self.store.get_exact_profile_subset(seed_tracks)
        subset_summary = self._summarize_profile_subset(subset, requested_pairs=seed_tracks)
        return {
            "seed_tracks": [
                {"bucket": bucket, "local_track": local_track}
                for bucket, local_track in seed_tracks
            ],
            "seed_track_logs": seed_track_logs,
            "subset": subset,
            "subset_summary": subset_summary,
        }

