from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import PremiseVerdict, QueryParse, SessionDelta, maybe_dict_list
from ..memory.schema import is_valid_bucket_track


class QueryReadoutMixin:
    @staticmethod
    def _build_lane_filter_parsed_query_payload(parsed: QueryParse) -> Dict[str, Any]:
        payload = {
            "intent_type": parsed.intent_type,
            "old_premise_focus": parsed.old_premise_focus,
            "basis_recovery_focus": parsed.basis_recovery_focus,
        }
        if parsed.intent_type == "TASK_PLANNING" and parsed.requested_action:
            payload["requested_action"] = parsed.requested_action
        return payload

    @staticmethod
    def delta_text(delta: SessionDelta) -> str:
        return (
            f"delta_kind: {delta.delta_kind}\n"
            f"bucket: {delta.bucket}\n"
            f"local_track: {delta.local_track}\n"
            f"proposed_value: {delta.proposed_value}\n"
            f"source_type: {delta.source_type}"
        )

    @staticmethod
    def profile_item_text(item: Dict[str, Any]) -> str:
        return (
            f"bucket: {item.get('bucket', '')}\n"
            f"local_track: {item.get('local_track', '')}\n"
            f"value: {item.get('value', '')}\n"
            f"status: {item.get('status', '')}"
        )

    @staticmethod
    def unknown_item_text(item: Dict[str, Any]) -> str:
        return (
            f"bucket: {item.get('bucket', '')}\n"
            f"local_track: {item.get('local_track', '')}\n"
            f"status: {item.get('status', '')}\n"
            f"reason: {item.get('reason', '')}"
        )

    @staticmethod
    def _latest_archive_event(item: Dict[str, Any]) -> Dict[str, Any]:
        archived_events = [
            revision
            for revision in maybe_dict_list(item.get("revision_history", []))
            if str(revision.get("event", "")).strip() == "archived"
        ]
        if not archived_events:
            return {}
        return archived_events[-1]

    def _build_query_retrieval_query(self, *, parsed: QueryParse, query_text: str) -> str:
        lines: List[str] = []
        for text in [parsed.old_premise_focus, parsed.hidden_old_setup, parsed.queried_proposition, query_text]:
            normalized = " ".join(str(text or "").split())
            if normalized and normalized not in lines:
                lines.append(normalized)
        secondary_hints: List[str] = []
        for presupposition in parsed.presuppositions:
            text = " ".join(str(presupposition.get("text", "")).split())
            if text and text not in secondary_hints:
                secondary_hints.append(text)
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
            lines.append("target_tracks: " + ", ".join(track_refs))
        if secondary_hints:
            lines.append("secondary_presupposition_hints: " + " | ".join(secondary_hints))
        return "\n".join(lines)

    def _retrieve_query_topk(self, *, parsed: QueryParse, query_text: str) -> List[Dict[str, Any]]:
        snapshot = self.store.to_snapshot()
        candidates: List[Dict[str, Any]] = []
        for items in snapshot["active_profile"].values():
            for item in items:
                strength = str(item.get("active_strength", "STRONG")).upper()
                bucket = str(item.get("bucket", "")).strip()
                local_track = str(item.get("local_track", "")).strip()
                retrieval_text = self.profile_item_text(item) + f"\nactive_strength: {strength}"
                candidates.append(
                    {
                        "source": "active",
                        "source_detail": "weak_active" if strength == "WEAK" else "strong_active",
                        "id": item["item_id"],
                        "payload": item,
                        "bucket": bucket,
                        "local_track": local_track,
                        "retrieval_text": retrieval_text,
                    }
                )

        for item in snapshot["stale_archive"]:
            bucket = str(item.get("bucket", "")).strip()
            local_track = str(item.get("local_track", "")).strip()
            retrieval_text = self.profile_item_text(item)
            candidates.append(
                {
                    "source": "stale",
                    "source_detail": "stale",
                    "id": item["item_id"],
                    "payload": item,
                    "bucket": bucket,
                    "local_track": local_track,
                    "retrieval_text": retrieval_text,
                }
            )

        for item in snapshot["unknown_current"]:
            bucket = str(item.get("bucket", "")).strip()
            local_track = str(item.get("local_track", "")).strip()
            candidates.append(
                {
                    "source": "unknown",
                    "source_detail": "unknown_current",
                    "id": f"{bucket}/{local_track}",
                    "payload": item,
                    "bucket": bucket,
                    "local_track": local_track,
                    "retrieval_text": self.unknown_item_text(item),
                }
            )

        if not candidates:
            return []

        query_top_k = max(1, self.thresholds.query_top_k)
        embedding_top_k = min(len(candidates), max(query_top_k, query_top_k + 4))
        ranked = self.embedding.rank(
            query_text=self._build_query_retrieval_query(parsed=parsed, query_text=query_text),
            candidates=candidates,
            text_getter=lambda item: item["retrieval_text"],
            top_k=query_top_k,
        )
        final_hits: List[Dict[str, Any]] = []
        for rank_index, entry in enumerate(ranked, start=1):
            candidate = entry["item"]
            final_hits.append(
                {
                    "rank_index": rank_index,
                    "score": entry["score"],
                    "source": candidate["source"],
                    "source_detail": candidate["source_detail"],
                    "id": candidate["id"],
                    "payload": candidate["payload"],
                    "bucket": candidate["bucket"],
                    "local_track": candidate["local_track"],
                    "session_ids": self._infer_session_ids_from_payload(candidate["payload"]),
                    "text_preview": self._preview_text(candidate["retrieval_text"]),
                }
            )
        return final_hits

    def _retrieve_query_peripheral_topk(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        exclude_refs: Optional[set[Tuple[str, str]]] = None,
        source_allowlist: Optional[set[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        exclude_refs = exclude_refs or set()
        source_allowlist = {str(item).strip() for item in (source_allowlist or set()) if str(item).strip()}
        if not source_allowlist:
            source_allowlist = {"stale", "unknown"}
        snapshot = self.store.to_snapshot()
        candidates: List[Dict[str, Any]] = []

        for items in snapshot["active_profile"].values():
            for item in items:
                if source_allowlist and "active" not in source_allowlist:
                    continue
                ref = ("active", str(item.get("item_id", "")).strip())
                if ref in exclude_refs:
                    continue
                strength = str(item.get("active_strength", "STRONG")).upper()
                candidates.append(
                    {
                        "source": "active",
                        "source_detail": "weak_active" if strength == "WEAK" else "strong_active",
                        "id": item["item_id"],
                        "payload": item,
                        "bucket": item.get("bucket", ""),
                        "local_track": item.get("local_track", ""),
                        "retrieval_text": self.profile_item_text(item) + f"\nactive_strength: {strength}",
                    }
                )
        for item in snapshot["stale_archive"]:
            if source_allowlist and "stale" not in source_allowlist:
                continue
            ref = ("stale", str(item.get("item_id", "")).strip())
            if ref in exclude_refs:
                continue
            candidates.append(
                {
                    "source": "stale",
                    "source_detail": "stale",
                    "id": item["item_id"],
                    "payload": item,
                    "bucket": item.get("bucket", ""),
                    "local_track": item.get("local_track", ""),
                    "retrieval_text": self.profile_item_text(item),
                }
            )
        for item in snapshot["unknown_current"]:
            if source_allowlist and "unknown" not in source_allowlist:
                continue
            ref = ("unknown", f"{item.get('bucket', '')}/{item.get('local_track', '')}")
            if ref in exclude_refs:
                continue
            candidates.append(
                {
                    "source": "unknown",
                    "source_detail": "unknown_current",
                    "id": f"{item.get('bucket', '')}/{item.get('local_track', '')}",
                    "payload": item,
                    "bucket": item.get("bucket", ""),
                    "local_track": item.get("local_track", ""),
                    "retrieval_text": self.unknown_item_text(item),
                }
            )
        if not candidates:
            return []

        resolved_top_k = int(top_k if top_k is not None else self.thresholds.fallback_top_k)
        if resolved_top_k <= 0:
            return []

        ranked = self.embedding.rank(
            query_text=self._build_query_retrieval_query(parsed=parsed, query_text=query_text),
            candidates=candidates,
            text_getter=lambda item: item["retrieval_text"],
            top_k=resolved_top_k,
        )
        return [
            {
                "rank_index": rank_index,
                "score": entry["score"],
                "source": entry["item"]["source"],
                "source_detail": entry["item"]["source_detail"],
                "id": entry["item"]["id"],
                "payload": entry["item"]["payload"],
                "bucket": entry["item"]["bucket"],
                "local_track": entry["item"]["local_track"],
                "session_ids": self._infer_session_ids_from_payload(entry["item"]["payload"]),
                "text_preview": self._preview_text(entry["item"]["retrieval_text"]),
            }
            for rank_index, entry in enumerate(ranked, start=1)
        ]

    def _build_conflict_centered_query(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        relevant_context: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        seen: set[str] = set()

        for text in [query_text, parsed.queried_proposition, parsed.old_premise_focus, parsed.decision_basis_focus]:
            normalized = " ".join(str(text or "").split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                lines.append(normalized)

        for bundle in relevant_context.get("topk_hit_bundles", []):
            primary_hit = bundle.get("primary_hit", {})
            source = str(primary_hit.get("source", "")).strip()
            source_detail = str(primary_hit.get("source_detail", "")).strip()
            if source not in {"stale", "unknown"} and source_detail not in {"weak_active"}:
                continue

            payload = primary_hit.get("payload", {})
            if source == "unknown":
                summary = self.unknown_item_text(payload)
            else:
                summary = self.profile_item_text(payload)
            normalized = " ".join(str(summary or "").split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                lines.append(f"conflict_hit: {normalized}")

            chain_view = bundle.get("chain_view", {}).get("primary_chain_view", {})
            archive_event = chain_view.get("archive_event") or {}
            archive_reason = " ".join(
                str(archive_event.get(key, "") or "").strip()
                for key in ["reason", "invalidated_by_delta_id"]
                if str(archive_event.get(key, "") or "").strip()
            )
            if archive_reason:
                archive_line = f"archive_event: {archive_reason}"
                if archive_line not in seen:
                    seen.add(archive_line)
                    lines.append(archive_line)

        return "\n".join(lines)

    def _retrieve_query_conflict_evidence(
        self,
        *,
        parsed: QueryParse,
        query_text: str,
        relevant_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        exclude_refs = {
            (str(hit.get("source", "")).strip(), str(hit.get("id", "")).strip())
            for hit in relevant_context.get("topk_primary_hits", [])
            if str(hit.get("source", "")).strip() and str(hit.get("id", "")).strip()
        }
        conflict_query = self._build_conflict_centered_query(
            parsed=parsed,
            query_text=query_text,
            relevant_context=relevant_context,
        )
        evidence = self._retrieve_query_peripheral_topk(
            parsed=parsed,
            query_text=conflict_query,
            exclude_refs=exclude_refs,
            source_allowlist={"stale", "unknown"},
            top_k=self.thresholds.fallback_top_k,
        )
        filtered: List[Dict[str, Any]] = []
        for item in evidence:
            source = str(item.get("source", "")).strip()
            source_detail = str(item.get("source_detail", "")).strip()
            if source == "active" and source_detail == "strong_active":
                continue
            filtered.append(item)
        return filtered


    @staticmethod
    def _hit_ref(hit: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "source": hit.get("source", ""),
            "source_detail": hit.get("source_detail", ""),
            "id": hit.get("id", ""),
            "rank_index": hit.get("rank_index"),
            "score": hit.get("score"),
            "bucket": hit.get("bucket", ""),
            "local_track": hit.get("local_track", ""),
        }

    def _build_topk_status_context(self, topk_primary_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        status_context: List[Dict[str, Any]] = []
        for hit in topk_primary_hits:
            source = str(hit.get("source", "")).strip()
            source_detail = str(hit.get("source_detail", "")).strip()
            payload = hit.get("payload", {})
            status_label = ""
            if source == "active":
                status_label = "ACTIVE_WEAK" if source_detail == "weak_active" else "ACTIVE_STRONG"
            elif source == "stale":
                status_label = "STALE"
            elif source == "unknown":
                status_label = "UNKNOWN_CURRENT"
            status_context.append(
                {
                    "hit_ref": self._hit_ref(hit),
                    "bucket": hit.get("bucket", ""),
                    "local_track": hit.get("local_track", ""),
                    "status_label": status_label,
                    "view_type": "track_unknown_status" if source == "unknown" else "item_status",
                    "item": payload if source in {"active", "stale"} else None,
                    "unknown_current": payload if source == "unknown" else None,
                }
            )
        return status_context

    @staticmethod
    def _pick_single_track_replacement(
        items: List[Dict[str, Any]],
        *,
        exclude_item_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        filtered = [
            item
            for item in items
            if str(item.get("item_id", "")).strip() and str(item.get("item_id", "")).strip() != exclude_item_id
        ]
        if not filtered:
            return None
        ranked = sorted(
            filtered,
            key=lambda item: (
                1 if str(item.get("active_strength", "STRONG")).upper() != "WEAK" else 0,
                float(item.get("confidence", 0.0) or 0.0),
                str(item.get("last_updated", "")),
                str(item.get("item_id", "")),
            ),
            reverse=True,
        )
        return ranked[0]

    def _build_topk_chain_context(self, topk_primary_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        snapshot = self.store.to_snapshot()
        active_by_track: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        unknown_by_track: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for items in snapshot["active_profile"].values():
            for item in items:
                pair = (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
                active_by_track.setdefault(pair, []).append(item)
        for item in snapshot["unknown_current"]:
            pair = (str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
            unknown_by_track[pair] = item

        delta_by_id = {delta.delta_id: delta for delta in self.delta_store}
        links_by_stale_item: Dict[str, List[Dict[str, Any]]] = {}
        for link in self.store.stale_support_links:
            links_by_stale_item.setdefault(link.stale_item_id, []).append(link.to_dict())

        chain_context: List[Dict[str, Any]] = []
        for hit in topk_primary_hits:
            source = str(hit.get("source", "")).strip()
            payload = hit.get("payload", {})
            bucket = str(hit.get("bucket", "")).strip()
            local_track = str(hit.get("local_track", "")).strip()
            pair = (bucket, local_track)
            item_id = str(payload.get("item_id", "")).strip()

            archive_event = None
            direct_links: List[Dict[str, Any]] = []
            track_current_replacement = None
            track_unknown_current = None

            if source == "stale":
                archive_event = self._latest_archive_event(payload)
                candidate_links = sorted(
                    links_by_stale_item.get(item_id, []),
                    key=lambda item: (
                        -int(item.get("link_session_idx", -1)),
                        str(item.get("linking_delta_id", "")),
                    ),
                )[:2]
                direct_links = []
                for link in candidate_links:
                    delta_id = str(link.get("linking_delta_id", "")).strip()
                    delta = delta_by_id.get(delta_id)
                    direct_links.append(
                        {
                            **link,
                            "linking_delta": delta.to_dict() if delta is not None else None,
                        }
                    )
                replacement = self._pick_single_track_replacement(
                    active_by_track.get(pair, []),
                    exclude_item_id=item_id,
                )
                if replacement is not None:
                    track_current_replacement = replacement
                elif pair in unknown_by_track:
                    track_unknown_current = unknown_by_track[pair]
            elif source == "unknown":
                track_unknown_current = payload

            chain_context.append(
                {
                    "hit_ref": self._hit_ref(hit),
                    "bucket": bucket,
                    "local_track": local_track,
                    "archive_event": archive_event,
                    "direct_links": direct_links,
                    "track_current_replacement": track_current_replacement,
                    "track_unknown_current": track_unknown_current,
                }
            )
        return chain_context

    @staticmethod
    def _primary_hit_status_label(hit: Dict[str, Any]) -> str:
        source = str(hit.get("source", "")).strip()
        source_detail = str(hit.get("source_detail", "")).strip()
        if source == "active":
            return "ACTIVE_WEAK" if source_detail == "weak_active" else "ACTIVE_STRONG"
        if source == "stale":
            return "STALE"
        if source == "unknown":
            return "UNKNOWN_CURRENT"
        return source.upper()

    def _group_query_hit_family_keys(self, topk_primary_hits: List[Dict[str, Any]]) -> Dict[str, Tuple[Any, ...]]:
        pair_to_statuses: Dict[Tuple[str, str], set[str]] = {}
        for hit in topk_primary_hits:
            pair = (str(hit.get("bucket", "")).strip(), str(hit.get("local_track", "")).strip())
            if not is_valid_bucket_track(*pair):
                continue
            pair_to_statuses.setdefault(pair, set()).add(self._primary_hit_status_label(hit))

        family_keys: Dict[str, Tuple[Any, ...]] = {}
        for hit in topk_primary_hits:
            hit_id = f"{hit.get('source', '')}:{hit.get('id', '')}"
            pair = (str(hit.get("bucket", "")).strip(), str(hit.get("local_track", "")).strip())
            statuses = pair_to_statuses.get(pair, set())
            source = str(hit.get("source", "")).strip()
            if (
                is_valid_bucket_track(*pair)
                and (
                    "UNKNOWN_CURRENT" in statuses
                    or ("STALE" in statuses and ("ACTIVE_STRONG" in statuses or "ACTIVE_WEAK" in statuses))
                )
            ):
                family_keys[hit_id] = ("track_conflict_family", pair[0], pair[1])
            else:
                family_keys[hit_id] = ("singleton_hit", source, str(hit.get("id", "")).strip())
        return family_keys

    @staticmethod
    def _primary_hit_rank_key(hit: Dict[str, Any]) -> Tuple[Any, ...]:
        source_detail = str(hit.get("source_detail", "")).strip()
        source = str(hit.get("source", "")).strip()
        status_priority = {
            "strong_active": 4,
            "weak_active": 3,
            "stale": 2,
            "unknown_current": 1,
        }.get(source_detail or source, 0)
        return (
            float(hit.get("score", 0.0) or 0.0),
            status_priority,
            -int(hit.get("rank_index", 10_000) or 10_000),
            str(hit.get("id", "")),
        )

    def _build_topk_hit_bundles(
        self,
        *,
        topk_primary_hits: List[Dict[str, Any]],
        topk_status_context: List[Dict[str, Any]],
        topk_chain_context: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        status_by_hit: Dict[str, Dict[str, Any]] = {
            f"{item.get('hit_ref', {}).get('source', '')}:{item.get('hit_ref', {}).get('id', '')}": item
            for item in topk_status_context
        }
        chain_by_hit: Dict[str, Dict[str, Any]] = {
            f"{item.get('hit_ref', {}).get('source', '')}:{item.get('hit_ref', {}).get('id', '')}": item
            for item in topk_chain_context
        }
        family_keys = self._group_query_hit_family_keys(topk_primary_hits)
        grouped_hits: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for hit in topk_primary_hits:
            hit_key = f"{hit.get('source', '')}:{hit.get('id', '')}"
            grouped_hits.setdefault(family_keys[hit_key], []).append(hit)

        bundles: List[Dict[str, Any]] = []
        deduped_primary_hits: List[Dict[str, Any]] = []
        for family_key, hits in grouped_hits.items():
            ranked_hits = sorted(hits, key=self._primary_hit_rank_key, reverse=True)
            primary_hit = ranked_hits[0]
            related_hits = ranked_hits[1:]
            primary_key = f"{primary_hit.get('source', '')}:{primary_hit.get('id', '')}"
            primary_status_view = status_by_hit.get(primary_key, {})
            primary_chain_view = chain_by_hit.get(primary_key, {})

            status_view = {
                "primary_status_view": primary_status_view,
                "collapsed_related_status_views": [
                    status_by_hit.get(f"{hit.get('source', '')}:{hit.get('id', '')}", {})
                    for hit in related_hits
                    if status_by_hit.get(f"{hit.get('source', '')}:{hit.get('id', '')}", {})
                ],
            }

            if str(primary_hit.get("source_detail", "")).strip() == "strong_active":
                primary_chain_view = {
                    "hit_ref": primary_chain_view.get("hit_ref", self._hit_ref(primary_hit)),
                    "bucket": primary_chain_view.get("bucket", primary_hit.get("bucket", "")),
                    "local_track": primary_chain_view.get("local_track", primary_hit.get("local_track", "")),
                    "archive_event": None,
                    "direct_links": [],
                    "track_current_replacement": None,
                    "track_unknown_current": None,
                }

            chain_view = {
                "primary_chain_view": primary_chain_view,
                "collapsed_related_chain_views": [
                    chain_by_hit.get(f"{hit.get('source', '')}:{hit.get('id', '')}", {})
                    for hit in related_hits
                    if chain_by_hit.get(f"{hit.get('source', '')}:{hit.get('id', '')}", {})
                ],
            }

            bundles.append(
                {
                    "bundle_key": list(family_key),
                    "primary_hit": primary_hit,
                    "status_view": status_view,
                    "chain_view": chain_view,
                    "collapsed_related_hits": [self._hit_ref(hit) for hit in related_hits],
                }
            )
            deduped_primary_hits.append(primary_hit)

        bundles.sort(
            key=lambda item: (
                int(item.get("primary_hit", {}).get("rank_index", 10_000) or 10_000),
                -float(item.get("primary_hit", {}).get("score", 0.0) or 0.0),
            )
        )
        deduped_primary_hits.sort(
            key=lambda hit: (
                int(hit.get("rank_index", 10_000) or 10_000),
                -float(hit.get("score", 0.0) or 0.0),
            )
        )
        return bundles, deduped_primary_hits

    def _build_verdict_conditioned_hit_bundles(
        self,
        *,
        verdict: PremiseVerdict,
        relevant_context: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        bundles = relevant_context.get("topk_hit_bundles", [])
        bundle_lookup: Dict[str, Dict[str, Any]] = {}
        for bundle in bundles:
            primary_hit = bundle.get("primary_hit", {})
            bundle_key = f"{primary_hit.get('source', '')}:{primary_hit.get('id', '')}"
            if bundle_key:
                bundle_lookup[bundle_key] = bundle

        usable_ids = set(self._unique_preserve_order(verdict.usable_current_basis_item_ids))
        blocked_ids = set(self._unique_preserve_order(verdict.blocked_old_basis_item_ids))
        historical_ids = set(self._unique_preserve_order(verdict.historical_but_overridden_item_ids))
        unknown_track_pairs = {
            (
                str(item.get("bucket", "")).strip(),
                str(item.get("local_track", "")).strip(),
            )
            for item in verdict.unknown_tracks
            if is_valid_bucket_track(str(item.get("bucket", "")).strip(), str(item.get("local_track", "")).strip())
        }

        result: Dict[str, List[Dict[str, Any]]] = {
            "usable_current_basis_bundles": [],
            "blocked_old_basis_bundles": [],
            "historical_but_overridden_bundles": [],
            "unknown_track_bundles": [],
        }
        assigned_keys: set[str] = set()

        for bundle_key, bundle in bundle_lookup.items():
            primary_hit = bundle.get("primary_hit", {})
            item_id = self._hit_item_id(primary_hit)
            track_pair = self._hit_track_pair(primary_hit)
            source = str(primary_hit.get("source", "")).strip()
            if item_id and item_id in usable_ids:
                result["usable_current_basis_bundles"].append(bundle)
                assigned_keys.add(bundle_key)
                continue
            if item_id and item_id in blocked_ids:
                result["blocked_old_basis_bundles"].append(bundle)
                assigned_keys.add(bundle_key)
                continue
            if item_id and item_id in historical_ids:
                result["historical_but_overridden_bundles"].append(bundle)
                assigned_keys.add(bundle_key)
                continue
            if source == "unknown" and track_pair in unknown_track_pairs:
                result["unknown_track_bundles"].append(bundle)
                assigned_keys.add(bundle_key)

        if verdict.status == "SUPPORTED" and not result["usable_current_basis_bundles"]:
            for item_id in self._unique_preserve_order(verdict.supporting_active_item_ids):
                for bundle_key, bundle in bundle_lookup.items():
                    if bundle_key in assigned_keys:
                        continue
                    if self._hit_item_id(bundle.get("primary_hit", {})) == item_id:
                        result["usable_current_basis_bundles"].append(bundle)
                        assigned_keys.add(bundle_key)
                        break

        if verdict.status == "OUTDATED" and not result["blocked_old_basis_bundles"]:
            for item_id in self._unique_preserve_order(verdict.supporting_active_item_ids + verdict.outdated_item_ids):
                for bundle_key, bundle in bundle_lookup.items():
                    if bundle_key in assigned_keys:
                        continue
                    if self._hit_item_id(bundle.get("primary_hit", {})) == item_id:
                        result["blocked_old_basis_bundles"].append(bundle)
                        assigned_keys.add(bundle_key)
                        break

        return result

