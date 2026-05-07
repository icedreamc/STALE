from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..memory.models import SessionDelta, maybe_dict_list
from ..prompt_lib.templates import STALE_SUPPORT_JUDGE_PROMPT


class StaleLinkerMixin:
    def _retrieve_stale_link_candidates(
        self,
        *,
        delta: SessionDelta,
        affected_buckets: List[Dict[str, Any]],
        invalidated_stale_items: Optional[List[Dict[str, Any]]] = None,
        limit: int = 4,
    ) -> Dict[str, Any]:
        ranked_payload: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _rank_group(items: List[Any], source_group: str, *, top_k: int) -> List[Dict[str, Any]]:
            ranked = self.embedding.rank(
                query_text=self.delta_text(delta),
                candidates=items,
                text_getter=lambda item: self.profile_item_text(item.to_dict() if hasattr(item, "to_dict") else item),
                top_k=top_k,
            )
            payload: List[Dict[str, Any]] = []
            for rank_index, entry in enumerate(ranked, start=1):
                raw_item = entry["item"]
                item_dict = raw_item.to_dict() if hasattr(raw_item, "to_dict") else dict(raw_item)
                if item_dict["item_id"] in seen_ids:
                    continue
                item_dict.update(
                    {
                        "retrieval_score": entry["score"],
                        "rank_index": rank_index,
                        "candidate_source_group": source_group,
                    }
                )
                payload.append(item_dict)
                seen_ids.add(item_dict["item_id"])
            return payload

        fresh_candidates: List[Any] = []
        for item in maybe_dict_list(invalidated_stale_items or []):
            item_id = str(item.get("item_id", "")).strip()
            if not item_id or item_id in seen_ids:
                continue
            if str(item.get("status", "")).strip() != "STALE":
                continue
            try:
                item_session_index = int(item.get("created_session_index", -1))
            except (TypeError, ValueError):
                item_session_index = -1
            if item_session_index >= 0 and delta.session_index >= 0 and item_session_index >= delta.session_index:
                continue
            fresh_candidates.append(item)
        if fresh_candidates:
            ranked_payload.extend(
                _rank_group(
                    fresh_candidates,
                    "freshly_invalidated_stale",
                    top_k=min(limit, len(fresh_candidates)),
                )
            )

        remaining = max(0, limit - len(ranked_payload))
        same_track_stale = [
            item
            for item in self.store.get_stale_for_track(delta.bucket, delta.local_track)
            if self.store._is_older_than_session(item, delta.session_index)
        ]
        if remaining > 0:
            ranked_payload.extend(_rank_group(same_track_stale, "same_track_stale", top_k=remaining))

        affected_bucket_items: List[Any] = []
        affected_bucket_names = [str(item.get("bucket", "")).strip() for item in affected_buckets if str(item.get("bucket", "")).strip()]
        for bucket in affected_bucket_names:
            for item in self.store.get_stale_by_bucket(bucket):
                if item.item_id not in seen_ids and self.store._is_older_than_session(item, delta.session_index):
                    affected_bucket_items.append(item)
        remaining = max(0, limit - len(ranked_payload))
        if remaining > 0:
            ranked_payload.extend(_rank_group(affected_bucket_items, "affected_bucket_stale", top_k=remaining))

        remaining = max(0, limit - len(ranked_payload))
        if remaining > 0:
            global_items = [
                item
                for item in self.store.get_all_stale_items()
                if item.item_id not in seen_ids and self.store._is_older_than_session(item, delta.session_index)
            ]
            ranked_payload.extend(_rank_group(global_items, "global_stale_rescue", top_k=min(2, remaining)))

        return {
            "affected_bucket_names": affected_bucket_names,
            "ranked_candidates": ranked_payload[:limit],
        }

    def link_stale_support(
        self,
        *,
        session_id: str,
        session_index: int,
        deltas: List[SessionDelta],
        affected_buckets: List[Dict[str, Any]],
        invalidated_stale_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not deltas:
            return {
                "link_logs": [],
                "summary": {
                    "processed_delta_count": 0,
                    "candidate_count": 0,
                    "added_count": 0,
                    "deduped_count": 0,
                    "linked_count": 0,
                },
            }

        link_logs: List[Dict[str, Any]] = []
        added_count = 0
        deduped_count = 0
        candidate_count = 0
        linked_count = 0
        for delta in deltas:
            candidate_result = self._retrieve_stale_link_candidates(
                delta=delta,
                affected_buckets=affected_buckets,
                invalidated_stale_items=invalidated_stale_items,
            )
            ranked_candidates = candidate_result["ranked_candidates"]
            candidate_count += len(ranked_candidates)
            for candidate in ranked_candidates:
                payload = self._llm_call_json(
                    phase="stale_support_judge",
                    system_prompt=STALE_SUPPORT_JUDGE_PROMPT,
                    user_payload=json.dumps(
                        {
                            "session_id": session_id,
                            "delta": delta.to_dict(),
                            "affected_bucket_hints": affected_buckets,
                            "stale_candidate": candidate,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    session_id=session_id,
                )
                link_type = str(payload.get("link_type", "")).strip().upper()
                if link_type not in {"STALE_SUPPORT_LINK", "STALE_SUCCESSOR_LINK", "NO_LINK"}:
                    link_type = "NO_LINK"
                reason = str(payload.get("reason", "")).strip()
                store_result = None
                if link_type != "NO_LINK":
                    linked_count += 1
                    store_result = self.store.add_stale_support_link(
                        stale_item_id=str(candidate.get("item_id", "")).strip(),
                        linking_delta_id=delta.delta_id,
                        link_type=link_type,
                        reason=reason,
                        link_session_idx=session_index,
                    )
                    if store_result["status"] == "added":
                        added_count += 1
                    elif store_result["status"] == "deduped":
                        deduped_count += 1
                link_logs.append(
                    {
                        "delta_id": delta.delta_id,
                        "stale_candidate_id": candidate.get("item_id", ""),
                        "candidate_source_group": candidate.get("candidate_source_group", ""),
                        "judge_raw_payload": payload,
                        "link_type": link_type,
                        "reason": reason,
                        "store_result": store_result,
                    }
                )
        return {
            "link_logs": link_logs,
            "summary": {
                "processed_delta_count": len(deltas),
                "candidate_count": candidate_count,
                "linked_count": linked_count,
                "added_count": added_count,
                "deduped_count": deduped_count,
            },
        }

