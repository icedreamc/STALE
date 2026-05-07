from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..memory.models import InvalidationProposal, UpdateDecision
from .update_resolver import UpdateResolverMixin
from ..prompt_lib.templates import INVALIDATION_JUDGE_PROMPT
from ..memory.schema import get_track_cardinality
from ..store_layer import ProfileStore


class InvalidationJudgeMixin:
    """Final LLM judge over a selected invalidation proposal and candidate items."""

    def judge_invalidation(
        self,
        proposal: InvalidationProposal,
        *,
        store_view: Optional[ProfileStore] = None,
    ) -> Dict[str, Any]:
        store_ref = store_view or self.store
        candidate_groups = store_ref.retrieve_candidates_for_track_before_session(
            proposal.target_bucket,
            proposal.target_local_track,
            session_index=self._session_index_from_session_id(proposal.session_id),
        )
        source_group_by_item_id = {
            item.item_id: "same_track_active"
            for item in candidate_groups["same_track_active"]
        }
        source_group_by_item_id.update(
            {
                item.item_id: "same_bucket_active"
                for item in candidate_groups["same_bucket_active"]
            }
        )
        ranked_same_track = self.embedding.rank(
            query_text=UpdateResolverMixin.proposal_text(proposal),
            candidates=candidate_groups["same_track_active"],
            text_getter=lambda item: self.profile_item_text(item.to_dict()),
            top_k=self.thresholds.candidate_top_k,
        )
        ranked_same_bucket = self.embedding.rank(
            query_text=UpdateResolverMixin.proposal_text(proposal),
            candidates=candidate_groups["same_bucket_active"],
            text_getter=lambda item: self.profile_item_text(item.to_dict()),
            top_k=self.thresholds.candidate_top_k,
        )
        same_track_payload = []
        for rank_index, entry in enumerate(ranked_same_track, start=1):
            item_dict = entry["item"].to_dict()
            item_dict.update(
                {
                    "retrieval_score": entry["score"],
                    "rank_index": rank_index,
                    "candidate_source_group": source_group_by_item_id.get(item_dict["item_id"], "unknown"),
                }
            )
            same_track_payload.append(item_dict)
        same_bucket_payload = []
        for rank_index, entry in enumerate(ranked_same_bucket, start=1):
            item_dict = entry["item"].to_dict()
            item_dict.update(
                {
                    "retrieval_score": entry["score"],
                    "rank_index": rank_index,
                    "candidate_source_group": source_group_by_item_id.get(item_dict["item_id"], "unknown"),
                }
            )
            same_bucket_payload.append(item_dict)
        same_track_target_ids = {item["item_id"] for item in same_track_payload}
        same_bucket_target_ids = {item["item_id"] for item in same_bucket_payload}
        hint_provided = bool(proposal.target_item_id_hint)
        hint_hit_same_track = proposal.target_item_id_hint in same_track_target_ids if hint_provided else False
        hint_hit_same_bucket = proposal.target_item_id_hint in same_bucket_target_ids if hint_provided else False
        hint_hit_active_candidate = hint_hit_same_track or hint_hit_same_bucket
        unknown_item = candidate_groups["unknown_current"]
        payload = self._llm_call_json(
            phase="invalidation_judge",
            system_prompt=INVALIDATION_JUDGE_PROMPT,
            user_payload=json.dumps(
                {
                    "bucket_track_schema": self.schema_text,
                    "latent_invalidation_proposal": proposal.to_dict(),
                    "track_cardinality": get_track_cardinality(proposal.target_bucket, proposal.target_local_track),
                    "same_track_unknown_current": unknown_item.to_dict() if unknown_item is not None else None,
                    "same_track_stale_items": [item.to_dict() for item in candidate_groups["same_track_stale"]],
                    "same_track_target_candidates": same_track_payload,
                    "same_bucket_context_items": same_bucket_payload,
                    "target_item_id_hint_status": {
                        "hint_provided": hint_provided,
                        "hint_hit_same_track": hint_hit_same_track,
                        "hint_hit_same_bucket": hint_hit_same_bucket,
                        "hint_hit_active_candidate": hint_hit_active_candidate,
                    },
                    "candidate_existing_items": same_track_payload + same_bucket_payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            session_id=proposal.session_id,
        )
        decision = str(payload.get("decision", "")).strip().upper()
        if decision not in {"INDIRECT_INVALIDATE", "WEAK_CHALLENGE", "SET_UNKNOWN_CURRENT", "NO_OP"}:
            decision = "NO_OP"
        target_item_id = str(payload.get("target_item_id", "")).strip()
        valid_target_ids = set(same_track_target_ids)
        if hint_hit_active_candidate:
            valid_target_ids.add(proposal.target_item_id_hint)
        if target_item_id not in valid_target_ids:
            target_item_id = proposal.target_item_id_hint if proposal.target_item_id_hint in valid_target_ids else ""
        decision_obj = UpdateDecision(
            decision=decision,
            target_item_id=target_item_id,
            reason=str(payload.get("reason", "")).strip() or proposal.reason,
        )
        result = {
            "candidate_groups": {
                "same_track_active": [item.to_dict() for item in candidate_groups["same_track_active"]],
                "same_bucket_active": [item.to_dict() for item in candidate_groups["same_bucket_active"]],
                "same_track_stale": [item.to_dict() for item in candidate_groups["same_track_stale"]],
                "unknown_current": unknown_item.to_dict() if unknown_item is not None else None,
            },
            "candidate_pool_summary": self._candidate_group_summary(
                candidate_groups=candidate_groups,
                ranked_candidate_count=len(same_track_payload) + len(same_bucket_payload),
            ),
            "ranked_candidates": same_track_payload + same_bucket_payload,
            "raw_payload": payload,
            "decision": decision_obj,
        }
        return self._attach_debug_trace(
            result,
            section="write",
            fields={
                "target_item_id_hint_status": {
                    "hint_provided": hint_provided,
                    "hint_hit_same_track": hint_hit_same_track,
                    "hint_hit_same_bucket": hint_hit_same_bucket,
                    "hint_hit_active_candidate": hint_hit_active_candidate,
                    "allowed_target_item_ids": sorted(valid_target_ids),
                }
            },
        )
