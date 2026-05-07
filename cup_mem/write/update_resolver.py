from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple

from ..memory.models import InvalidationProposal, SessionDelta, UpdateDecision
from ..prompt_lib.templates import UPDATE_JUDGE_PROMPT
from ..store_layer import ProfileStore
from ..memory.schema import get_track_cardinality


class UpdateResolverMixin:
    def proposal_text(proposal: InvalidationProposal) -> str:
        return (
            f"proposal_type: {proposal.proposal_type}\n"
            f"target_bucket: {proposal.target_bucket}\n"
            f"target_local_track: {proposal.target_local_track}\n"
            f"target_item_id_hint: {proposal.target_item_id_hint}\n"
            f"target_old_value: {proposal.target_old_value}\n"
            f"reason: {proposal.reason}"
        )

    @staticmethod
    def _is_semantic_accept(applied_action: str) -> bool:
        normalized = str(applied_action or "").strip().upper()
        return normalized in {
            "ADD",
            "REFINE",
            "REPLACE",
            "INDIRECT_INVALIDATE",
            "SET_UNKNOWN_CURRENT",
            "WEAK_CHALLENGE",
            "WEAK_CHALLENGE_IDEMPOTENT",
            "ADD_RESCUE_SINGLE_TRACK_REPLACE",
        }

    @staticmethod
    def _normalize_basis_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return normalized.strip()

    @staticmethod
    def _same_session_action_priority(decision: str) -> int:
        normalized = str(decision or "").strip().upper()
        priority_map = {
            "REPLACE": 70,
            "INDIRECT_INVALIDATE": 60,
            "REFINE": 50,
            "ADD": 40,
            "WEAK_CHALLENGE": 30,
            "SET_UNKNOWN_CURRENT": 20,
            "NO_OP": 0,
        }
        return priority_map.get(normalized, 0)

    def _same_session_action_confidence(
        self,
        *,
        source: str,
        record: Dict[str, Any],
    ) -> float:
        if source == "update":
            return float(record["delta"].confidence)
        return float(record["proposal"].confidence)

    def _same_session_action_specificity(
        self,
        *,
        source: str,
        record: Dict[str, Any],
    ) -> int:
        decision = record["judge_result"]["decision"]
        if str(decision.target_item_id or "").strip():
            return 3
        if source == "update":
            delta = record["delta"]
            if str(delta.proposed_value or "").strip():
                return 2
            return 1
        proposal = record["proposal"]
        if str(proposal.target_old_value or "").strip():
            return 2
        if str(decision.reason or proposal.reason or "").strip():
            return 1
        return 0

    @staticmethod
    def _same_session_action_track(
        *,
        source: str,
        record: Dict[str, Any],
    ) -> Tuple[str, str]:
        if source == "update":
            delta = record["delta"]
            return delta.bucket, delta.local_track
        proposal = record["proposal"]
        return proposal.target_bucket, proposal.target_local_track

    def _arbitrate_same_session_actions(
        self,
        *,
        update_decision_records: List[Dict[str, Any]],
        invalidation_decision_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        suppressed_records: Dict[Tuple[str, int], Dict[str, Any]] = {}
        arbitration_logs: List[Dict[str, Any]] = []
        item_grouped_actions: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        slot_grouped_actions: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

        for source, records in [
            ("update", update_decision_records),
            ("invalidation", invalidation_decision_records),
        ]:
            for record_index, record in enumerate(records):
                decision = record["judge_result"]["decision"]
                if str(decision.decision or "").strip().upper() == "NO_OP":
                    continue
                bucket, local_track = self._same_session_action_track(source=source, record=record)
                action_entry = {
                    "source": source,
                    "record_index": record_index,
                    "decision": str(decision.decision or "").strip().upper(),
                    "target_item_id": str(decision.target_item_id or "").strip(),
                    "reason": str(decision.reason or "").strip(),
                    "priority": self._same_session_action_priority(decision.decision),
                    "specificity": self._same_session_action_specificity(source=source, record=record),
                    "confidence": self._same_session_action_confidence(source=source, record=record),
                    "bucket": bucket,
                    "local_track": local_track,
                }
                if action_entry["target_item_id"]:
                    item_grouped_actions.setdefault(
                        ("item", action_entry["target_item_id"]),
                        [],
                    ).append(action_entry)

                if get_track_cardinality(bucket, local_track) == "single":
                    slot_grouped_actions.setdefault(("slot", bucket, local_track), []).append(action_entry)

        for arbitration_key, actions in item_grouped_actions.items():
            if len(actions) <= 1:
                continue
            ranked_actions = sorted(
                actions,
                key=lambda item: (
                    item["priority"],
                    item["specificity"],
                    item["confidence"],
                    1 if item["source"] == "invalidation" else 0,
                ),
                reverse=True,
            )
            winner = ranked_actions[0]
            arbitration_logs.append(
                {
                    "phase": "old_item_group",
                    "arbitration_key": list(arbitration_key),
                    "winner": winner,
                    "candidates": ranked_actions,
                }
            )
            for loser in ranked_actions[1:]:
                loser_key = (loser["source"], loser["record_index"])
                if loser_key in suppressed_records:
                    continue
                suppressed_records[loser_key] = {
                    "suppression_reason": "same_target_item_competing_action",
                    "winner_source": winner["source"],
                    "winner_record_index": winner["record_index"],
                    "winner_decision": winner["decision"],
                    "arbitration_key": list(arbitration_key),
                }

        for arbitration_key, actions in slot_grouped_actions.items():
            remaining_actions = [
                action for action in actions if (action["source"], action["record_index"]) not in suppressed_records
            ]
            if len(remaining_actions) <= 1:
                continue
            grounded_updates = [
                action
                for action in remaining_actions
                if action["source"] == "update"
                and action["decision"] in {"ADD", "REFINE", "REPLACE"}
            ]
            slot_level_invalidations = [
                action
                for action in remaining_actions
                if action["source"] == "invalidation"
                and not action["target_item_id"]
                and action["decision"] in {"INDIRECT_INVALIDATE", "SET_UNKNOWN_CURRENT"}
            ]
            if grounded_updates and slot_level_invalidations:
                ranked_updates = sorted(
                    grounded_updates,
                    key=lambda item: (
                        item["specificity"],
                        item["priority"],
                        item["confidence"],
                    ),
                    reverse=True,
                )
                winner = ranked_updates[0]
                arbitration_logs.append(
                    {
                        "phase": "single_track_slot_update_over_slot_invalidation",
                        "arbitration_key": list(arbitration_key),
                        "winner": winner,
                        "suppressed_slot_level_invalidations": slot_level_invalidations,
                        "grounded_update_candidates": ranked_updates,
                        "reason": "A grounded same-session current-state update should not be suppressed by a non-exact slot-level invalidation on the same single-track slot.",
                    }
                )
                for loser in slot_level_invalidations:
                    loser_key = (loser["source"], loser["record_index"])
                    if loser_key in suppressed_records:
                        continue
                    suppressed_records[loser_key] = {
                        "suppression_reason": "single_track_slot_invalidation_superseded_by_grounded_update",
                        "winner_source": winner["source"],
                        "winner_record_index": winner["record_index"],
                        "winner_decision": winner["decision"],
                        "arbitration_key": list(arbitration_key),
                    }
                remaining_actions = [
                    action for action in remaining_actions if (action["source"], action["record_index"]) not in suppressed_records
                ]
                if len(remaining_actions) <= 1:
                    continue
            preserve_update_ids: set[Tuple[str, int]] = set()
            coexist_invalidation_ids: set[Tuple[str, int]] = set()
            target_specific_invalidations = [
                action
                for action in remaining_actions
                if action["source"] == "invalidation"
                and action["target_item_id"]
                and action["decision"] in {"INDIRECT_INVALIDATE", "WEAK_CHALLENGE"}
            ]
            if grounded_updates and target_specific_invalidations:
                for action in grounded_updates:
                    if action["decision"] == "ADD" and float(action.get("confidence", 0.0) or 0.0) >= 0.85:
                        preserve_update_ids.add((action["source"], action["record_index"]))
                if preserve_update_ids:
                    coexist_invalidation_ids = {
                        (action["source"], action["record_index"])
                        for action in target_specific_invalidations
                    }
                    arbitration_logs.append(
                        {
                            "phase": "single_track_preserve_grounded_update_with_target_invalidation",
                            "arbitration_key": list(arbitration_key),
                            "preserved_updates": [
                                action
                                for action in grounded_updates
                                if (action["source"], action["record_index"]) in preserve_update_ids
                            ],
                            "target_specific_invalidations": target_specific_invalidations,
                            "reason": "A concrete high-confidence ADD may be the current basis that also makes an old item unsafe; keep it eligible instead of making invalidation and new basis mutually exclusive.",
                        }
                    )
            ranked_actions = sorted(
                remaining_actions,
                key=lambda item: (
                    item["specificity"],
                    item["priority"],
                    item["confidence"],
                    1 if item["source"] == "update" else 0,
                ),
                reverse=True,
            )
            winner = ranked_actions[0]
            arbitration_logs.append(
                {
                    "phase": "single_track_slot_group",
                    "arbitration_key": list(arbitration_key),
                    "winner": winner,
                    "candidates": ranked_actions,
                }
            )
            for loser in ranked_actions[1:]:
                loser_key = (loser["source"], loser["record_index"])
                if loser_key in suppressed_records:
                    continue
                if loser_key in preserve_update_ids:
                    arbitration_logs.append(
                        {
                            "phase": "single_track_preserved_update_not_suppressed",
                            "arbitration_key": list(arbitration_key),
                            "preserved_update": loser,
                            "winner": winner,
                        }
                    )
                    continue
                if loser_key in coexist_invalidation_ids:
                    arbitration_logs.append(
                        {
                            "phase": "single_track_target_invalidation_not_suppressed",
                            "arbitration_key": list(arbitration_key),
                            "preserved_invalidation": loser,
                            "winner": winner,
                        }
                    )
                    continue
                suppressed_records[loser_key] = {
                    "suppression_reason": "single_track_slot_competing_action",
                    "winner_source": winner["source"],
                    "winner_record_index": winner["record_index"],
                    "winner_decision": winner["decision"],
                    "arbitration_key": list(arbitration_key),
                }

        return {
            "suppressed_records": suppressed_records,
            "arbitration_logs": arbitration_logs,
        }

    def judge_update(self, delta: SessionDelta, *, store_view: Optional[ProfileStore] = None) -> Dict[str, Any]:
        store_ref = store_view or self.store
        candidate_groups = store_ref.retrieve_candidates(delta)
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
            query_text=self.delta_text(delta),
            candidates=candidate_groups["same_track_active"],
            text_getter=lambda item: self.profile_item_text(item.to_dict()),
            top_k=self.thresholds.candidate_top_k,
        )
        ranked_same_bucket = self.embedding.rank(
            query_text=self.delta_text(delta),
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
        unknown_item = candidate_groups["unknown_current"]
        payload = self._llm_call_json(
            phase="update_judge",
            system_prompt=UPDATE_JUDGE_PROMPT,
            user_payload=json.dumps(
                {
                    "bucket_track_schema": self.schema_text,
                    "delta": delta.to_dict(),
                    "track_cardinality": get_track_cardinality(delta.bucket, delta.local_track),
                    "same_track_unknown_current": unknown_item.to_dict() if unknown_item is not None else None,
                    "same_track_stale_items": [item.to_dict() for item in candidate_groups["same_track_stale"]],
                    "same_track_target_candidates": same_track_payload,
                    "same_bucket_context_items": same_bucket_payload,
                    "candidate_existing_items": same_track_payload + same_bucket_payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            session_id=delta.session_id,
        )
        decision = str(payload.get("decision", "")).strip().upper()
        if decision not in {"ADD", "REFINE", "REPLACE", "INDIRECT_INVALIDATE", "NO_OP"}:
            decision = "NO_OP"
        target_item_id = str(payload.get("target_item_id", "")).strip()
        valid_target_ids = {item["item_id"] for item in same_track_payload}
        if target_item_id not in valid_target_ids:
            target_item_id = ""
        decision_obj = UpdateDecision(
            decision=decision,
            target_item_id=target_item_id,
            reason=str(payload.get("reason", "")).strip(),
        )
        return {
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
