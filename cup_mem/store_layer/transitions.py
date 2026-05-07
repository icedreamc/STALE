from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from ..memory.models import InvalidationProposal, ProfileItem, SessionDelta, StaleSupportLink, UnknownCurrent, UpdateDecision
from ..memory.schema import get_track_cardinality


class StoreTransitionMixin:
    """ProfileStore state transitions."""

    def _archive_active_item(self, item: ProfileItem, *, reason: str, delta_id: str, timestamp: str) -> None:
        archived = deepcopy(item)
        archived.status = "STALE"
        archived.last_updated = timestamp
        archived.revision_history = archived.revision_history + [
            {
                "event": "archived",
                "reason": reason,
                "invalidated_by_delta_id": delta_id,
                "timestamp": timestamp,
            }
        ]
        self.stale_archive.append(archived)

    def _clear_unknown(self, bucket: str, local_track: str) -> None:
        self.unknown_current.pop((bucket, local_track), None)

    def _make_active_item(self, delta: SessionDelta, *, timestamp: str) -> ProfileItem:
        return ProfileItem(
            item_id=self._next_item_id("p"),
            bucket=delta.bucket,
            local_track=delta.local_track,
            value=delta.proposed_value,
            status="ACTIVE",
            confidence=delta.confidence,
            first_seen=timestamp,
            last_updated=timestamp,
            source_delta_ids=[delta.delta_id],
            evidence_chunk_ids=list(delta.evidence_chunk_ids),
            created_session_id=delta.session_id,
            created_session_index=delta.session_index,
            created_session_time=delta.session_time,
            active_strength="STRONG",
        )

    def _find_active_item(self, item_id: str) -> Optional[ProfileItem]:
        for items in self.active_items.values():
            for item in items:
                if item.item_id == item_id:
                    return item
        return None

    def _remove_active_item(self, item_id: str) -> None:
        for key, items in list(self.active_items.items()):
            filtered = [item for item in items if item.item_id != item_id]
            if len(filtered) != len(items):
                self.active_items[key] = filtered
                return

    def apply_update(self, delta: SessionDelta, decision: UpdateDecision, *, timestamp: str) -> Dict:
        key = (delta.bucket, delta.local_track)
        before = self.to_snapshot()
        applied_action = decision.decision
        if decision.decision == "NO_OP":
            return {
                "decision": decision.to_dict(),
                "applied_action": applied_action,
                "before": before,
                "after": self.to_snapshot(),
            }

        target = self._find_active_item(decision.target_item_id) if decision.target_item_id else None
        if target is not None and (target.bucket, target.local_track) != key:
            target = None

        if decision.decision == "ADD":
            current_items = list(self.active_items.get(key, []))
            if get_track_cardinality(delta.bucket, delta.local_track) == "single" and current_items:
                rescue_reason = ""
                replace_target = None
                if delta.confidence >= 0.85:
                    weakest = min(
                        current_items,
                        key=lambda item: (
                            0 if item.active_strength == "WEAK" else 1,
                            float(item.confidence),
                        ),
                    )
                    if weakest.active_strength == "WEAK":
                        replace_target = weakest
                        rescue_reason = "replace_weak_single_track_item"
                    elif delta.confidence >= max(0.0, float(weakest.confidence) + 0.15):
                        replace_target = weakest
                        rescue_reason = "replace_lower_confidence_single_track_item"
                if replace_target is not None:
                    self._remove_active_item(replace_target.item_id)
                    self._archive_active_item(
                        replace_target,
                        reason="ADD_RESCUE_SINGLE_TRACK_REPLACED",
                        delta_id=delta.delta_id,
                        timestamp=timestamp,
                    )
                    self._clear_unknown(*key)
                    self.active_items[key].append(self._make_active_item(delta, timestamp=timestamp))
                    applied_action = "ADD_RESCUE_SINGLE_TRACK_REPLACE"
                    return {
                        "decision": decision.to_dict(),
                        "applied_action": applied_action,
                        "rescue_reason": rescue_reason,
                        "replaced_item_id": replace_target.item_id,
                        "before": before,
                        "after": self.to_snapshot(),
                    }
                return {
                    "decision": decision.to_dict(),
                    "applied_action": "ADD_BLOCKED_SINGLE_TRACK",
                    "blocked_existing_item_ids": [item.item_id for item in current_items],
                    "before": before,
                    "after": self.to_snapshot(),
                }
            self._clear_unknown(*key)
            self.active_items[key].append(self._make_active_item(delta, timestamp=timestamp))

        elif decision.decision == "REFINE":
            if target is None:
                return {
                    "decision": decision.to_dict(),
                    "applied_action": "NO_OP_MISSING_TARGET",
                    "before": before,
                    "after": self.to_snapshot(),
                }
            target.revision_history.append(
                {
                    "event": "refine",
                    "previous_value": target.value,
                    "new_value": delta.proposed_value,
                    "source_delta_id": delta.delta_id,
                    "source_session_id": delta.session_id,
                    "source_session_index": delta.session_index,
                    "source_session_time": delta.session_time,
                    "timestamp": timestamp,
                    "reason": decision.reason,
                }
            )
            target.value = delta.proposed_value
            target.confidence = max(target.confidence, delta.confidence)
            target.last_updated = timestamp
            target.active_strength = "STRONG"
            if delta.delta_id not in target.source_delta_ids:
                target.source_delta_ids.append(delta.delta_id)
            for chunk_id in delta.evidence_chunk_ids:
                if chunk_id not in target.evidence_chunk_ids:
                    target.evidence_chunk_ids.append(chunk_id)
            self._clear_unknown(*key)

        elif decision.decision == "REPLACE":
            if target is None:
                return {
                    "decision": decision.to_dict(),
                    "applied_action": "NO_OP_MISSING_TARGET",
                    "before": before,
                    "after": self.to_snapshot(),
                }
            self._remove_active_item(target.item_id)
            self._archive_active_item(target, reason="REPLACED", delta_id=delta.delta_id, timestamp=timestamp)
            self._clear_unknown(*key)
            self.active_items[key].append(self._make_active_item(delta, timestamp=timestamp))

        elif decision.decision == "INDIRECT_INVALIDATE":
            if target is None:
                return {
                    "decision": decision.to_dict(),
                    "applied_action": "NO_OP_MISSING_TARGET",
                    "before": before,
                    "after": self.to_snapshot(),
                }
            self._remove_active_item(target.item_id)
            self._archive_active_item(target, reason="INDIRECT_INVALIDATE", delta_id=delta.delta_id, timestamp=timestamp)
            self.unknown_current[key] = UnknownCurrent(
                bucket=delta.bucket,
                local_track=delta.local_track,
                status="UNKNOWN_CURRENT",
                reason=decision.reason,
                invalidated_by_delta_id=delta.delta_id,
                last_updated=timestamp,
                created_session_id=delta.session_id,
                created_session_index=delta.session_index,
                created_session_time=delta.session_time,
            )

        return {
            "decision": decision.to_dict(),
            "applied_action": applied_action,
            "before": before,
            "after": self.to_snapshot(),
        }

    def _newer_target_guard_payload(
        self,
        *,
        decision: UpdateDecision,
        target: Optional[ProfileItem],
        trigger_session_index: int,
    ) -> Optional[Dict[str, Any]]:
        if target is None or trigger_session_index < 0:
            return None
        target_session_index = self._item_session_index(target)
        if target_session_index < 0 or target_session_index < trigger_session_index:
            return None
        return {
            "decision": decision.to_dict(),
            "applied_action": "NO_OP_TEMPORAL_TARGET_NOT_OLDER",
            "temporal_guard": {
                "target_item_id": target.item_id,
                "target_created_session_index": target_session_index,
                "trigger_session_index": trigger_session_index,
            },
            "before": self.to_snapshot(),
            "after": self.to_snapshot(),
        }

    def apply_weak_challenge(self, proposal: InvalidationProposal, decision: UpdateDecision, *, timestamp: str) -> Dict:
        before = self.to_snapshot()
        target = self._find_active_item(decision.target_item_id) if decision.target_item_id else None
        if target is None:
            return {
                "decision": decision.to_dict(),
                "applied_action": "NO_OP_MISSING_TARGET",
                "before": before,
                "after": self.to_snapshot(),
            }
        guard_payload = self._newer_target_guard_payload(
            decision=decision,
            target=target,
            trigger_session_index=self._session_index_from_id(proposal.session_id),
        )
        if guard_payload is not None:
            return guard_payload
        if target.active_strength == "WEAK":
            return {
                "decision": decision.to_dict(),
                "applied_action": "WEAK_CHALLENGE_IDEMPOTENT",
                "before": before,
                "after": self.to_snapshot(),
            }
        target.active_strength = "WEAK"
        target.last_updated = timestamp
        target.revision_history.append(
            {
                "event": "weak_challenge",
                "reason": decision.reason or proposal.reason,
                "source_proposal_id": proposal.proposal_id,
                "source_delta_ids": list(proposal.trigger_delta_ids),
                "source_session_id": proposal.session_id,
                "source_session_index": self._session_index_from_id(proposal.session_id),
                "timestamp": timestamp,
            }
        )
        return {
            "decision": decision.to_dict(),
            "applied_action": "WEAK_CHALLENGE",
            "before": before,
            "after": self.to_snapshot(),
        }

    def apply_invalidation(self, proposal: InvalidationProposal, decision: UpdateDecision, *, timestamp: str) -> Dict:
        before = self.to_snapshot()
        applied_action = decision.decision
        if decision.decision == "NO_OP":
            return {
                "decision": decision.to_dict(),
                "applied_action": applied_action,
                "before": before,
                "after": self.to_snapshot(),
            }

        target = self._find_active_item(decision.target_item_id) if decision.target_item_id else None
        if decision.decision == "WEAK_CHALLENGE":
            return self.apply_weak_challenge(proposal, decision, timestamp=timestamp)

        guard_payload = self._newer_target_guard_payload(
            decision=decision,
            target=target,
            trigger_session_index=self._session_index_from_id(proposal.session_id),
        )
        if guard_payload is not None:
            return guard_payload

        if decision.decision == "INDIRECT_INVALIDATE" and target is None:
            return {
                "decision": decision.to_dict(),
                "applied_action": "NO_OP_MISSING_TARGET",
                "before": before,
                "after": self.to_snapshot(),
            }

        if target is not None:
            key = (target.bucket, target.local_track)
        else:
            key = (proposal.target_bucket, proposal.target_local_track)
        cardinality = get_track_cardinality(*key)
        trigger_session_index = self._session_index_from_id(proposal.session_id)
        current_items = [
            item
            for item in self.active_items.get(key, [])
            if self._is_older_than_session(item, trigger_session_index)
        ]

        if target is not None:
            self._remove_active_item(target.item_id)
            self._archive_active_item(target, reason=decision.decision, delta_id=proposal.proposal_id, timestamp=timestamp)
        elif decision.decision == "SET_UNKNOWN_CURRENT" and current_items and (cardinality == "single" or len(current_items) == 1):
            self.active_items[key] = []
            for item in current_items:
                self._archive_active_item(item, reason="SET_UNKNOWN_CURRENT", delta_id=proposal.proposal_id, timestamp=timestamp)

        self.unknown_current[key] = UnknownCurrent(
            bucket=key[0],
            local_track=key[1],
            status="UNKNOWN_CURRENT",
            reason=decision.reason or proposal.reason,
            invalidated_by_delta_id=proposal.proposal_id,
            last_updated=timestamp,
            created_session_id=proposal.session_id,
            created_session_index=trigger_session_index,
            created_session_time=timestamp,
        )

        return {
            "decision": decision.to_dict(),
            "applied_action": applied_action,
            "before": before,
            "after": self.to_snapshot(),
        }

    def add_stale_support_link(
        self,
        *,
        stale_item_id: str,
        linking_delta_id: str,
        link_type: str,
        reason: str,
        link_session_idx: int,
    ) -> Dict[str, Any]:
        for link in self.stale_support_links:
            if (
                link.stale_item_id == stale_item_id
                and link.linking_delta_id == linking_delta_id
                and link.link_type == link_type
            ):
                if reason:
                    link.reason = reason
                link.link_session_idx = link_session_idx
                return {
                    "status": "deduped",
                    "link": link.to_dict(),
                }
        link = StaleSupportLink(
            stale_item_id=stale_item_id,
            linking_delta_id=linking_delta_id,
            link_type=link_type,
            reason=reason,
            link_session_idx=link_session_idx,
        )
        self.stale_support_links.append(link)
        return {
            "status": "added",
            "link": link.to_dict(),
        }

