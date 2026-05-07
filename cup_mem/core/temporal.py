from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SessionContext:
    """Temporal context for one session write."""

    session_id: str
    session_index: int
    session_time: str = ""

    @classmethod
    def from_session_id(cls, session_id: str, *, session_time: str = "") -> "SessionContext":
        return cls(
            session_id=str(session_id or ""),
            session_index=session_index_from_id(session_id),
            session_time=str(session_time or ""),
        )


def session_index_from_id(session_id: str) -> int:
    text = str(session_id or "").strip()
    if text.startswith("s_"):
        text = text[2:]
    try:
        return int(text)
    except (TypeError, ValueError):
        return -1


def item_session_index(item: Any) -> int:
    if item is None:
        return -1
    getter = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
    try:
        index = int(getter("created_session_index", -1))
    except (TypeError, ValueError):
        index = -1
    if index >= 0:
        return index

    history = getter("revision_history", []) or []
    for event in reversed(history):
        if not isinstance(event, dict):
            continue
        source_session = str(event.get("source_session_id", "") or event.get("session_id", "")).strip()
        if source_session:
            return session_index_from_id(source_session)
    return -1


@dataclass(frozen=True)
class TemporalPolicy:
    """Temporal causality policy for writes and links."""

    allow_unknown_legacy_time: bool = True

    def is_older_than_session(self, item: Any, session_index: int) -> bool:
        if session_index < 0:
            return True
        target_index = item_session_index(item)
        if target_index < 0:
            return self.allow_unknown_legacy_time
        return target_index < session_index

    def can_invalidate(self, *, trigger_session_index: int, target_item: Any) -> bool:
        return self.is_older_than_session(target_item, trigger_session_index)

    def can_link_successor(self, *, linking_session_index: int, stale_item: Any) -> bool:
        return self.is_older_than_session(stale_item, linking_session_index)

    def guard_payload(
        self,
        *,
        target_item: Any,
        trigger_session_index: int,
        target_item_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        if self.can_invalidate(trigger_session_index=trigger_session_index, target_item=target_item):
            return None
        return {
            "target_item_id": target_item_id or str(getattr(target_item, "item_id", "")),
            "target_created_session_index": item_session_index(target_item),
            "trigger_session_index": trigger_session_index,
        }
