from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.temporal import TemporalPolicy, item_session_index, session_index_from_id
from ..memory.models import InvalidationProposal, ProfileItem, SessionDelta, StaleSupportLink, UnknownCurrent, UpdateDecision
from ..memory.schema import normalize_bucket_track_pairs
from .transitions import StoreTransitionMixin


TrackKey = Tuple[str, str]


class ProfileStore(StoreTransitionMixin):
    def __init__(self):
        self.active_items: Dict[TrackKey, List[ProfileItem]] = defaultdict(list)
        self.stale_archive: List[ProfileItem] = []
        self.unknown_current: Dict[TrackKey, UnknownCurrent] = {}
        self.stale_support_links: List[StaleSupportLink] = []
        self._counter = 0
        self.temporal_policy = TemporalPolicy()

    def clone(self) -> "ProfileStore":
        return deepcopy(self)

    def reset(self) -> None:
        self.active_items = defaultdict(list)
        self.stale_archive = []
        self.unknown_current = {}
        self.stale_support_links = []
        self._counter = 0
        self.temporal_policy = TemporalPolicy()

    def _next_item_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:05d}"

    def get_active_for_track(self, bucket: str, local_track: str) -> List[ProfileItem]:
        return list(self.active_items.get((bucket, local_track), []))

    def get_active_by_bucket(self, bucket: str) -> List[ProfileItem]:
        items: List[ProfileItem] = []
        for (bucket_name, _), track_items in self.active_items.items():
            if bucket_name == bucket:
                items.extend(track_items)
        return items

    def get_stale_for_track(self, bucket: str, local_track: str) -> List[ProfileItem]:
        return [item for item in self.stale_archive if item.bucket == bucket and item.local_track == local_track]

    def get_stale_by_bucket(self, bucket: str) -> List[ProfileItem]:
        return [item for item in self.stale_archive if item.bucket == bucket]

    def get_unknown_for_track(self, bucket: str, local_track: str) -> Optional[UnknownCurrent]:
        return self.unknown_current.get((bucket, local_track))

    def get_unknown_by_bucket(self, bucket: str) -> List[UnknownCurrent]:
        return [item for (bucket_name, _), item in self.unknown_current.items() if bucket_name == bucket]

    @staticmethod
    def _session_index_from_id(session_id: str) -> int:
        return session_index_from_id(session_id)

    @staticmethod
    def _item_session_index(item: ProfileItem) -> int:
        return item_session_index(item)

    def _is_older_than_session(self, item: ProfileItem, session_index: int) -> bool:
        return self.temporal_policy.is_older_than_session(item, session_index)

    def get_profile_subset_before_session(
        self,
        bucket_tracks: Iterable[TrackKey],
        *,
        session_index: int,
    ) -> Dict[str, List[Dict]]:
        subset = self.get_profile_subset(bucket_tracks)
        return self.filter_profile_subset_before_session(subset, session_index=session_index)

    def get_exact_profile_subset_before_session(
        self,
        bucket_tracks: Iterable[TrackKey],
        *,
        session_index: int,
    ) -> Dict[str, List[Dict]]:
        subset = self.get_exact_profile_subset(bucket_tracks)
        return self.filter_profile_subset_before_session(subset, session_index=session_index)

    @staticmethod
    def filter_profile_subset_before_session(subset: Dict[str, List[Dict]], *, session_index: int) -> Dict[str, List[Dict]]:
        if session_index < 0:
            return deepcopy(subset)

        def _older(item: Dict[str, Any]) -> bool:
            try:
                item_index = int(item.get("created_session_index", -1))
            except (TypeError, ValueError):
                item_index = -1
            if item_index < 0:
                return True
            return item_index < session_index

        return {
            "active_items": [dict(item) for item in subset.get("active_items", []) if _older(item)],
            "stale_items": [dict(item) for item in subset.get("stale_items", []) if _older(item)],
            "unknown_items": [dict(item) for item in subset.get("unknown_items", []) if _older(item)],
        }

    def retrieve_candidates_for_track_before_session(
        self,
        bucket: str,
        local_track: str,
        *,
        session_index: int,
    ) -> Dict[str, List[ProfileItem] | UnknownCurrent | None]:
        candidate_groups = self.retrieve_candidates_for_track(bucket, local_track)
        same_track_active = [
            item for item in candidate_groups["same_track_active"]
            if self._is_older_than_session(item, session_index)
        ]
        same_bucket_active = [
            item for item in candidate_groups["same_bucket_active"]
            if self._is_older_than_session(item, session_index)
        ]
        same_track_stale = [
            item for item in candidate_groups["same_track_stale"]
            if self._is_older_than_session(item, session_index)
        ]
        unknown_item = candidate_groups["unknown_current"]
        if unknown_item is not None:
            try:
                unknown_index = int(getattr(unknown_item, "created_session_index", -1))
            except (TypeError, ValueError):
                unknown_index = -1
            if unknown_index >= 0 and session_index >= 0 and unknown_index >= session_index:
                unknown_item = None
        return {
            "same_track_active": same_track_active,
            "same_bucket_active": same_bucket_active,
            "same_track_stale": same_track_stale,
            "unknown_current": unknown_item,
        }

    def get_exact_profile_subset(self, bucket_tracks: Iterable[TrackKey]) -> Dict[str, List[Dict]]:
        normalized = normalize_bucket_track_pairs(bucket_tracks)
        active: List[Dict] = []
        stale: List[Dict] = []
        unknown: List[Dict] = []
        for bucket, local_track in normalized:
            active.extend(item.to_dict() for item in self.get_active_for_track(bucket, local_track))
            stale.extend(item.to_dict() for item in self.get_stale_for_track(bucket, local_track))
            unknown_item = self.get_unknown_for_track(bucket, local_track)
            if unknown_item is not None:
                unknown.append(unknown_item.to_dict())
        dedup_active = {item["item_id"]: item for item in active}
        dedup_stale = {item["item_id"]: item for item in stale}
        dedup_unknown = {(item["bucket"], item["local_track"]): item for item in unknown}
        return {
            "active_items": list(dedup_active.values()),
            "stale_items": list(dedup_stale.values()),
            "unknown_items": list(dedup_unknown.values()),
        }

    def get_profile_subset(self, bucket_tracks: Iterable[TrackKey]) -> Dict[str, List[Dict]]:
        normalized = normalize_bucket_track_pairs(bucket_tracks)
        active: List[Dict] = []
        stale: List[Dict] = []
        unknown: List[Dict] = []
        for bucket, local_track in normalized:
            active.extend(item.to_dict() for item in self.get_active_for_track(bucket, local_track))
            stale.extend(item.to_dict() for item in self.get_stale_for_track(bucket, local_track))
            unknown_item = self.get_unknown_for_track(bucket, local_track)
            if unknown_item is not None:
                unknown.append(unknown_item.to_dict())
            for item in self.get_active_by_bucket(bucket):
                if item.local_track != local_track:
                    active.append(item.to_dict())
            for item in self.get_stale_by_bucket(bucket):
                if item.local_track != local_track:
                    stale.append(item.to_dict())
            for item in self.get_unknown_by_bucket(bucket):
                if item.local_track != local_track:
                    unknown.append(item.to_dict())
        dedup_active = {item["item_id"]: item for item in active}
        dedup_stale = {item["item_id"]: item for item in stale}
        dedup_unknown = {(item["bucket"], item["local_track"]): item for item in unknown}
        return {
            "active_items": list(dedup_active.values()),
            "stale_items": list(dedup_stale.values()),
            "unknown_items": list(dedup_unknown.values()),
        }

    def retrieve_candidates_for_track(self, bucket: str, local_track: str) -> Dict[str, List[ProfileItem] | UnknownCurrent | None]:
        same_track_active = self.get_active_for_track(bucket, local_track)
        same_bucket_active = [
            item
            for item in self.get_active_by_bucket(bucket)
            if item.local_track != local_track
        ]
        same_track_stale = self.get_stale_for_track(bucket, local_track)
        unknown_item = self.get_unknown_for_track(bucket, local_track)
        return {
            "same_track_active": same_track_active,
            "same_bucket_active": same_bucket_active,
            "same_track_stale": same_track_stale,
            "unknown_current": unknown_item,
        }

    def retrieve_candidates(self, delta: SessionDelta) -> Dict[str, List[ProfileItem] | UnknownCurrent | None]:
        return self.retrieve_candidates_for_track(delta.bucket, delta.local_track)

    def get_all_stale_items(self) -> List[ProfileItem]:
        return list(self.stale_archive)

    def get_stale_item(self, item_id: str) -> Optional[ProfileItem]:
        for item in self.stale_archive:
            if item.item_id == item_id:
                return item
        return None

    def to_snapshot(self) -> Dict:
        return {
            "active_profile": {
                f"{bucket}/{local_track}": [item.to_dict() for item in items]
                for (bucket, local_track), items in self.active_items.items()
            },
            "stale_archive": [item.to_dict() for item in self.stale_archive],
            "unknown_current": [item.to_dict() for item in self.unknown_current.values()],
            "stale_support_links": [link.to_dict() for link in self.stale_support_links],
        }
