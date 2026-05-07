from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class TrackSchema:
    name: str
    cardinality: str
    description: str


@dataclass(frozen=True)
class BucketSchema:
    name: str
    description: str
    tracks: Dict[str, TrackSchema]


def _bucket(name: str, description: str, tracks: List[TrackSchema]) -> BucketSchema:
    return BucketSchema(name=name, description=description, tracks={track.name: track for track in tracks})


BUCKET_SCHEMAS: Dict[str, BucketSchema] = {
    "identity_and_background": _bucket(
        "identity_and_background",
        "Relatively stable identity background, roles, and skill/language background.",
        [
            TrackSchema("core_identity_or_role", "multi", "Stable role or identity background."),
            TrackSchema("skill_or_language_background", "multi", "Skills, language, or professional background."),
            TrackSchema("stable_social_context", "multi", "Relatively stable interpersonal or social background."),
            TrackSchema("current_status_or_affiliation", "multi", "Current status, affiliation, or relationship status."),
        ],
    ),
    "stable_preferences": _bucket(
        "stable_preferences",
        "Long-term preferences, value tendencies, and stable habits.",
        [
            TrackSchema("enduring_preference", "multi", "Long-term preference."),
            TrackSchema("habitual_choice_pattern", "multi", "Stable choice pattern."),
            TrackSchema("value_or_priority_tendency", "multi", "Value orientation or priority tendency."),
        ],
    ),
    "location_and_living": _bucket(
        "location_and_living",
        "Current location, moving status, and living/settlement state.",
        [
            TrackSchema("current_base_location", "single", "Current primary location or residence."),
            TrackSchema("living_arrangement_or_settlement", "single", "Current settlement, post-move setup, or living arrangement."),
            TrackSchema("location_linked_condition", "multi", "Current practical conditions linked to location."),
        ],
    ),
    "weather_and_environment": _bucket(
        "weather_and_environment",
        "Current weather and external environmental conditions.",
        [
            TrackSchema("current_weather_pattern", "single", "Current dominant weather pattern."),
            TrackSchema("environmental_condition", "multi", "Current environmental condition."),
            TrackSchema("weather_linked_adjustment", "multi", "Current adjustment driven by weather."),
        ],
    ),
    "health_and_mobility": _bucket(
        "health_and_mobility",
        "Current physical state, mobility, and recovery period.",
        [
            TrackSchema("current_health_state", "single", "Current primary health state."),
            TrackSchema("functional_limitation", "multi", "Current functional or mobility limitation."),
            TrackSchema("health_linked_adjustment", "multi", "Practical adjustment caused by health state."),
        ],
    ),
    "work_and_schedule": _bucket(
        "work_and_schedule",
        "Current workload and time bandwidth.",
        [
            TrackSchema("current_workload", "multi", "Current workload state."),
            TrackSchema("schedule_pressure_or_bandwidth", "single", "Current time bandwidth or scheduling pressure."),
            TrackSchema("work_transition_or_change", "multi", "Work change or transition."),
            TrackSchema("standing_commitment_or_availability", "multi", "Stable recurring commitments or fixed availability/occupancy patterns."),
        ],
    ),
    "finance_and_resources": _bucket(
        "finance_and_resources",
        "Budget, practical resources, and availability conditions.",
        [
            TrackSchema("financial_constraint", "multi", "Financial constraint."),
            TrackSchema("resource_availability", "multi", "Currently available resource."),
            TrackSchema("resource_linked_adjustment", "multi", "Current adjustment driven by resources."),
            TrackSchema("resource_access_or_recoverability", "multi", "Accessibility, retention, or recoverability of resources, digital assets, subscriptions, or tools."),
        ],
    ),
    "family_and_caregiving": _bucket(
        "family_and_caregiving",
        "Family responsibilities and caregiving obligations.",
        [
            TrackSchema("caregiving_responsibility", "multi", "Caregiving responsibility."),
            TrackSchema("household_obligation", "multi", "Household or domestic obligation."),
            TrackSchema("family_linked_constraint", "multi", "Current constraint linked to family factors."),
        ],
    ),
    "routine_and_transport": _bucket(
        "routine_and_transport",
        "Routine routes, transport mode, and travel conditions.",
        [
            TrackSchema("current_commute_mode", "single", "Current primary commute or travel mode."),
            TrackSchema("transport_access_condition", "multi", "Transport accessibility or travel condition."),
            TrackSchema("routine_shift", "multi", "Routine or route shift."),
        ],
    ),
    "current_focus_and_goals": _bucket(
        "current_focus_and_goals",
        "Current main focus, near-term goals, and practical concerns.",
        [
            TrackSchema("current_primary_focus", "multi", "Current primary focus."),
            TrackSchema("short_horizon_goal", "multi", "Short-horizon goal."),
            TrackSchema("goal_linked_constraint", "multi", "Current constraint linked to a goal."),
        ],
    ),
}


def schema_prompt_text() -> str:
    lines: List[str] = []
    for bucket_name, bucket in BUCKET_SCHEMAS.items():
        lines.append(f"- {bucket_name}: {bucket.description}")
        for track_name, track in bucket.tracks.items():
            lines.append(
                f"  - {track_name} [{track.cardinality}]: {track.description}"
            )
    return "\n".join(lines)


def is_valid_bucket_track(bucket: str, local_track: str) -> bool:
    bucket_schema = BUCKET_SCHEMAS.get(bucket)
    return bucket_schema is not None and local_track in bucket_schema.tracks


def is_valid_bucket(bucket: str) -> bool:
    return bucket in BUCKET_SCHEMAS


def get_track_cardinality(bucket: str, local_track: str) -> str:
    bucket_schema = BUCKET_SCHEMAS.get(bucket)
    if bucket_schema is None:
        return "multi"
    track = bucket_schema.tracks.get(local_track)
    if track is None:
        return "multi"
    return track.cardinality


def normalize_bucket_track_pairs(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    result: List[Tuple[str, str]] = []
    for bucket, local_track in pairs:
        if not is_valid_bucket_track(bucket, local_track):
            continue
        key = (bucket, local_track)
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def bucket_track_pairs_for_bucket(bucket: str) -> List[Tuple[str, str]]:
    bucket_schema = BUCKET_SCHEMAS.get(bucket)
    if bucket_schema is None:
        return []
    return [(bucket, track_name) for track_name in bucket_schema.tracks.keys()]


def linked_tracks() -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for bucket_name, bucket in BUCKET_SCHEMAS.items():
        for track_name in bucket.tracks:
            if "linked_" in track_name:
                results.append((bucket_name, track_name))
    return results


def parent_bucket_for_track(track_name: str) -> str:
    for bucket_name, bucket in BUCKET_SCHEMAS.items():
        if track_name in bucket.tracks:
            return bucket_name
    return ""
