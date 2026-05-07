from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BucketTrackRef:
    bucket: str
    local_track: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionChunk:
    chunk_id: str
    session_id: str
    source_turn_ids: List[str]
    text: str
    chunk_type: str
    candidate_bucket_tracks: List[BucketTrackRef]
    temporal_scope: str
    state_salience: float
    evidence_strength: str
    change_signal_present: bool = False
    change_signal_summary: str = ""
    signal_strength: str = "weak"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["candidate_bucket_tracks"] = [track.to_dict() for track in self.candidate_bucket_tracks]
        return data


@dataclass
class SessionDelta:
    delta_id: str
    session_id: str
    session_index: int
    session_time: str
    bucket: str
    local_track: str
    proposed_value: str
    source_type: str
    evidence_chunk_ids: List[str]
    confidence: float
    op_hint: str = ""
    delta_kind: str = "observed_state"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InvalidationProposal:
    proposal_id: str
    session_id: str
    target_bucket: str
    target_local_track: str
    trigger_delta_ids: List[str]
    evidence_chunk_ids: List[str]
    reason: str
    confidence: float
    proposal_type: str = "INVALIDATE_OR_REASSESS"
    target_item_id_hint: str = ""
    target_old_value: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProfileItem:
    item_id: str
    bucket: str
    local_track: str
    value: str
    status: str
    confidence: float
    first_seen: str
    last_updated: str
    source_delta_ids: List[str] = field(default_factory=list)
    evidence_chunk_ids: List[str] = field(default_factory=list)
    created_session_id: str = ""
    created_session_index: int = -1
    created_session_time: str = ""
    active_strength: str = "STRONG"
    revision_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StaleSupportLink:
    stale_item_id: str
    linking_delta_id: str
    link_type: str
    reason: str
    link_session_idx: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnknownCurrent:
    bucket: str
    local_track: str
    status: str
    reason: str
    invalidated_by_delta_id: str
    last_updated: str
    created_session_id: str = ""
    created_session_index: int = -1
    created_session_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UpdateDecision:
    decision: str
    reason: str
    target_item_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryParse:
    intent_type: str
    target_bucket_tracks: List[BucketTrackRef]
    queried_proposition: str
    old_premise_focus: str = ""
    basis_recovery_focus: str = ""
    decision_basis_focus: str = ""
    presuppositions: List[Dict[str, Any]] = field(default_factory=list)
    requested_action: str = ""
    action_goal: str = ""
    action_object: str = ""
    action_options: List[str] = field(default_factory=list)
    hidden_old_setup: str = ""
    secondary_surface_details: List[str] = field(default_factory=list)
    query_assumptions: List[Dict[str, Any]] = field(default_factory=list)
    option_comparison_requires_viability_check_first: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["target_bucket_tracks"] = [track.to_dict() for track in self.target_bucket_tracks]
        return data


@dataclass
class PremiseVerdict:
    status: str
    confidence: float
    reasoning_summary: str
    supporting_active_item_ids: List[str] = field(default_factory=list)
    supporting_active_item_values: List[str] = field(default_factory=list)
    outdated_item_ids: List[str] = field(default_factory=list)
    outdated_item_values: List[str] = field(default_factory=list)
    unknown_tracks: List[Dict[str, str]] = field(default_factory=list)
    corrected_current_facts: List[str] = field(default_factory=list)
    usable_current_basis_item_ids: List[str] = field(default_factory=list)
    usable_current_basis_item_values: List[str] = field(default_factory=list)
    blocked_old_basis_item_ids: List[str] = field(default_factory=list)
    blocked_old_basis_item_values: List[str] = field(default_factory=list)
    historical_but_overridden_item_ids: List[str] = field(default_factory=list)
    historical_but_overridden_item_values: List[str] = field(default_factory=list)
    old_premise_safe: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintSet:
    hard_positive_constraints: List[str] = field(default_factory=list)
    hard_negative_constraints: List[str] = field(default_factory=list)
    unresolved_variables: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionReadinessVerdict:
    status: str
    confidence: float
    reasoning_summary: str = ""
    decisive_gate_bundle_ids: List[str] = field(default_factory=list)
    blocker_bundle_ids: List[str] = field(default_factory=list)
    supporting_basis_bundle_ids: List[str] = field(default_factory=list)
    rejected_nearby_bundle_ids: List[str] = field(default_factory=list)
    corrected_current_facts: List[str] = field(default_factory=list)
    reframe_mode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerResult:
    answer: str
    brief_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def maybe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def maybe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def maybe_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def maybe_dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def bucket_track_refs_from_payload(value: Any) -> List[BucketTrackRef]:
    refs: List[BucketTrackRef] = []
    if not isinstance(value, list):
        return refs
    for item in value:
        if not isinstance(item, dict):
            continue
        bucket = str(item.get("bucket", "")).strip()
        local_track = str(item.get("local_track", "")).strip()
        if bucket and local_track:
            refs.append(BucketTrackRef(bucket=bucket, local_track=local_track))
    return refs


def build_query_parse(payload: Dict[str, Any]) -> QueryParse:
    secondary_surface_details = maybe_str_list(payload.get("secondary_surface_details", []))
    option_viability_first = maybe_bool(
        payload.get(
            "option_comparison_requires_viability_check_first",
            payload.get("option_comparison_requires_action_viability_first", False),
        ),
        False,
    )
    return QueryParse(
        intent_type=str(payload.get("intent_type", "")).strip(),
        target_bucket_tracks=bucket_track_refs_from_payload(payload.get("target_bucket_tracks", [])),
        queried_proposition=str(payload.get("queried_proposition", "")).strip(),
        old_premise_focus=str(payload.get("old_premise_focus", "")).strip(),
        basis_recovery_focus=str(payload.get("basis_recovery_focus", "")).strip(),
        decision_basis_focus=str(payload.get("decision_basis_focus", "")).strip(),
        presuppositions=maybe_dict_list(payload.get("presuppositions", [])),
        requested_action=str(payload.get("requested_action", "")).strip(),
        action_goal=str(payload.get("action_goal", "")).strip(),
        action_object=str(payload.get("action_object", "")).strip(),
        action_options=maybe_str_list(payload.get("action_options", [])),
        hidden_old_setup=str(payload.get("hidden_old_setup", "")).strip(),
        secondary_surface_details=secondary_surface_details,
        query_assumptions=maybe_dict_list(payload.get("query_assumptions", [])),
        option_comparison_requires_viability_check_first=option_viability_first,
    )


def build_premise_verdict(payload: Dict[str, Any]) -> PremiseVerdict:
    return PremiseVerdict(
        status=str(payload.get("status", "")).strip(),
        confidence=maybe_float(payload.get("confidence", 0.0)),
        reasoning_summary=str(payload.get("reasoning_summary", "")).strip(),
        supporting_active_item_ids=maybe_str_list(payload.get("supporting_active_item_ids", [])),
        supporting_active_item_values=maybe_str_list(payload.get("supporting_active_item_values", [])),
        outdated_item_ids=maybe_str_list(payload.get("outdated_item_ids", [])),
        outdated_item_values=maybe_str_list(payload.get("outdated_item_values", [])),
        unknown_tracks=maybe_dict_list(payload.get("unknown_tracks", [])),
        corrected_current_facts=maybe_str_list(payload.get("corrected_current_facts", [])),
        usable_current_basis_item_ids=maybe_str_list(payload.get("usable_current_basis_item_ids", [])),
        usable_current_basis_item_values=maybe_str_list(payload.get("usable_current_basis_item_values", [])),
        blocked_old_basis_item_ids=maybe_str_list(payload.get("blocked_old_basis_item_ids", [])),
        blocked_old_basis_item_values=maybe_str_list(payload.get("blocked_old_basis_item_values", [])),
        historical_but_overridden_item_ids=maybe_str_list(payload.get("historical_but_overridden_item_ids", [])),
        historical_but_overridden_item_values=maybe_str_list(payload.get("historical_but_overridden_item_values", [])),
        old_premise_safe=maybe_bool(payload.get("old_premise_safe", True), True),
    )


def build_constraint_set(payload: Dict[str, Any]) -> ConstraintSet:
    return ConstraintSet(
        hard_positive_constraints=maybe_str_list(payload.get("hard_positive_constraints", [])),
        hard_negative_constraints=maybe_str_list(payload.get("hard_negative_constraints", [])),
        unresolved_variables=maybe_str_list(payload.get("unresolved_variables", [])),
    )


def build_action_readiness_verdict(payload: Dict[str, Any]) -> ActionReadinessVerdict:
    return ActionReadinessVerdict(
        status=str(payload.get("status", "")).strip(),
        confidence=maybe_float(payload.get("confidence", 0.0)),
        reasoning_summary=str(payload.get("reasoning_summary", "")).strip(),
        decisive_gate_bundle_ids=maybe_str_list(payload.get("decisive_gate_bundle_ids", [])),
        blocker_bundle_ids=maybe_str_list(payload.get("blocker_bundle_ids", [])),
        supporting_basis_bundle_ids=maybe_str_list(payload.get("supporting_basis_bundle_ids", [])),
        rejected_nearby_bundle_ids=maybe_str_list(payload.get("rejected_nearby_bundle_ids", [])),
        corrected_current_facts=maybe_str_list(payload.get("corrected_current_facts", [])),
        reframe_mode=str(payload.get("reframe_mode", "")).strip(),
    )


def build_answer_result(payload: Dict[str, Any]) -> AnswerResult:
    answer = str(payload.get("answer", "")).strip()
    rationale = str(payload.get("brief_rationale", "")).strip()
    return AnswerResult(answer=answer, brief_rationale=rationale)
