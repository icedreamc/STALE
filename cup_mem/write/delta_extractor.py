from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..core.temporal import session_index_from_id
from ..memory.models import SessionChunk, SessionDelta, bucket_track_refs_from_payload, maybe_bool, maybe_dict_list, maybe_float, maybe_str_list
from ..prompt_lib.templates import DELTA_EXTRACTOR_PROMPT
from ..memory.schema import is_valid_bucket_track, parent_bucket_for_track
from ..store_layer import ProfileStore


class DeltaExtractorMixin:
    def _build_delta_payload(
        self,
        *,
        session_id: str,
        valid_chunks: List[SessionChunk],
        profile_subset: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        return json.dumps(
            {
                "session_id": session_id,
                "bucket_track_schema": self.schema_text,
                "valid_chunks": [chunk.to_dict() for chunk in valid_chunks],
                "relevant_profile_subset": profile_subset,
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _session_index_from_session_id(session_id: str) -> int:
        return session_index_from_id(session_id)

    def _sanitize_delta(
        self,
        raw: Dict[str, Any],
        session_id: str,
        chunk_map: Dict[str, SessionChunk],
        *,
        session_time: str = "",
    ) -> Optional[SessionDelta]:
        bucket = str(raw.get("bucket", "")).strip()
        local_track = str(raw.get("local_track", "")).strip()
        if not is_valid_bucket_track(bucket, local_track):
            swapped_bucket = str(raw.get("local_track", "")).strip()
            swapped_track = str(raw.get("bucket", "")).strip()
            if is_valid_bucket_track(swapped_bucket, swapped_track):
                bucket, local_track = swapped_bucket, swapped_track
        if not is_valid_bucket_track(bucket, local_track):
            bucket_parent = parent_bucket_for_track(bucket)
            track_parent = parent_bucket_for_track(local_track)
            if bucket_parent and bucket_parent == track_parent:
                bucket, local_track = bucket_parent, bucket
        if not is_valid_bucket_track(bucket, local_track):
            return None
        proposed_value = str(raw.get("proposed_value", "")).strip()
        source_type = str(raw.get("source_type", "")).strip() or "inferred"
        if source_type not in {"explicit", "inferred"}:
            source_type = "inferred"
        evidence_ids = self._resolve_reference_ids(
            maybe_str_list(raw.get("evidence_chunk_ids", [])),
            chunk_map.keys(),
            prefix="c",
        )
        if not proposed_value or not evidence_ids:
            return None
        op_hint = str(raw.get("op_hint", "")).strip().upper()
        if op_hint not in {"ADD", "REFINE", "REPLACE", "NO_OP"}:
            op_hint = ""
        delta_kind = str(raw.get("delta_kind", "observed_state")).strip() or "observed_state"
        if delta_kind not in {"observed_state", "upstream_inferred_state"}:
            delta_kind = "observed_state"
        return SessionDelta(
            delta_id=self._next_delta_id(),
            session_id=session_id,
            session_index=self._session_index_from_session_id(session_id),
            session_time=session_time,
            bucket=bucket,
            local_track=local_track,
            proposed_value=proposed_value,
            source_type=source_type,
            evidence_chunk_ids=evidence_ids,
            confidence=max(0.0, min(1.0, maybe_float(raw.get("confidence", 0.0)))),
            op_hint=op_hint,
            delta_kind=delta_kind,
        )

    def _sanitize_delta_with_log(
        self,
        raw: Dict[str, Any],
        *,
        raw_index: int,
        session_id: str,
        chunk_map: Dict[str, SessionChunk],
        session_time: str = "",
    ) -> Tuple[Optional[SessionDelta], Dict[str, Any]]:
        bucket = str(raw.get("bucket", "")).strip()
        local_track = str(raw.get("local_track", "")).strip()
        warnings: List[str] = []
        if not is_valid_bucket_track(bucket, local_track):
            swapped_bucket = str(raw.get("local_track", "")).strip()
            swapped_track = str(raw.get("bucket", "")).strip()
            if is_valid_bucket_track(swapped_bucket, swapped_track):
                bucket, local_track = swapped_bucket, swapped_track
                warnings.append("bucket_track_swapped_from_raw_fields")
        if not is_valid_bucket_track(bucket, local_track):
            bucket_parent = parent_bucket_for_track(bucket)
            track_parent = parent_bucket_for_track(local_track)
            if bucket_parent and bucket_parent == track_parent:
                bucket, local_track = bucket_parent, bucket
                warnings.append("bucket_track_recovered_from_two_tracks_same_parent_bucket")
        proposed_value = str(raw.get("proposed_value", "")).strip()
        source_type = str(raw.get("source_type", "")).strip() or "inferred"
        delta_kind = str(raw.get("delta_kind", "observed_state")).strip() or "observed_state"
        confidence = max(0.0, min(1.0, maybe_float(raw.get("confidence", 0.0))))
        evidence_ids = self._resolve_reference_ids(
            maybe_str_list(raw.get("evidence_chunk_ids", [])),
            chunk_map.keys(),
            prefix="c",
        )
        drop_reasons: List[str] = []

        if not is_valid_bucket_track(bucket, local_track):
            drop_reasons.append("invalid_bucket_track")
        if not proposed_value:
            drop_reasons.append("empty_proposed_value")
        if not evidence_ids:
            drop_reasons.append("no_valid_evidence_chunk_ids")

        log_record: Dict[str, Any] = {
            "raw_delta_index": raw_index,
            "bucket": bucket,
            "local_track": local_track,
            "raw_bucket": str(raw.get("bucket", "")).strip(),
            "raw_local_track": str(raw.get("local_track", "")).strip(),
            "proposed_value": proposed_value,
            "source_type": source_type,
            "delta_kind": delta_kind,
            "confidence": confidence,
            "evidence_chunk_ids": evidence_ids,
            "evidence_chunk_summaries": [
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "evidence_strength": chunk.evidence_strength,
                    "change_signal_present": chunk.change_signal_present,
                    "state_salience": chunk.state_salience,
                }
                for chunk_id, chunk in chunk_map.items()
                if chunk_id in evidence_ids
            ],
            "accepted": False,
            "drop_reason": drop_reasons[0] if drop_reasons else "",
            "drop_reasons": drop_reasons,
            "warnings": warnings,
            "rule_evaluation": None,
        }

        if drop_reasons:
            return None, log_record

        delta = self._sanitize_delta(raw, session_id, chunk_map, session_time=session_time)
        if delta is None:
            log_record["drop_reason"] = "sanitize_delta_failed"
            log_record["drop_reasons"] = ["sanitize_delta_failed"]
            return None, log_record
        return delta, log_record

    def _evaluate_delta_rules(self, delta: SessionDelta, chunk_map: Dict[str, SessionChunk]) -> Dict[str, Any]:
        evidence_chunks = [chunk_map[chunk_id] for chunk_id in delta.evidence_chunk_ids if chunk_id in chunk_map]
        if not evidence_chunks:
            return {
                "accepted": False,
                "accept_rule": "",
                "threshold_used": None,
                "drop_reason": "no_evidence_chunks",
                "drop_reasons": ["no_evidence_chunks"],
                "explicit_support": False,
                "strong_support": False,
                "medium_support_count": 0,
                "weak_only": False,
                "has_change_signal": False,
                "evidence_chunks": [],
            }

        explicit_support = any(chunk.chunk_type == "explicit_state" for chunk in evidence_chunks)
        strong_support = any(chunk.evidence_strength == "strong" for chunk in evidence_chunks)
        medium_chunks = [chunk for chunk in evidence_chunks if chunk.evidence_strength == "medium"]
        weak_only = all(chunk.evidence_strength == "weak" for chunk in evidence_chunks)
        has_change_signal = any(chunk.change_signal_present for chunk in evidence_chunks)
        evaluation: Dict[str, Any] = {
            "accepted": False,
            "accept_rule": "",
            "threshold_used": None,
            "drop_reason": "",
            "drop_reasons": [],
            "explicit_support": explicit_support,
            "strong_support": strong_support,
            "medium_support_count": len(medium_chunks),
            "weak_only": weak_only,
            "has_change_signal": has_change_signal,
            "evidence_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "evidence_strength": chunk.evidence_strength,
                    "change_signal_present": chunk.change_signal_present,
                    "signal_strength": chunk.signal_strength,
                    "state_salience": chunk.state_salience,
                }
                for chunk in evidence_chunks
            ],
        }

        if delta.source_type == "explicit" or explicit_support:
            threshold = self.thresholds.explicit_min_confidence
            evaluation["threshold_used"] = threshold
            if delta.confidence >= threshold:
                evaluation["accepted"] = True
                evaluation["accept_rule"] = "explicit_or_explicit_chunk_support"
            else:
                evaluation["drop_reasons"] = [f"explicit_confidence_below_threshold:{delta.confidence:.2f}<{threshold:.2f}"]
        elif strong_support:
            threshold = self.thresholds.strong_support_min_confidence
            evaluation["threshold_used"] = threshold
            if delta.confidence >= threshold:
                evaluation["accepted"] = True
                evaluation["accept_rule"] = "strong_support"
            else:
                evaluation["drop_reasons"] = [f"strong_support_confidence_below_threshold:{delta.confidence:.2f}<{threshold:.2f}"]
        elif delta.delta_kind == "upstream_inferred_state" and has_change_signal:
            threshold = self.thresholds.strong_support_min_confidence
            evaluation["threshold_used"] = threshold
            if delta.confidence >= threshold:
                evaluation["accepted"] = True
                evaluation["accept_rule"] = "upstream_inferred_with_change_signal"
            else:
                evaluation["drop_reasons"] = [
                    f"upstream_change_signal_confidence_below_threshold:{delta.confidence:.2f}<{threshold:.2f}"
                ]
        elif weak_only:
            evaluation["drop_reasons"] = ["weak_only_evidence"]
        elif len(medium_chunks) >= 2:
            threshold = self.thresholds.medium_support_min_confidence
            evaluation["threshold_used"] = threshold
            if delta.confidence >= threshold:
                evaluation["accepted"] = True
                evaluation["accept_rule"] = "multiple_medium_support"
            else:
                evaluation["drop_reasons"] = [f"medium_support_confidence_below_threshold:{delta.confidence:.2f}<{threshold:.2f}"]
        else:
            evaluation["drop_reasons"] = ["insufficient_medium_support"]

        if not evaluation["accepted"] and not evaluation["drop_reasons"]:
            evaluation["drop_reasons"] = ["no_accept_rule_matched"]
        evaluation["drop_reason"] = evaluation["drop_reasons"][0] if evaluation["drop_reasons"] else ""
        return evaluation

    def extract_deltas(
        self,
        *,
        session_id: str,
        valid_chunks: List[SessionChunk],
        store_view: Optional[ProfileStore] = None,
        session_time: str = "",
    ) -> Dict[str, Any]:
        if not valid_chunks:
            empty_subset = {"active_items": [], "stale_items": [], "unknown_items": []}
            return {
                "raw_payload": {},
                "deltas": [],
                "profile_subset": empty_subset,
                "profile_subset_summary": self._summarize_profile_subset(empty_subset, requested_pairs=[]),
                "delta_filter_logs": [],
            }

        bucket_tracks = []
        for chunk in valid_chunks:
            for ref in chunk.candidate_bucket_tracks:
                bucket_tracks.append((ref.bucket, ref.local_track))
        store_ref = store_view or self.store
        profile_subset = store_ref.get_profile_subset(bucket_tracks)
        payload = self._llm_call_json(
            phase="delta_extractor",
            system_prompt=DELTA_EXTRACTOR_PROMPT,
            user_payload=self._build_delta_payload(session_id=session_id, valid_chunks=valid_chunks, profile_subset=profile_subset),
            session_id=session_id,
        )

        chunk_map = {chunk.chunk_id: chunk for chunk in valid_chunks}
        deltas: List[SessionDelta] = []
        delta_filter_logs: List[Dict[str, Any]] = []
        for raw_index, raw_delta in enumerate(maybe_dict_list(payload.get("deltas", [])), start=1):
            delta, filter_log = self._sanitize_delta_with_log(
                raw_delta,
                raw_index=raw_index,
                session_id=session_id,
                chunk_map=chunk_map,
                session_time=session_time,
            )
            if delta is None:
                delta_filter_logs.append(filter_log)
                continue
            rule_evaluation = self._evaluate_delta_rules(delta, chunk_map)
            filter_log["rule_evaluation"] = rule_evaluation
            filter_log["accepted"] = rule_evaluation["accepted"]
            filter_log["drop_reason"] = rule_evaluation["drop_reason"]
            filter_log["drop_reasons"] = rule_evaluation["drop_reasons"]
            filter_log["delta_id"] = delta.delta_id
            if rule_evaluation["accepted"]:
                deltas.append(delta)
            delta_filter_logs.append(filter_log)
        return {
            "raw_payload": payload,
            "profile_subset": profile_subset,
            "profile_subset_summary": self._summarize_profile_subset(profile_subset, requested_pairs=bucket_tracks),
            "delta_filter_logs": delta_filter_logs,
            "deltas": deltas,
        }

