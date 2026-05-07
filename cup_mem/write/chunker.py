from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..memory.models import SessionChunk, bucket_track_refs_from_payload, maybe_bool, maybe_dict_list, maybe_float, maybe_str_list
from ..prompt_lib.templates import SESSION_CHUNKER_PROMPT
from ..memory.schema import is_valid_bucket_track


class ChunkerMixin:

    def _evaluate_chunk_validity(self, chunk: SessionChunk) -> Dict[str, Any]:
        drop_reasons: List[str] = []
        allowed_chunk = (
            chunk.chunk_type in {"explicit_state", "support_evidence"}
        )
        temporal_scope_rescued = (
            chunk.chunk_type in {"explicit_state", "support_evidence"}
            and chunk.temporal_scope in {"past", "future"}
            and chunk.change_signal_present
            and chunk.signal_strength in {"strong", "medium"}
            and chunk.state_salience >= self.thresholds.state_salience_threshold
        )
        if not allowed_chunk:
            if chunk.chunk_type == "task_noise" and not chunk.change_signal_present:
                drop_reasons.append("task_noise_without_change_signal")
            else:
                drop_reasons.append(f"chunk_type_filtered:{chunk.chunk_type}")
        if chunk.temporal_scope not in {"current", "mixed"} and not temporal_scope_rescued:
            drop_reasons.append(f"temporal_scope_filtered:{chunk.temporal_scope}")
        if chunk.state_salience < self.thresholds.state_salience_threshold:
            drop_reasons.append(
                f"state_salience_below_threshold:{chunk.state_salience:.2f}<{self.thresholds.state_salience_threshold:.2f}"
            )
        passed = not drop_reasons
        return {
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "temporal_scope": chunk.temporal_scope,
            "state_salience": chunk.state_salience,
            "state_salience_threshold": self.thresholds.state_salience_threshold,
            "evidence_strength": chunk.evidence_strength,
            "change_signal_present": chunk.change_signal_present,
            "change_signal_summary": chunk.change_signal_summary,
            "signal_strength": chunk.signal_strength,
            "temporal_scope_rescued": temporal_scope_rescued,
            "candidate_bucket_tracks": [ref.to_dict() for ref in chunk.candidate_bucket_tracks],
            "source_turn_ids": chunk.source_turn_ids,
            "text_preview": self._preview_text(chunk.text),
            "passed_valid_chunk_filter": passed,
            "drop_reasons": drop_reasons,
        }

    def _llm_call_json(
        self,
        *,
        phase: str,
        system_prompt: str,
        user_payload: str,
        session_id: str = "",
        query_label: str = "",
        extra_meta: Optional[Dict[str, Any]] = None,
        extra_request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        call_meta: Dict[str, Any] = {
            "phase": phase,
        }
        if session_id:
            call_meta["session_id"] = session_id
        if query_label:
            call_meta["query_label"] = query_label
        if extra_meta:
            call_meta.update(extra_meta)
        return self.llm.call_json(
            system_prompt,
            user_payload,
            extra_meta=call_meta,
            extra_request_kwargs=extra_request_kwargs,
        )

    @staticmethod
    def session_to_user_turns(session: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        turns: List[Dict[str, str]] = []
        for idx, message in enumerate(session):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).strip() != "user":
                continue
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            turns.append(
                {
                    "turn_id": f"t_{idx}",
                    "content": content,
                }
            )
        return turns

    def _build_chunker_payload(self, *, session_id: str, user_turns: List[Dict[str, str]]) -> str:
        return json.dumps(
            {
                "session_id": session_id,
                "bucket_track_schema": self.schema_text,
                "user_turns": user_turns,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _sanitize_chunk(self, raw: Dict[str, Any], session_id: str, valid_turn_ids: set[str]) -> Optional[SessionChunk]:
        chunk_type = str(raw.get("chunk_type", "")).strip()
        temporal_scope = str(raw.get("temporal_scope", "")).strip()
        evidence_strength = str(raw.get("evidence_strength", "")).strip()
        if chunk_type not in {"explicit_state", "support_evidence", "task_noise", "past_future"}:
            return None
        if temporal_scope not in {"current", "past", "future", "mixed"}:
            temporal_scope = "past"
        if evidence_strength not in {"strong", "medium", "weak"}:
            evidence_strength = "weak"
        change_signal_present = maybe_bool(raw.get("change_signal_present", False), False)
        change_signal_summary = str(raw.get("change_signal_summary", "")).strip()
        signal_strength = str(raw.get("signal_strength", "weak")).strip().lower() or "weak"
        if signal_strength not in {"strong", "medium", "weak"}:
            signal_strength = "weak"
        if change_signal_summary and not change_signal_present:
            change_signal_present = True
        if not change_signal_present:
            change_signal_summary = ""
            signal_strength = "weak"
        elif not change_signal_summary:
            change_signal_summary = "current-state-change signal present"

        source_turn_ids = [turn_id for turn_id in maybe_str_list(raw.get("source_turn_ids", [])) if turn_id in valid_turn_ids]
        text = str(raw.get("text", "")).strip()
        if not source_turn_ids or not text:
            return None

        refs = []
        for ref in bucket_track_refs_from_payload(raw.get("candidate_bucket_tracks", []))[:3]:
            if is_valid_bucket_track(ref.bucket, ref.local_track):
                refs.append(ref)

        return SessionChunk(
            chunk_id=self._next_chunk_id(),
            session_id=session_id,
            source_turn_ids=source_turn_ids,
            text=text,
            chunk_type=chunk_type,
            candidate_bucket_tracks=refs,
            temporal_scope=temporal_scope,
            state_salience=max(0.0, min(1.0, maybe_float(raw.get("state_salience", 0.0)))),
            evidence_strength=evidence_strength,
            change_signal_present=change_signal_present,
            change_signal_summary=change_signal_summary,
            signal_strength=signal_strength,
        )

    def chunk_session(self, *, session_id: str, user_turns: List[Dict[str, str]]) -> Dict[str, Any]:
        if not user_turns:
            return {"raw_payload": {}, "chunks": []}
        payload = self._llm_call_json(
            phase="session_chunker",
            system_prompt=SESSION_CHUNKER_PROMPT,
            user_payload=self._build_chunker_payload(session_id=session_id, user_turns=user_turns),
            session_id=session_id,
        )
        valid_turn_ids = {turn["turn_id"] for turn in user_turns}
        chunks: List[SessionChunk] = []
        for raw_chunk in maybe_dict_list(payload.get("chunks", [])):
            chunk = self._sanitize_chunk(raw_chunk, session_id, valid_turn_ids)
            if chunk is not None:
                chunks.append(chunk)
        self.chunk_bank.extend(chunks)
        return {
            "raw_payload": payload,
            "chunks": chunks,
        }
