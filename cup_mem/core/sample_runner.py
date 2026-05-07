from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class SampleRunnerMixin:
    """Dataset sample execution helper."""

    def _build_sample_result(
        self,
        *,
        item: Dict[str, Any],
        sample_index: int,
        session_mode: str,
        selected_session_indices: List[int],
        session_logs: List[Dict[str, Any]],
        query_logs: Dict[str, Any],
        run_status: str,
        current_phase: str,
        sample_elapsed_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        session_elapsed_seconds = [
            float(log.get("elapsed_seconds", 0.0) or 0.0)
            for log in session_logs
            if log.get("elapsed_seconds") is not None
        ]
        query_elapsed_seconds = [
            float(log.get("elapsed_seconds", 0.0) or 0.0)
            for log in query_logs.values()
            if isinstance(log, dict) and log.get("elapsed_seconds") is not None
        ]
        timing_summary: Dict[str, Any] = {
            "session_elapsed_seconds_total": round(sum(session_elapsed_seconds), 3),
            "query_elapsed_seconds_total": round(sum(query_elapsed_seconds), 3),
            "timed_session_count": len(session_elapsed_seconds),
            "timed_query_count": len(query_elapsed_seconds),
        }
        if sample_elapsed_seconds is not None:
            timing_summary["sample_elapsed_seconds"] = round(float(sample_elapsed_seconds), 3)
        return {
            "sample_meta": {
                "uid": item.get("uid"),
                "sample_index": sample_index,
                "M_old": item.get("M_old"),
                "M_new": item.get("M_new"),
                "explanation": item.get("explanation"),
                "time_gap": item.get("time_gap"),
                "relevant_session_index": item.get("relevant_session_index"),
                "selected_session_indices": selected_session_indices,
                "session_mode": session_mode,
            },
            "run_status": run_status,
            "current_phase": current_phase,
            "completed_session_count": len(session_logs),
            "completed_query_count": len(query_logs),
            "session_logs": session_logs,
            "final_profile_snapshot": self.store.to_snapshot(),
            "chunk_bank": [self._chunk_trace_dict(chunk) for chunk in self.chunk_bank],
            "delta_store": [self._delta_trace_dict(delta) for delta in self.delta_store],
            "query_logs": query_logs,
            "timing_summary": timing_summary,
        }

    def run_sample(
        self,
        item: Dict[str, Any],
        *,
        sample_index: int,
        session_mode: str = "full",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        self.reset()
        if hasattr(self.llm, "reset_usage_tracking"):
            self.llm.reset_usage_tracking()
        sessions = item.get("haystack_session", [])
        timestamps = item.get("timestamps", [])
        relevant_session_index = item.get("relevant_session_index", [])

        if session_mode == "relevant_only" and isinstance(relevant_session_index, list) and relevant_session_index:
            indices = [int(idx) for idx in relevant_session_index if 0 <= int(idx) < len(sessions)]
        else:
            indices = list(range(len(sessions)))

        session_logs: List[Dict[str, Any]] = []
        query_logs: Dict[str, Any] = {}
        uid = str(item.get("uid"))
        sample_started_at = time.perf_counter()

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "sample_start",
                    "uid": uid,
                    "sample_index": sample_index,
                    "selected_session_indices": indices,
                    "total_sessions": len(indices),
                    "result": self._build_sample_result(
                        item=item,
                        sample_index=sample_index,
                        session_mode=session_mode,
                        selected_session_indices=indices,
                        session_logs=session_logs,
                        query_logs=query_logs,
                        run_status="running",
                        current_phase="sample_start",
                        sample_elapsed_seconds=time.perf_counter() - sample_started_at,
                    ),
                }
            )

        for idx in indices:
            session = sessions[idx]
            session_time = timestamps[idx] if idx < len(timestamps) else ""
            session_started_at = time.perf_counter()
            session_log = self.process_session(
                session=session,
                session_index=idx,
                session_time=session_time,
            )
            session_elapsed_seconds = time.perf_counter() - session_started_at
            session_log["elapsed_seconds"] = round(session_elapsed_seconds, 3)
            session_log.setdefault("session_summary", {})["elapsed_seconds"] = round(session_elapsed_seconds, 3)
            session_logs.append(session_log)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "session_complete",
                        "uid": uid,
                        "sample_index": sample_index,
                        "session_index": idx,
                        "completed_sessions": len(session_logs),
                        "total_sessions": len(indices),
                        "valid_chunks": len(session_log.get("valid_chunks", [])),
                        "delta_count": len(session_log.get("delta_logs", [])),
                        "invalidation_count": len(session_log.get("invalidation_logs", [])),
                        "chunk_count": session_log.get("session_summary", {}).get("chunk_count"),
                        "dropped_chunks": session_log.get("session_summary", {}).get("dropped_chunk_count"),
                        "raw_delta_count": session_log.get("session_summary", {}).get("raw_delta_count"),
                        "accepted_delta_count": session_log.get("session_summary", {}).get("accepted_delta_count"),
                        "dropped_delta_count": session_log.get("session_summary", {}).get("dropped_delta_count"),
                        "affected_bucket_count": session_log.get("session_summary", {}).get("affected_bucket_count"),
                        "raw_invalidation_proposal_count": session_log.get("session_summary", {}).get("raw_invalidation_proposal_count"),
                        "accepted_invalidation_proposal_count": session_log.get("session_summary", {}).get("accepted_invalidation_proposal_count"),
                        "dropped_invalidation_proposal_count": session_log.get("session_summary", {}).get("dropped_invalidation_proposal_count"),
                        "elapsed_seconds": round(session_elapsed_seconds, 3),
                        "result": self._build_sample_result(
                            item=item,
                            sample_index=sample_index,
                            session_mode=session_mode,
                            selected_session_indices=indices,
                            session_logs=session_logs,
                            query_logs=query_logs,
                            run_status="running",
                            current_phase="session_complete",
                            sample_elapsed_seconds=time.perf_counter() - sample_started_at,
                        ),
                    }
                )

        probing_queries = item.get("probing_queries", {})
        for label, query_text in probing_queries.items():
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "query_start",
                        "uid": uid,
                        "sample_index": sample_index,
                        "query_label": label,
                        "result": self._build_sample_result(
                            item=item,
                            sample_index=sample_index,
                            session_mode=session_mode,
                            selected_session_indices=indices,
                            session_logs=session_logs,
                            query_logs=query_logs,
                            run_status="running",
                            current_phase=f"query_start:{label}",
                            sample_elapsed_seconds=time.perf_counter() - sample_started_at,
                        ),
                    }
                )
            query_started_at = time.perf_counter()
            query_log = self.answer_query(query_label=label, query_text=str(query_text))
            query_elapsed_seconds = time.perf_counter() - query_started_at
            query_log["elapsed_seconds"] = round(query_elapsed_seconds, 3)
            query_logs[label] = query_log
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "query_complete",
                        "uid": uid,
                        "sample_index": sample_index,
                        "query_label": label,
                        "verdict_status": query_log.get("verdict", {}).get("status", ""),
                        "verdict_confidence": query_log.get("verdict", {}).get("confidence"),
                        "elapsed_seconds": round(query_elapsed_seconds, 3),
                        "result": self._build_sample_result(
                            item=item,
                            sample_index=sample_index,
                            session_mode=session_mode,
                            selected_session_indices=indices,
                            session_logs=session_logs,
                            query_logs=query_logs,
                            run_status="running",
                            current_phase=f"query_complete:{label}",
                            sample_elapsed_seconds=time.perf_counter() - sample_started_at,
                        ),
                    }
                )

        sample_elapsed_seconds = time.perf_counter() - sample_started_at
        final_result = self._build_sample_result(
            item=item,
            sample_index=sample_index,
            session_mode=session_mode,
            selected_session_indices=indices,
            session_logs=session_logs,
            query_logs=query_logs,
            run_status="completed",
            current_phase="completed",
            sample_elapsed_seconds=sample_elapsed_seconds,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "sample_complete",
                    "uid": uid,
                    "sample_index": sample_index,
                    "elapsed_seconds": round(sample_elapsed_seconds, 3),
                    "result": final_result,
                }
            )
        return final_result
