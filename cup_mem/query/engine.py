from __future__ import annotations

from typing import Any, Dict, List

from .support import QuerySupportMixin


class QueryEngineMixin(QuerySupportMixin):
    def answer_query(self, *, query_label: str, query_text: str) -> Dict[str, Any]:
        parsed = self._parse_query(query_text, query_label=query_label)
        relevant_context = self._collect_relevant_state_context(parsed, query_text=query_text)
        relevant_context_summary = self._summarize_query_readout_context(relevant_context)
        relevant_context_summary["query_topk_summary"] = relevant_context.get("query_topk_summary", {})

        initial_verify_result = self._verify_premise(
            parsed=parsed,
            query_text=query_text,
            relevant_context=relevant_context,
            query_label=query_label,
        )
        initial_verdict = initial_verify_result["verdict"]
        verdict = initial_verdict
        basis_recovery_hint = self._build_basis_recovery_hint(
            parsed=parsed,
            query_text=query_text,
            verdict=verdict,
            relevant_context=relevant_context,
        )
        action_decision_mode = self._action_decision_mode_info(
            parsed=parsed,
            query_text=query_text,
            verdict=verdict,
            basis_recovery_hint=basis_recovery_hint,
        )
        self._set_action_decision_mode(parsed, bool(action_decision_mode.get("enabled")))
        relevant_context["action_decision_mode"] = action_decision_mode
        basis_recovery_result = self._run_basis_recovery(
            parsed=parsed,
            query_text=query_text,
            verdict=verdict,
            basis_recovery_hint=basis_recovery_hint,
            relevant_context=relevant_context,
        )
        verdict = self._merge_basis_recovery_into_verdict(
            verdict=verdict,
            basis_recovery_result=basis_recovery_result,
        )
        verdict = self._refresh_verdict_basis_views(
            verdict=verdict,
            parsed=parsed,
            relevant_context=relevant_context,
            basis_recovery_result=basis_recovery_result,
        )
        relevant_context["basis_recovery_hint"] = basis_recovery_hint
        relevant_context["basis_recovery_result"] = basis_recovery_result
        fallback_evidence: List[Dict[str, Any]] = []
        fallback_trigger_reasons: List[str] = self._should_run_second_pass(
            verdict=verdict,
            relevant_context=relevant_context,
        )
        second_pass_mode = ""
        verify_with_evidence_raw = None
        final_verifier_raw_payload = initial_verify_result["raw_payload"]
        if fallback_trigger_reasons:
            conflict_centered_reasons = {
                "core_conflict_strong_vs_stale_or_unknown",
                "premise_conflict_line_present",
                "unsafe_premise_with_conflict_line",
            }
            if any(reason in conflict_centered_reasons for reason in fallback_trigger_reasons):
                second_pass_mode = "premise_conflict_centered"
                fallback_evidence = self._retrieve_query_conflict_evidence(
                    parsed=parsed,
                    query_text=query_text,
                    relevant_context=relevant_context,
                )
            else:
                second_pass_mode = "general_fallback"
                fallback_evidence = self._fallback_retrieve(query_text)
            verify_with_evidence_result = self._verify_premise(
                parsed=parsed,
                query_text=query_text,
                relevant_context=relevant_context,
                extra_evidence=fallback_evidence,
                second_pass_mode=second_pass_mode,
                query_label=query_label,
            )
            verdict = verify_with_evidence_result["verdict"]
            basis_recovery_hint = self._build_basis_recovery_hint(
                parsed=parsed,
                query_text=query_text,
                verdict=verdict,
                relevant_context=relevant_context,
            )
            action_decision_mode = self._action_decision_mode_info(
                parsed=parsed,
                query_text=query_text,
                verdict=verdict,
                basis_recovery_hint=basis_recovery_hint,
            )
            self._set_action_decision_mode(parsed, bool(action_decision_mode.get("enabled")))
            relevant_context["action_decision_mode"] = action_decision_mode
            basis_recovery_result = self._run_basis_recovery(
                parsed=parsed,
                query_text=query_text,
                verdict=verdict,
                basis_recovery_hint=basis_recovery_hint,
                relevant_context=relevant_context,
            )
            verdict = self._merge_basis_recovery_into_verdict(
                verdict=verdict,
                basis_recovery_result=basis_recovery_result,
            )
            verdict = self._refresh_verdict_basis_views(
                verdict=verdict,
                parsed=parsed,
                relevant_context=relevant_context,
                basis_recovery_result=basis_recovery_result,
            )
            relevant_context["basis_recovery_hint"] = basis_recovery_hint
            relevant_context["basis_recovery_result"] = basis_recovery_result
            verify_with_evidence_raw = verify_with_evidence_result["raw_payload"]
            final_verifier_raw_payload = verify_with_evidence_result["raw_payload"]

        action_result = None
        action_readiness = None
        should_resolve_action = bool(action_decision_mode.get("enabled"))
        if should_resolve_action:
            action_result = self._judge_action_readiness(
                parsed=parsed,
                query_text=query_text,
                verdict=verdict,
                relevant_context=relevant_context,
                basis_recovery_result=basis_recovery_result,
                query_label=query_label,
            )
            action_readiness = action_result["action_readiness"]
            verdict = self._apply_action_readiness(
                readiness=action_readiness,
                action_result=action_result,
                verdict=verdict,
                relevant_context=relevant_context,
            )

        constraints_result = None
        constraints = None
        if parsed.intent_type == "TASK_PLANNING":
            constraints_result = self._build_constraints(
                query_text=query_text,
                parsed=parsed,
                verdict=verdict,
                relevant_context=relevant_context,
                action_readiness=action_readiness,
                query_label=query_label,
            )
            constraints = constraints_result["constraints"]

        answer_result = self._compose_answer(
            query_text=query_text,
            parsed=parsed,
            verdict=verdict,
            relevant_context=relevant_context,
            constraints=constraints,
            action_readiness=action_readiness,
            query_label=query_label,
        )
        answer_input_summary = self._build_answer_input_summary(
            parsed=parsed,
            verdict=verdict,
            relevant_context=relevant_context,
            constraints=constraints,
            basis_recovery_hint=basis_recovery_hint,
            basis_recovery_result=basis_recovery_result,
        )
        result_relevant_context = self._move_aux_fields_to_debug_trace(
            dict(relevant_context),
            section="query",
            field_names=[
                "query_topk_summary",
                "premise_conflict_subset_summary",
                "premise_conflict_subset_seed_tracks",
                "premise_conflict_subset_seed_track_logs",
            ],
        )
        result = {
            "query_label": query_label,
            "query_text": query_text,
            "parsed_query": parsed.to_dict(),
            "relevant_context": result_relevant_context,
            "verdict": verdict.to_dict(),
            "basis_recovery_hint": basis_recovery_hint,
            "basis_recovery_result": basis_recovery_result,
            "action_readiness": action_readiness.to_dict() if action_readiness is not None else None,
            "constraints": constraints.to_dict() if constraints is not None else None,
            "answer": answer_result["answer"].to_dict(),
        }
        return self._attach_debug_trace(
            result,
            section="query",
            fields={
                "initial_verdict": initial_verdict.to_dict(),
                "relevant_context_summary": relevant_context_summary,
                "answer_input_summary": answer_input_summary,
                "premise_verifier_initial_raw_payload": initial_verify_result["raw_payload"],
                "premise_verifier_raw_payload": final_verifier_raw_payload,
                "premise_verifier_with_evidence_raw_payload": verify_with_evidence_raw,
                "premise_verifier_final_raw_payload": final_verifier_raw_payload,
                "action_grounding_raw_payload": action_result["raw_payload"] if action_result else None,
                "constraint_builder_raw_payload": constraints_result["raw_payload"] if constraints_result else None,
                "answer_composer_raw_payload": answer_result["raw_payload"],
            },
        )
