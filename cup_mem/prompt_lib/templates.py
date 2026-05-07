SESSION_CHUNKER_PROMPT = """Pick the smallest current-state changes from one session's USER turns.

This step preserves only compact state-bearing units and supporting current-state evidence.
It does not summarize the profile, predict invalidation, or decide future updates.
Use only user text. Do not use assistant text as evidence.
Do not split by fixed length. Keep boundaries that preserve one current proposition or one current-state clue intact.
Return at most 2 chunks for the session. Prefer the 1-2 clauses that most clearly state a changed or currently decision-relevant user state.

Output JSON only:
{
  "chunks": [
    {
      "source_turn_ids": ["turn ids"],
      "text": "merged chunk text",
      "chunk_type": "explicit_state|support_evidence|task_noise|past_future",
      "candidate_bucket_tracks": [{"bucket": "...", "local_track": "..."}],
      "temporal_scope": "current|past|future|mixed",
      "state_salience": 0.0,
      "evidence_strength": "strong|medium|weak",
      "change_signal_present": true,
      "change_signal_summary": "short abstract description",
      "signal_strength": "strong|medium|weak"
    }
  ]
}

Chunk types:
- explicit_state: directly states a current fact, role, condition, arrangement, obligation, capability, preference, affiliation, access condition, or recurring constraint.
- support_evidence: does not state the final current fact directly, but gives grounded evidence for a current-state shift or for a currently relevant condition.
- task_noise: request shell, checklist, template, or logistics with no current-state meaning.
- past_future: only past/future, with no grounded current continuation.

Rules:
- Preserve the smallest unit that still keeps one current proposition or one supporting current-state clue intact.
- Do not output pure request shells, checklist requests, template requests, or broad framing. If such text contains a current-state clause, output only that state-bearing clause.
- If a request, plan, explanation, or story contains a current-state clause, keep the state-bearing clause and drop the surrounding request shell.
- If nearby text contains both broad setup/framing and a more concrete clause that carries the present-state change, prefer the concrete clause as the chunk. Keep the broader framing only when it independently states a separate current fact.
- Do not let a generic "thinking about", "trying to figure out", or other meta-framing statement become the main chunk when a more premise-bearing current relation is stated in the same nearby text.
- Preserve clues about what is currently true, currently available, currently constrained, still retained, already lost, can access, cannot access, currently affiliated with, or currently arranged in a recurring way.
- When the current meaning depends on who currently does what, who currently has what, or what is currently available to whom, keep that relation-bearing wording rather than flattening it into a broad shared summary.
- If a completed action or transition creates an ongoing current condition, keep the resulting current condition as a state-bearing unit rather than treating the text as only past.
- Treat resets, rebuilds, restores, migrations, enrollments, role transitions, commute changes, recurring schedule changes, and institutional changes as potentially current when their effects still hold now.
- When a sentence implies changed continuity, recoverability, access, affiliation, feasibility, or recurring arrangement in the present, prefer `explicit_state` or `support_evidence` over `past_future`.
- `candidate_bucket_tracks` is best-effort guidance only. Do not over-optimize placement here.
- `task_noise` means no current-state meaning, not just request-like wording.
- When unsure between `task_noise` and `support_evidence`, choose `support_evidence` if the text contains a current-state change, limitation, access condition, affiliation, obligation, location/setup, or recurring arrangement.
- Use `past_future` only when there is no grounded current continuation or current consequence.
"""


DELTA_EXTRACTOR_PROMPT = """You turn valid chunks into faithful memory-ready CURRENT-state deltas.

Goal:
1. extract the smallest grounded current proposition
2. place it in a valid bucket/track without changing its meaning

Input:
- valid user-derived chunks
- a relevant local profile subset

Output JSON only:
{
  "deltas": [
    {
      "bucket": "...",
      "local_track": "...",
      "proposed_value": "...",
      "source_type": "explicit|inferred",
      "evidence_chunk_ids": ["chunk ids"],
      "confidence": 0.0,
      "op_hint": "ADD|REFINE|REPLACE|NO_OP",
      "delta_kind": "observed_state|upstream_inferred_state"
    }
  ]
}

Rules:
- Extract the smallest standalone current proposition the chunk supports.
- Prefer current basis facts that can later govern an action or answer: eligibility, capability, limitation, arrangement, dependency, location/commute basis, access condition, household setup, recurring constraint, affiliation, or availability.
- Preserve the concrete object, relation, and important qualifier. Do not flatten a concrete state into a broad topic.
- If the chunk contains both surface detail and a deeper current-state basis, prefer the deeper basis unless the surface detail itself is the hard constraint.
- If the chunk describes an event or transition, write the current consequence that still holds now, not just the event label.
- `proposed_value` must be a current proposition, not a topic summary, planning shell, personality label, or generic concern.
- Placement must not rewrite the proposition. If unsure, keep the proposition concrete and choose the most conservative valid bucket/track.
- `bucket` must be a bucket name from the schema. `local_track` must be a track inside that bucket. Never output two track names.
- `explicit_state` may produce a direct delta. `support_evidence` may produce an inferred delta only when it strongly supports a concrete current state.
- Emit no delta only when no standalone current proposition can be grounded.
"""


BUCKET_BRIDGE_PROPOSER_PROMPT = """Given valid chunks and extracted deltas, propose additional buckets whose old CURRENT defaults may now be unsafe because a practical basis changed.

This is a scope-expansion step, not a final invalidation step.

Output JSON only:
{
  "affected_buckets": [
    {
      "bucket": "...",
      "reason": "what practical basis changed and why old defaults in this bucket may now be unsafe",
      "confidence": 0.0
    }
  ]
}

Rules:
- Start from the changed proposition expressed by the new delta, not from the general conversation topic.
- Propose a bucket only when you can name a practical basis that changed and explain how an old default in that bucket could now be less safe.
- Practical basis includes access, availability, continuity, recoverability, feasibility, responsibility, arrangement, recurring coordination, and institutional/status dependency.
- Ask whether the new condition changes what an old default depends on: whether it is still available, still feasible, still recoverable, still institutionally compatible, or still safe as a current default.
- Prefer buckets that are directly downstream of the changed basis, not buckets that are merely semantically nearby.
- Shared life domain, topic overlap, or general schedule pressure is not enough unless it clearly changes the basis of an old default.
- This step expands candidate scope only. It does not identify the exact old item and does not make a final invalidation judgment.
- Use only the provided schema.
- If no plausible downstream dependency exists, output an empty list.
"""


LATENT_INVALIDATION_PROPOSER_PROMPT = """Work within one provided invalidation candidate scope at a time.

Given session deltas and a scope-bounded ACTIVE old-item subset, propose which old CURRENT defaults are most likely being challenged.
This stage nominates candidates. It does not decide the final outcome.

Output JSON only:
{
  "proposals": [
    {
      "target_bucket": "...",
      "target_local_track": "...",
      "target_item_id_hint": "optional existing item id",
      "target_old_value": "old value text if applicable",
      "trigger_delta_ids": ["delta ids"],
      "evidence_chunk_ids": ["chunk ids"],
      "proposal_type": "INVALIDATE_OR_REASSESS",
      "reason": "why this old current item may no longer be safe",
      "confidence": 0.0
    }
  ]
}

Rules:
- Read the provided scope label first and stay inside the provided candidate scope.
- Bucket/track expansion is already fixed by the provided inputs; do not invent new buckets.
- Prefer targets from the provided active old-item subset.
- The subset is a candidate panel, not a hint to pick the nearest item by default.
- `target_item_id_hint` and `target_old_value` are important whenever possible.
- Choose the old item whose exact proposition or relied-on setup is made unsafe, not the item that is merely topically nearby.
- Prefer the more premise-bearing old item over a broad umbrella/background item.
- Write the reason from the old item's perspective: what basis of that old item changed?
- If the delta looks like a replacement or successor, try to name the exact old setup in `target_old_value`.
- Propose item-level challenges only. Topic drift or generic workload/focus drift is not enough.
- `direct_bucket`: prioritize old defaults directly challenged inside this bucket.
- `affected_bucket`: prioritize old defaults whose practical basis is downstream from the new delta.
- `global_recall`: use the provided globally retrieved old-item subset only as a bounded rescue lane when the true old premise may sit outside the direct/affected buckets; still stay inside the provided subset.
- Return up to 3 proposals when multiple old defaults are plausibly at risk.
- If no plausible old item/default is at risk in this provided scope, output no proposal.
"""


UPDATE_JUDGE_PROMPT = """Given one new delta and a small same-track target pool, decide how the profile should update while preserving proposition fidelity.

Output JSON only:
{
  "decision": "ADD|REFINE|REPLACE|INDIRECT_INVALIDATE|NO_OP",
  "target_item_id": "optional existing item id, otherwise empty string",
  "reason": "brief grounded reason"
}

Rules:
- Only items in `same_track_target_candidates` may be selected as `target_item_id`.
- `same_bucket_context_items` are context only; never target them.
- Judge sameness by proposition, practical basis, and current item identity, not by loose similarity.
- Do not treat two items as the same proposition just because they share a track label. If the core object, relation, or relied-on implication changes, prefer ADD over REFINE/REPLACE.
- REFINE: same proposition / same item, just more specific or updated.
- REPLACE: same current-default slot, but old item is outdated and the new delta should take over.
- ADD: new coexisting current item, or no same-track item is truly the same proposition/item.
- INDIRECT_INVALIDATE: same-track old item loses current-default safety but exact replacement is still unknown.
- NO_OP: not enough grounded evidence.
- Respect cardinality: single track usually has one active default; multi track may allow coexistence.
- If no same-track item is a grounded target, prefer ADD or NO_OP over forced REFINE/REPLACE.
"""


INVALIDATION_JUDGE_PROMPT = """Given one latent invalidation proposal and a small candidate set, decide whether an exact old CURRENT item has lost current-default status.

Output JSON only:
{
  "decision": "INDIRECT_INVALIDATE|WEAK_CHALLENGE|SET_UNKNOWN_CURRENT|NO_OP",
  "target_item_id": "optional existing item id, otherwise empty string",
  "reason": "brief grounded reason"
}

Rules:
- `same_track_target_candidates` are a strong default candidate pool, but not an automatic winner.
- `same_bucket_context_items` are context only; choose one only when `target_item_id_hint` clearly points to that exact active item.
- Decide only for this exact proposal and this exact target pool.
- INDIRECT_INVALIDATE: an exact old item lost current-default safety because its practical basis is broken or replaced.
- WEAK_CHALLENGE: an exact old item is still relevant, but its current-default status is weakened.
- SET_UNKNOWN_CURRENT: the old default is unsafe, but you still cannot confidently name one exact target or settled replacement.
- NO_OP: not enough exact item-level tension.
- Broad implication, topic drift, or general bucket change is not enough.
- Same-track proximity alone is not enough.
- Prefer the candidate whose actual basis is made unsafe, even if another item is easier to match superficially.
- If `target_old_value` is concrete, treat it as a strong anchor. Do not silently retarget to a nearby item with a different proposition/setup.
- Do not invalidate a newly added current item just because the track is related.
"""


STALE_SUPPORT_JUDGE_PROMPT = """Given one accepted semantic delta and one stale memory item, decide whether the delta provides a meaningful continuity link to that stale item.

Output JSON only:
{
  "link_type": "STALE_SUPPORT_LINK|STALE_SUCCESSOR_LINK|NO_LINK",
  "reason": "brief grounded reason",
  "confidence": 0.0
}

Rules:
- STALE_SUPPORT_LINK: the delta materially relates to or supports the stale item's line of state/history.
- STALE_SUCCESSOR_LINK: the delta looks like a grounded follow-up or successor to that stale item.
- Prefer `STALE_SUCCESSOR_LINK` when the delta can naturally replace the stale item's old setup with a new current setup, blocker, eligibility state, location basis, arrangement, or other answer-governing basis.
- If the stale item was directly invalidated by the same session and the delta expresses the replacement/current basis, preserve that bridge when grounded.
- NO_LINK: the connection is too weak, too broad, or only topical.
- Do not invent links from broad life-theme similarity alone.
- This step records linkage only; it does not reactivate or rewrite stale memory.
"""


QUERY_PARSER_PROMPT = """Parse a user query into compact fields for memory-based answering.

Output JSON only:
{
  "intent_type": "VALIDITY_CHECK|PREMISE_LADEN_QA|TASK_PLANNING",
  "target_bucket_tracks": [{"bucket": "...", "local_track": "..."}],
  "queried_proposition": "what the user is directly asking",
  "old_premise_focus": "old premise/setup being tested by the query",
  "basis_recovery_focus": "current-state basis to retrieve if that old premise is unsafe",
  "decision_basis_focus": "optional short legacy hint",
  "requested_action": "concrete action/decision, if any",
  "action_goal": "goal of the action, if any",
  "action_object": "main object, if any",
  "action_options": ["explicit compared options, if any"],
  "hidden_old_setup": "setup that must still hold for the action to make sense",
  "secondary_surface_details": ["surface details only"],
  "query_assumptions": [{"text": "assumption expressed by query, not confirmed memory", "bucket": "...", "local_track": "..."}],
  "option_comparison_requires_viability_check_first": true,
  "presuppositions": [{"text": "presupposed fact if any", "bucket": "...", "local_track": "..."}]
}

Rules:
- Use only the provided bucket/track schema.
- `queried_proposition` follows the user question; `old_premise_focus` is the old setup being tested; `basis_recovery_focus` is the current user-state/setup/capability/relation/location/eligibility basis the current-basis search should look for.
- Keep premise checking and current-basis search separate: do not replace the old premise with the current-basis search target, and do not turn the current-basis search into a broad topic summary.
- For planning/advice, identify the requested action first, then the action frame before logistics. The action frame is the current user state that makes the action valid or invalid, safe or unsafe, suitable or unsuitable, feasible or infeasible.
- Prefer the action frame for `basis_recovery_focus`. Use schedule, availability, payment logistics, route logistics, tool/method details, or generic convenience as `basis_recovery_focus` only when that surface detail is the actual gate.
- For planning queries, look for the current status, affiliation, obligation, capability/limitation, physical setup, location/base arrangement, access condition, safety condition, or recurring routine that would change the action. Do not collapse these into generic schedule, budget, productivity, or logistics unless those are truly the deciding state.
- Price, deposit, timing, schedule, tool/method choice, convenience, brand, wording, packaging, and option labels are usually `secondary_surface_details`, unless they are the direct hard gate.
- `query_assumptions` and `presuppositions` are not confirmed memory facts.
- For option/method comparisons, set `option_comparison_requires_viability_check_first` when the action itself may be unsuitable before comparing options.
- Preserve the query's core object/relation wording when it matters; avoid replacing it with generic practical topics.
"""


PREMISE_VERIFIER_PROMPT = """You will receive the parsed query (including `old_premise_focus`), a small set of primary hit bundles, a bounded premise-conflict subset, and optional secondary clarification evidence.

Output JSON only:
{
  "status": "SUPPORTED|OUTDATED|UNRESOLVED",
  "confidence": 0.0,
  "reasoning_summary": "brief explanation",
  "supporting_active_item_ids": ["item ids"],
  "supporting_active_item_values": ["item values"],
  "outdated_item_ids": ["item ids"],
  "outdated_item_values": ["item values"],
  "unknown_tracks": [{"bucket": "...", "local_track": "..."}],
  "corrected_current_facts": ["current facts usable for answering, if directly grounded"],
  "usable_current_basis_item_ids": ["active item ids allowed as current answer basis"],
  "usable_current_basis_item_values": ["active item values allowed as current answer basis"],
  "blocked_old_basis_item_ids": ["item ids that should not ground the current answer"],
  "blocked_old_basis_item_values": ["item values that should not ground the current answer"],
  "historical_but_overridden_item_ids": ["item ids usable only for history/explanation"],
  "historical_but_overridden_item_values": ["item values usable only for history/explanation"],
  "old_premise_safe": true
}

Input roles:
- `topk_hit_bundles` are the primary support for the initial judgment.
- `premise_conflict_subset` is a bounded exact-track supplement for adjudicating the tested old premise.
- `peripheral_context` and `retrieved_evidence` are secondary aids for clarification only.

Rules:
- Start from `parsed_query.old_premise_focus` and the main `topk_hit_bundles`.
- Use `premise_conflict_subset` only to inspect the tested premise's exact active / stale / unknown conflict when the main top-k line is incomplete or slightly off-target.
- Treat ACTIVE / STALE / UNKNOWN_CURRENT as store-side priors, not automatic truth labels.

Verdict:
- `SUPPORTED` normally requires settled current ACTIVE basis.
- `OUTDATED` when the tested old premise is made unsafe by current basis, stale/unknown conflict, or chain evidence.
- `UNRESOLVED` only when it is still unclear whether the tested old premise itself has failed.
- If the relevant track is `UNKNOWN_CURRENT`, do not support the old premise.
- If no ACTIVE item supports the premise, do not return `SUPPORTED`, even when old/raw evidence is highly relevant.
- Lack of ACTIVE support may reflect store lag rather than premise failure; do not convert that fact alone into `OUTDATED`.
- Prefer `OUTDATED` only when the tested old premise itself is made unsafe, not merely because a nearby conflict line or non-equivalent replacement exists.
- If relevant stale/archive evidence shows invalidation, prefer `OUTDATED` over `UNRESOLVED` even if the replacement is still unknown.

Output field guidance:
- `supporting_active_item_ids` must contain only ACTIVE item ids; `supporting_active_item_values` should mirror them as concrete values.
- `usable_current_basis_item_ids` are ACTIVE items that can safely ground the answer under this verdict; the matching values belong in `usable_current_basis_item_values`.
- Use `blocked_old_basis_item_ids` for items whose use would make the rejected old premise current again; store the mirrored values in `blocked_old_basis_item_values`.
- Use `historical_but_overridden_item_ids` for items that explain the historical setup but should not support current advice; store the mirrored values in `historical_but_overridden_item_values`.
- A current item may still be blocked if it directly supports an outdated old premise.
- `old_premise_safe` must be false whenever the old presupposition should not be used for planning or advice.

Facts:
- `corrected_current_facts` should be minimal, answer-usable facts grounded in usable current evidence or conflict-chain evidence.
- Do not promote nearby surface details into the main basis.
- Weak ACTIVE may support a cautious reading, but weak items alone are not settled current basis.
- Chain information may explain invalidation or succession, but does not by itself establish a settled replacement.
- If the old setup is unsafe and one concrete current condition already provides a more usable replacement setup for answering, prefer that concrete replacement fact over a purely generic “current state unknown” summary.

Clarification pass:
- Use `retrieved_evidence` only to clarify an already-surfaced premise conflict line.
- Old chunks or raw deltas that merely restate the old premise are historical support, not fresh current support.
- `peripheral_context`, `retrieved_evidence`, raw deltas, and raw evidence may explain ambiguity or support `OUTDATED` / `UNRESOLVED`, but they cannot by themselves authorize `SUPPORTED`.
"""


CONSTRAINT_BUILDER_PROMPT = """Build planning constraints from the compact current-state grounding selected earlier.

Output JSON only:
{
  "hard_positive_constraints": ["..."],
  "hard_negative_constraints": ["..."],
  "unresolved_variables": ["..."]
}

Input roles:
- `premise_verdict`: premise-level safety and blocked old basis
- `action_readiness_verdict`: action-level grounding decision when present
- `canonical_answer_grounding.decisive_basis`: strongest current-memory grounding
- `canonical_answer_grounding.blocker_basis`: direct blockers when present
- `canonical_answer_grounding.supporting_basis`: small supporting context only

Rules:
- Translate `canonical_answer_grounding.decisive_basis` and `blocker_basis` into constraints before looking at supporting context.
- If a current fact directly limits the capability, eligibility, access, setup, suitability, or safety required by the requested action, put it in `hard_negative_constraints`, not merely in unresolved variables.
- If action readiness says `BLOCK_AND_REFRAME`, include the blocking current condition as a hard negative and do not build positives from the rejected old setup.
- If action readiness says `PROCEED_WITH_ADJUSTMENT`, include the required adjustment as a hard positive or hard negative as appropriate.
- `query_assumptions` are not hard positive constraints unless current memory independently confirms them.
- Do not derive current hard constraints from `blocked_old_basis_item_ids` or historical-only items.
- Do not invent constraints from raw retrieval or debug evidence not present in `canonical_answer_grounding`.
- Supporting basis may add nuance, but should not override decisive/blocker grounding.
- Keep constraints concrete and planning-relevant.
"""

ACTION_GROUNDING_RESOLVER_PROMPT = """Resolve the current memory basis for the requested action.

You receive a user query, parsed action fields, a premise verdict, and candidate memory bundles. Do not re-check the old premise. Use `premise_verdict` as given. Keep only candidates whose current condition would change whether the requested action should proceed, be adjusted, blocked, or reframed.

Output JSON only:
{
  "status": "PROCEED|PROCEED_WITH_ADJUSTMENT|BLOCK_AND_REFRAME|ASK_MIN_CLARIFICATION",
  "confidence": 0.0,
  "reasoning_summary": "brief explanation",
  "decisive_gate_bundle_ids": ["bundle ids"],
  "blocker_bundle_ids": ["bundle ids"],
  "supporting_basis_bundle_ids": ["bundle ids"],
  "rejected_nearby_bundle_ids": ["bundle ids"],
  "corrected_current_facts": ["current facts that should ground the answer"],
  "reframe_mode": "short description of the adjusted or blocked action, if any"
}

Rules:
- Do not use blocked old or historical-only facts.
- Select at most 2 decisive bundles and at most 1 supporting bundle.
- Prefer the following kinds of grounding when available:
  1. Direct successor or replacement of an outdated old premise.
  2. Concrete current blocker/enabler for capability, eligibility, status, obligation, physical setup, location, access, safety, or required routine.
  3. Current affiliation/status/setup that frames what the action means.
  4. Schedule or logistics only if schedule/logistics is the actual gate.
- Reject nearby current facts that are merely relevant but would not change the answer.
- If a generic schedule/logistics bundle competes with a capability/status/obligation/setup/safety bundle, reject the generic bundle.
- Query assumptions are not memory-confirmed facts.
- If all compared options depend on the same blocked capability or invalid setup, do not choose among the options; block or reframe the action.
- UNKNOWN_CURRENT may explain why an old premise is unsafe, but use it as decisive only when it names a specific blocker and no concrete candidate gives the answer-changing condition.
- `corrected_current_facts` must be copied or compressed only from `decisive_gate_bundle_ids` or `blocker_bundle_ids`.
- `PROCEED`: current memory supports the action as framed.
- `PROCEED_WITH_ADJUSTMENT`: the action can continue only with a current-state adjustment or narrower version.
- `BLOCK_AND_REFRAME`: current memory shows the framed action is unsuitable, unsafe, infeasible, ineligible, or based on an invalid setup; answer should recommend a safer current-state-compatible alternative or stop.
- `ASK_MIN_CLARIFICATION`: use only when premise is not safe and no current basis is sufficient to choose a safe bounded answer.
- Keep `corrected_current_facts` concrete, minimal, and answer-usable.
- Use only bundle ids that appear in `candidate_bundles`.
"""


ANSWER_COMPOSER_PROMPT = """Compose the final answer from the query, parsed query, premise verdict, optional action readiness, constraints, and compact `canonical_answer_grounding`.

Output JSON only:
{
  "answer": "final natural-language answer",
  "brief_rationale": "very short explanation"
}

Authority order:
1. `premise_verdict`
2. `action_readiness_verdict` when present
3. `canonical_answer_grounding`
4. `constraint_set`

Rules:
- Do not re-judge truth or mention internal labels such as basis, bundle, ACTIVE, STALE, UNKNOWN_CURRENT, or verdict names.
- Use only facts in the provided grounding, corrected-current-facts, or constraints.
- Do not revive blocked or historical old items as current grounding.
- Use directly relevant decisive, blocker, and supporting facts when compatible; do not list nearby details that do not change the answer.

Answer shape:
- Assumption check: briefly say whether the assumption is still safe. If the old assumption is unsafe and concrete current grounding is present, state the old assumption is no longer valid/safe and cite that current fact; do not answer only “unclear” unless no concrete current grounding is available.
- Premise-laden content request: if the request relies on an unsafe old setup, correct that setup first and do not produce the requested template, plan, schedule, comparison, or instructions under it. Give only a short current-state-safe alternative if grounded.
- Current action/planning decision: decide the action from current grounding. Give the safest concrete next move, and make the main reason the current condition that changes suitability, feasibility, safety, eligibility, or setup validity.

False-premise requests:
- If `old_premise_safe` is false or the premise status says the old setup is outdated, the first sentence must push back on the query's premise.
- Do not satisfy the old-premise request just because the user asks for a routine, checklist, schedule, draft, comparison, or plan.
- If the current replacement is concrete, name it briefly and give at most one safe alternative direction.
- If the replacement is incomplete, still refuse the old-premise request; say the old setup is no longer safe to assume rather than giving advice under it.

When current grounding already gives a concrete replacement state or blocker:
- Do not fall back to “first assess”, “first verify”, “first confirm”, or “first diagnose”.
- Do not compare surface options under a blocked setup.
- Do not choose a convenient but ungrounded reason over a more decision-governing current fact.

Style:
- Be direct, concise, and grounded.
- If the action is blocked, say so clearly and give the safest supported reframe, alternative, postponement, or stop recommendation.
- If grounding is incomplete, say so briefly, but still give the safest bounded recommendation available.
"""

QUERY_LANE_FILTER_PROMPT = """You filter one bounded query candidate group.

You receive a user query, parsed query fields, optional premise context, a candidate-group label, and embedding-ranked candidate bundles.
Keep only candidates that can materially change the answer. Do not keep merely nearby memories.

A useful candidate may be:
- current user-state grounding that changes the answer;
- a direct action blocker or enabler for suitability, feasibility, safety, eligibility, access, setup, or option choice;
- conflict/replacement/successor evidence explaining why an old premise should not be used;
- a concrete practical implication the final answer must respect.

Output JSON only:
{
  "kept_refs": ["source:id refs from candidates"],
  "useful_facts": ["brief facts grounded in kept candidates"],
  "rejected_refs": [{"ref": "source:id", "reason": "brief reason"}],
  "reasoning_summary": "brief summary"
}

Rules:
- Do not decide the final answer or re-run premise verification.
- Filter for answer-changing evidence: keep a candidate only if its concrete condition would change the correction, refusal, plan, reframe, or next step.
- Treat store status as context, not the decision rule. A STALE item can still be useful when it names the old setup, conflict line, successor clue, or concrete state that explains why the old premise should not be used.
- Prefer explicit memory propositions over uncertainty summaries when selecting answer-governing evidence. Keep uncertainty summaries only when they provide the most concrete available blocker or conflict signal.
- For premise-laden queries, keep candidates that invalidate, replace, or make unsafe the queried old setup, even without the same surface words.
- For action/planning queries, keep candidates that directly affect whether the requested action should proceed, be reframed, postponed, or avoided. If removing the candidate would not change the answer, reject it as nearby.
- Prefer concrete user-state conditions over task-object status, generic logistics, same-topic memories, or historical background.
- `useful_facts` must be grounded in kept candidates only.
- If nothing is decision-governing, return empty `kept_refs` and empty `useful_facts`.
"""
