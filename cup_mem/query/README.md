# Query modules

The query package implements conflict-aware memory readout and answer grounding.
It is organized as a multi-step memory-query process rather than a single retrieval prompt.

## Stages

- `readout.py`: primary item retrieval, peripheral conflict retrieval for the
  verifier clarification pass, and bundle construction.
- `track_hints.py`: parser-derived bucket/track hints and bounded premise
  conflict subsets.
- `premise_verifier.py`: validates whether the query's old premise is still safe
  and separates usable current basis from blocked or historical-only items.
- `basis_recovery_impl.py`: current-basis recovery from compact item and
  conflict candidates.
- `action_grounding.py`: selects the memory facts that govern action/planning
  decisions.
- `support.py`: shared query helpers, LLM calls, answer/constraint construction,
  and trace summaries.
- `engine.py`: end-to-end query orchestration.

## Clarification retrieval

The query engine may run a bounded clarification pass when the initial premise verdict
is unresolved, low-confidence, or exposes a stale/unknown conflict line. This is
not a general answer fallback: the additional evidence is used to re-adjudicate
premise safety before current-basis recovery and answer grounding continue.
