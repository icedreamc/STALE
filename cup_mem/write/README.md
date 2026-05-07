# Write pipeline modules

The write package implements chronological profile construction from user
sessions. It extracts compact state evidence, converts it into typed deltas,
applies local updates, searches for propagated invalidations, and records
stale/support links for later query-time reasoning.

- `chunker.py`: session turns -> state chunks
- `delta_extractor.py`: chunks -> state deltas
- `update_resolver.py`: deltas -> ADD / REFINE / REPLACE / NO_OP decisions
- `invalidation_lanes.py`: affected-bucket discovery and scoped invalidation proposal generation
- `invalidation_merge.py`: merge and deterministic selection of scoped proposals
- `invalidation_judge.py`: final invalidation judge over selected proposals and candidate items
- `invalidation_resolver.py`: orchestration for invalidation proposal generation
- `stale_linker.py`: stale/successor relation construction
- `writer.py`: one-session write orchestration and trace assembly
