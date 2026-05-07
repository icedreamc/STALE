# STALE and CUP-Mem

This repository contains two related components:

- `STALE/`: dataset generation and evaluation scripts for the STALE benchmark.
- `cup_mem/`: the CUP-Mem memory pipeline for session-by-session profile updates and conflict-aware query answering.

## Setup

Use Python 3.10 or newer. Create one environment for each component:

```bash
cd STALE
conda create -n stale python=3.10 -y
conda activate stale
python -m pip install -r requirements.txt
cp .env.example .env
```

```bash
cd cup_mem
conda create -n cupmem python=3.10 -y
conda activate cupmem
python -m pip install -r requirements.txt
```

Fill `STALE/.env` with provider keys and local paths before running generation or evaluation.

## Components

`STALE/` provides:

- ontology-seed based data generation
- timestamp and haystack assembly
- target-model response generation
- automatic response judging and performance summaries

See `STALE/README.md` for commands and expected input/output formats.

`cup_mem/` provides:

- structured memory write and invalidation logic
- retrieval and premise verification
- conflict-aware readout for memory-dependent queries
- OpenAI-compatible and Responses API client wrappers
- a single-sample runner that can consume `STALE/outputs/*_MAIN.json` directly

See `cup_mem/README.md` for minimal usage.
