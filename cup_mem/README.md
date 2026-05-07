# CUP-Mem

This directory contains the CUP-Mem code.
CUP-Mem is a structured memory pipeline for session-by-session profile updates
and conflict-aware query answering.

## Requirements

Core runtime dependencies:

- Python 3.10+
- `openai`
- `httpx`
- `torch`
- `transformers`

CUP-Mem expects an OpenAI-compatible chat endpoint or Responses endpoint for
LLM calls, and a local sentence embedding model directory for retrieval. The
experiments used a local `all-MiniLM-L6-v2` model. No API keys or model weights
are included in this package.

Install minimal dependencies with:

```bash
pip install -r requirements.txt
```

The minimal runner uses:

- `STALE/.env`
- `STALE/outputs/*_MAIN.json`
- `cup_mem/models/all-MiniLM-L6-v2`

`run_cup_mem.py` loads `STALE/.env` automatically and uses
`cup_mem/models/all-MiniLM-L6-v2` automatically when that directory exists.

## Minimal runner

Generate a demo split first:

```bash
cd STALE
cp .env.example .env
python Generation/StepALL_IC_gen.py \
  --seed-file data/ontology_seeds_demo5.json \
  --output-name demo_T1 \
  --conflict-type T1 \
  --output-dir outputs \
  --num-workers 1
```

Then run one sample from the repository root:

```bash
python -m cup_mem.run_cup_mem \
  --data-path STALE/outputs/demo_T1_MAIN.json \
  --sample-index 0 \
  --session-mode relevant_only
```

Use `--session-mode full` for the full 50-session context. If the embedding
model is stored outside `cup_mem/models/all-MiniLM-L6-v2`, pass
`--embedding-model-path` explicitly.
