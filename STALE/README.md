# STALE Dataset Generation

This directory contains the generation pipeline for our STALE dataset. The pipeline creates old user states, generates updated states that implicitly invalidate them, builds probing questions, turns facts into chat sessions, and assembles final evidence and haystack-session files.

## Structure

- `Generation/StepALL_IC_gen.py`: end-to-end generation entrypoint.
- `Generation/Step0_gen_funcs.py`: old-state and implicit-conflict pair generation helpers.
- `Generation/Step1_qagen_eval_funcs.py`: probing-query generation helpers.
- `Generation/Step2_info2session_funcs.py`: conversion from facts to chat sessions.
- `Generation/Step3_ICdatasetgen.py`: timestamp auditing, noise-session filtering, and final haystack assembly helpers.
- `Generation/IC_gen_config.py`: environment-variable based runtime configuration.
- `Generation/clients.py`: OpenAI-compatible client construction.
- `Evaluation/run_target_model.py`: runs a target model on generated probing queries, with optional token-aware trimming.
- `Evaluation/full_eval_performance.py`: judges target-model responses and summarizes accuracy/usage.

## Setup

```bash
cd STALE
conda create -n stale python=3.10 -y
conda activate stale
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

If Python 3.10+ is already installed locally, `venv` is also fine:

```bash
cd STALE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with provider API keys and paths. The runtime loads `STALE/.env` automatically, and API keys should never be committed.

Download a proper noise-session data is required.

Noise-session data should be a JSON list of chat sessions. Each session is a list of messages with `role` and `content`.

## Environment Variables

The client helper reads provider-specific variables. For example, `GPT_PROVIDER=OPENAI` uses `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`; `QWEN_PROVIDER=QWEN` uses `QWEN_API_KEY` and optional `QWEN_BASE_URL`.

Important path variables:

- `NOISE_DATASET_DIR`: JSON file containing background chat sessions.
- `ROOT_DATASET_DIR`: output directory for generated files.
- `IC_DATA_DIR`: default local data directory.

## Usage

Run the full pipeline from the `STALE` directory:

```bash
python Generation/StepALL_IC_gen.py \
  --seed-file data/ontology_seeds_demo5.json \
  --output-name demo_T1 \
  --conflict-type T1 \
  --output-dir outputs \
  --num-workers 1
```

Use `data/ontology_seeds_demo5.json` for a small smoke-test run, or `data/ontology_seeds.json` for the full ontology seed list.

Run target-model responses on a generated dataset:

```bash
python Evaluation/run_target_model.py \
  --icds-path outputs/demo_T1_MAIN.json \
  --output-path outputs/demo_T1_answers.json
```

By default this sends the full haystack without trimming. Enable token-aware trimming only when needed:

```bash
python Evaluation/run_target_model.py \
  --icds-path outputs/demo_T2_MAIN.json \
  --output-path outputs/demo_T2_answers_trimmed.json \
  --enable-trim
```

For vLLM or any OpenAI-compatible local server, pass the base URL directly:

```bash
python Evaluation/run_target_model.py \
  --icds-path outputs/demo_T2_MAIN.json \
  --output-path outputs/demo_T2_vllm_answers.json \
  --model Qwen3.5-9B \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY
```

Judge the target-model responses:

```bash
python Evaluation/full_eval_performance.py \
  --answers-path outputs/demo_T1_answers.json \
  --dataset-path outputs/demo_T1_MAIN.json \
  --output-path outputs/demo_T1_eval.json \
  --conflict-type T1 \
  --model-method target_model
```

`Evaluation/full_eval_performance.py` is dataset-name agnostic. It can judge any pair of JSON files that follow this format:

Ground-truth dataset records:

```json
{
  "uid": "sample id",
  "M_old": "old user state",
  "M_new": "new user state",
  "explanation": "why M_new invalidates M_old",
  "probing_queries": {
    "dim1_query": "...",
    "dim2_query": "...",
    "dim3_query": "..."
  }
}
```

Target-model answer records:

```json
{
  "uid": "sample id",
  "target_model_responses": {
    "dim1_response": "...",
    "dim2_response": "...",
    "dim3_response": "..."
  }
}
```

Both files may be either a JSON list of records or an object with a list-valued `data` field.

The pipeline writes:

- `<output-name>_MAIN.json`: final dataset entries with timestamps and haystack sessions.
- `<output-name>_EVID.json`: evidence records for generated conflict pairs.
- `<output-name>_part*_*.json`: worker outputs before aggregation.
- `<output-name>_part*_checkpoint.pkl`: intermediate checkpoints.

Timestamp generation first anchors `M_old` to a plausible 2027 datetime, generates a later `M_new` timestamp from the old/new pair and the annotated gap, audits whether `query_time` still supports the dim1 and dim3 probes, and ensures the final timestamp equals `query_time`. Before writing the final schedule, all timestamps are shifted by whole years so the final query timestamp falls in 2025 while preserving relative intervals.

Haystack generation samples middle noise sessions against `M_old` and post-`S_new` noise sessions against both `M_old` and `M_new`, while avoiding duplicate sessions.

## Input Format

Seed items can be minimal ontology entries:

```json
{
  "attribute": ["major attribute", "sub attribute"]
}
```

Noise-session data should be a JSON list of chat sessions. Each session is a list of messages with `role` and `content`.
