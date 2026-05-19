[![arXiv](https://img.shields.io/badge/arXiv-2605.06527-b31b1b.svg)](https://arxiv.org/abs/2605.06527)

# STALE and CUP-Mem

This repository contains the code and resources for our paper:

**STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?**  
[arXiv:2605.06527](https://arxiv.org/abs/2605.06527)

- `STALE/`: dataset generation and evaluation scripts for the STALE benchmark.
- `cup_mem/`: the CUP-Mem memory pipeline for session-by-session profile updates and conflict-aware query answering.

## Main Results

We evaluate each model on two implicit-conflict types and three probing dimensions:
State Resolution (SR), Premise Resistance (PR), and Implicit Policy Adaptation (IPA).
Overall denotes the average accuracy across all six settings.

| Model | Type I SR | Type I PR | Type I IPA | Type II SR | Type II PR | Type II IPA | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o-mini* | 30.0 | 0.0 | 11.0 | 9.5 | 0.0 | 1.5 | 8.7 |
| GPT-5.4-nano | 20.5 | 1.5 | 21.5 | 9.0 | 0.0 | 6.5 | 9.8 |
| GPT-5.4 | 35.0 | 2.0 | 29.0 | 9.0 | 2.0 | 17.0 | 15.7 |
| Gemini-3.1-flash-lite | 41.0 | 1.5 | 42.0 | 25.0 | 1.5 | 23.5 | 22.4 |
| Gemini-3.1-pro | 92.0 | 30.0 | 71.0 | 69.0 | 14.0 | 55.0 | 55.2 |
| Claude-Opus-4.6 | 50.5 | 10.0 | 42.5 | 21.0 | 9.0 | 36.5 | 28.3 |
| Llama-3.3-70B-Instruct* | 6.5 | 0.0 | 3.0 | 6.0 | 0.0 | 0.0 | 2.6 |
| Qwen3.5-9B | 36.0 | 1.0 | 21.5 | 21.5 | 0.0 | 7.5 | 14.6 |
| Qwen3.5-27B | 76.0 | 4.0 | 39.0 | 42.0 | 3.5 | 23.0 | 31.3 |
| MiniMax-M2.5 | 10.5 | 1.5 | 8.0 | 5.5 | 5.0 | 2.5 | 5.5 |
| LightMem | 52.5 | 1.0 | 23.5 | 21.5 | 0.5 | 7.5 | 17.8 |
| Zep | 10.0 | 0.0 | 19.0 | 3.0 | 1.0 | 3.0 | 6.0 |
| LiCoMemory | 15.5 | 0.5 | 22.5 | 1.5 | 1.5 | 4.0 | 7.6 |
| A-mem | 13.5 | 0.0 | 7.5 | 8.0 | 0.0 | 1.5 | 5.1 |
| mem-0 | 17.0 | 1.0 | 22.0 | 3.5 | 0.0 | 6.5 | 8.3 |
| **CUPMem (Ours)** | 91.0 | **78.0** | 32.0 | **89.0** | **75.0** | 43.0 | **68.0** |

All numbers are accuracies (%).  
`*` indicates settings where evidence-preserving truncation was applied because the full context exceeded the model's context window.

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
