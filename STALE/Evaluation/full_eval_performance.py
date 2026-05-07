import json
import asyncio
import time
import argparse
import os
import sys
from pathlib import Path
try:
    from tqdm.asyncio import tqdm
except ImportError:
    class _AsyncTqdmFallback:
        @staticmethod
        async def gather(*aws, desc=None):
            return await asyncio.gather(*aws)

    tqdm = _AsyncTqdmFallback()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Generation.clients import get_Async_client
from judge_prompts import SYSTEM_PROMPT_ALL_IN_ONE_JUDGE
import re

EVAL_PROVIDER = os.getenv("JUDGE_PROVIDER", "OPENAI")
eval_client = get_Async_client(EVAL_PROVIDER)
EVAL_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.1-flash-lite-preview")
CONCURRENCY_LIMIT = 20

def load_records(path):
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj["data"]

    raise ValueError(f"{path} must be a JSON list or an object with a list-valued 'data' field.")


def validate_dataset_record(item):
    uid = item.get("uid")
    if not uid:
        raise ValueError("Dataset record is missing uid.")
    if not (item.get("M_old") or item.get("old_info")):
        raise ValueError(f"Dataset record uid={uid} is missing M_old or old_info.")
    if "M_new" not in item:
        raise ValueError(f"Dataset record uid={uid} is missing M_new.")

    probing_queries = item.get("probing_queries")
    if not isinstance(probing_queries, dict):
        raise ValueError(f"Dataset record uid={uid} is missing probing_queries.")
    for dim_key in ("dim1_query", "dim2_query", "dim3_query"):
        if not isinstance(probing_queries.get(dim_key), str):
            raise ValueError(f"Dataset record uid={uid} is missing probing_queries.{dim_key}.")


def validate_answer_record(item):
    uid = item.get("uid")
    if not uid:
        raise ValueError("Answer record is missing uid.")

    responses = item.get("target_model_responses")
    if not isinstance(responses, dict):
        raise ValueError(f"Answer record uid={uid} is missing target_model_responses.")
    for dim_key in ("dim1_response", "dim2_response", "dim3_response"):
        if not isinstance(responses.get(dim_key), str):
            raise ValueError(f"Answer record uid={uid} is missing target_model_responses.{dim_key}.")

def safe_div(a, b):
    return a / b if b else 0.0


def normalize_usage(usage_obj):
    if usage_obj is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    if isinstance(usage_obj, dict):
        return {
            "prompt_tokens": int(usage_obj.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_obj.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_obj.get("total_tokens", 0) or 0),
        }

    return {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }


def init_usage_stats():
    return {
        "count": 0,
        "elapsed_seconds_sum": 0.0,
        "prompt_tokens_sum": 0,
        "completion_tokens_sum": 0,
        "total_tokens_sum": 0,
    }


def add_usage_stats(bucket, elapsed_seconds, usage):
    usage = normalize_usage(usage)

    bucket["count"] += 1
    bucket["elapsed_seconds_sum"] += float(elapsed_seconds or 0.0)
    bucket["prompt_tokens_sum"] += usage["prompt_tokens"]
    bucket["completion_tokens_sum"] += usage["completion_tokens"]
    bucket["total_tokens_sum"] += usage["total_tokens"]


def finalize_usage_stats(bucket):
    count = bucket["count"]
    return {
        **bucket,
        "elapsed_seconds_avg": safe_div(bucket["elapsed_seconds_sum"], count),
        "prompt_tokens_avg": safe_div(bucket["prompt_tokens_sum"], count),
        "completion_tokens_avg": safe_div(bucket["completion_tokens_sum"], count),
        "total_tokens_avg": safe_div(bucket["total_tokens_sum"], count),
    }


def get_query_text(info, dim_key):
    probing_queries = info.get("probing_queries", {}) or {}
    value = probing_queries.get(f"{dim_key}_query", "")
    if isinstance(value, str):
        return value
    print(f'WARNING:{value}')
    return ""


async def async_evaluate_all_dimensions(
    semaphore,
    uid, task_type,
    old_info: str, M_new: str, explanation: str,
    dim1_q: str, dim1_a: str,
    dim2_q: str, dim2_a: str,
    dim3_q: str, dim3_a: str,
    client, model_name,
):
    user_prompt = f"""
[Ground Truth Context]
- M_old: "{old_info}"
- M_new: "{M_new}"
- Hidden Logic: {explanation}

--------------------------------------------------
[Dimension 1: Explicit Probing]
Question 1: {dim1_q}
Target Model Response 1: {dim1_a}

--------------------------------------------------
[Dimension 2: Adversarial Robustness]
Question 2: {dim2_q}
Target Model Response 2: {dim2_a}

--------------------------------------------------
[Dimension 3: Implicit Task]
Question 3: {dim3_q}
Target Model Response 3: {dim3_a}
"""

    async with semaphore:
        judge_start = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ALL_IN_ONE_JUDGE},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            judge_elapsed = time.perf_counter() - judge_start

            content = response.choices[0].message.content
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
            if match:
                content = match.group(1)
            result = json.loads(content)

            judge_usage = normalize_usage(getattr(response, "usage", None))

            return uid, task_type, result, {
                "elapsed_seconds": judge_elapsed,
                "usage": judge_usage
            }

        except Exception as e:
            judge_elapsed = time.perf_counter() - judge_start
            print(f"[Judge Error] uid={uid}: {e}")
            return uid, task_type, {
                "dim1_eval": {"reasoning": f"Error: {e}", "pass": False},
                "dim2_eval": {"reasoning": f"Error: {e}", "pass": False},
                "dim3_eval": {"reasoning": f"Error: {e}", "pass": False}
            }, {
                "elapsed_seconds": judge_elapsed,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                "error": str(e)
            }


async def run_evaluation(answers_path, dataset_path, output_path, model_method, conflict_type):
    if eval_client is None:
        raise RuntimeError("Judge client is not configured. Set the provider API key or pass --judge-provider.")

    print("Loading datasets...")
    t_answers = load_records(answers_path)
    t_info = load_records(dataset_path)

    for item in t_answers:
        validate_answer_record(item)
    for item in t_info:
        validate_dataset_record(item)

    t_info_dict = {item['uid']: item for item in t_info}
    t_answer_dict = {item['uid']: item for item in t_answers}

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []

    def build_eval_tasks(answers, info_dict, task_type):
        for ans in answers:
            uid = ans.get('uid')
            info = info_dict.get(uid)
            if not info:
                continue

            old_info = info.get('old_info', info.get('M_old', ''))

            dim1_q = get_query_text(info, 'dim1')
            dim2_q = get_query_text(info, 'dim2')
            dim3_q = get_query_text(info, 'dim3')

            target_model_responses = ans.get('target_model_responses', {})

            tasks.append(async_evaluate_all_dimensions(
                semaphore, uid, task_type,
                old_info,
                info.get('M_new', ''),
                info.get('explanation', ''),
                dim1_q, target_model_responses.get('dim1_response', ''),
                dim2_q, target_model_responses.get('dim2_response', ''),
                dim3_q, target_model_responses.get('dim3_response', ''),
                eval_client, EVAL_MODEL
            ))

    print(f"Building evaluation tasks for {conflict_type}...")
    build_eval_tasks(t_answers, t_info_dict, conflict_type)

    print(f"Firing {len(tasks)} Judge APIs...")
    run_start = time.perf_counter()
    results = await tqdm.gather(*tasks, desc="Judging in progress")
    run_wall_clock_seconds = time.perf_counter() - run_start

    dims = ['dim1', 'dim2', 'dim3']

    accuracy_stats = {
        conflict_type: {
            "dim1": {"correct": 0, "total": 0},
            "dim2": {"correct": 0, "total": 0},
            "dim3": {"correct": 0, "total": 0},
        }
    }

    target_stats = {
        conflict_type: {
            "dim1": init_usage_stats(),
            "dim2": init_usage_stats(),
            "dim3": init_usage_stats(),
            "overall": init_usage_stats(),
        }
    }

    judge_stats = {
        conflict_type: init_usage_stats()
    }

    final_results_log = []

    for uid, task_type, eval_res, judge_meta in results:
        ans = t_answer_dict.get(uid, {})
        target_model_meta = ans.get("target_model_meta", {}) or {}

        final_results_log.append({
            "uid": uid,
            "task_type": task_type,
            "evaluation": eval_res,
            "target_model_meta": target_model_meta,
            "judge_meta": judge_meta,
        })

        for dim_key in dims:
            passed = bool(eval_res.get(f'{dim_key}_eval', {}).get('pass', False))
            accuracy_stats[task_type][dim_key]["total"] += 1
            if passed:
                accuracy_stats[task_type][dim_key]["correct"] += 1

        for dim_key in dims:
            dim_meta = target_model_meta.get(f"{dim_key}_meta", {}) or {}
            elapsed_seconds = dim_meta.get("elapsed_seconds", 0.0)
            usage = dim_meta.get("usage", {})

            add_usage_stats(target_stats[task_type][dim_key], elapsed_seconds, usage)
            add_usage_stats(target_stats[task_type]["overall"], elapsed_seconds, usage)

        add_usage_stats(
            judge_stats[task_type],
            judge_meta.get("elapsed_seconds", 0.0),
            judge_meta.get("usage", {})
        )

    accuracy_summary = {}
    for task_type, dim_dict in accuracy_stats.items():
        accuracy_summary[task_type] = {}
        overall_correct = 0
        overall_total = 0

        for dim_key in dims:
            correct = dim_dict[dim_key]["correct"]
            total = dim_dict[dim_key]["total"]
            acc = safe_div(correct, total)

            accuracy_summary[task_type][dim_key] = {
                "correct": correct,
                "total": total,
                "accuracy": acc
            }

            overall_correct += correct
            overall_total += total

        accuracy_summary[task_type]["overall"] = {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": safe_div(overall_correct, overall_total)
        }

    target_stats_summary = {}
    for task_type, stat_dict in target_stats.items():
        target_stats_summary[task_type] = {
            dim_key: finalize_usage_stats(stat_dict[dim_key])
            for dim_key in ["dim1", "dim2", "dim3", "overall"]
        }

    judge_stats_summary = {}
    for task_type, stat_dict in judge_stats.items():
        judge_stats_summary[task_type] = finalize_usage_stats(stat_dict)

    output_json = {
        "config": {
            "model_method": model_method,
            "conflict_type": conflict_type,
            "judge_model": EVAL_MODEL,
            "concurrency_limit": CONCURRENCY_LIMIT,
            "num_samples": len(final_results_log),
        },
        "run_summary": {
            "judge_wall_clock_seconds": run_wall_clock_seconds,
            "judge_request_count": len(results),
        },
        "summary": {
            "accuracy": accuracy_summary,
            "target_model_stats": target_stats_summary,
            "judge_stats": judge_stats_summary,
        },
        "details": final_results_log
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    print(f"Saved evaluation summary to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Judge target-model responses against STALE ground truth.")
    parser.add_argument("--answers-path", required=True, help="Path to a target-model response JSON list, or an object with a data list.")
    parser.add_argument("--dataset-path", required=True, help="Path to a ground-truth dataset JSON list, or an object with a data list.")
    parser.add_argument("--output-path", required=True, help="Path to save judged evaluation results.")
    parser.add_argument("--model-method", default="target_model", help="Label for the evaluated method/model.")
    parser.add_argument("--conflict-type", required=True, help="Dataset/task label used in the output summary.")
    parser.add_argument("--judge-model", default=EVAL_MODEL, help="Judge model name.")
    parser.add_argument("--judge-provider", default=EVAL_PROVIDER, help="Provider name resolved by environment variables.")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY_LIMIT, help="Concurrent judge API calls.")
    return parser.parse_args()


if __name__ == "__main__":
    import platform
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    args = parse_args()
    EVAL_MODEL = args.judge_model
    EVAL_PROVIDER = args.judge_provider
    CONCURRENCY_LIMIT = args.concurrency
    eval_client = get_Async_client(EVAL_PROVIDER)

    asyncio.run(
        run_evaluation(
            answers_path=args.answers_path,
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            model_method=args.model_method,
            conflict_type=args.conflict_type,
        )
    )
