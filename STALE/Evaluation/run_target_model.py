import json
import asyncio
import random
import copy
import time
import platform
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

try:
    import tiktoken
except ImportError:
    tiktoken = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Generation.clients import get_Async_client

EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", "OPENAI")
aclient = get_Async_client(EVAL_PROVIDER)
EVAL_MODEL = os.getenv("TARGET_MODEL", "gpt-4o-mini")

CONCURRENCY_LIMIT = 10
MAX_RETRIES = 3
ENABLE_TRIM = False
MAX_OUTPUT_TOKENS = None

TOKEN_LIMIT = 128000
RESERVED_OUTPUT_TOKENS = 2048
SAFETY_MARGIN = 512
INPUT_TOKEN_LIMIT = TOKEN_LIMIT - RESERVED_OUTPUT_TOKENS - SAFETY_MARGIN


def format_haystack(haystack_sessions, timestamps=None):
    formatted_history = ""
    for idx, session in enumerate(haystack_sessions):
        if not session:
            continue
        time_str = f" [Time: {timestamps[idx]}]" if timestamps and idx < len(timestamps) else ""
        formatted_history += f"\n=== Session {idx + 1}{time_str} ===\n"
        for turn in session:
            role = "User" if turn["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {turn['content']}\n"
    return formatted_history


def get_encoder(model_name: str):
    """
    Use tiktoken for the target model.
    Fall back to o200k_base when the local tiktoken version does not recognize the model.
    """
    if tiktoken is None:
        raise RuntimeError("tiktoken is required for token trimming. Install dependencies with: pip install -r requirements.txt")
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


ENCODER = None


def num_tokens_from_messages(messages, model=None) -> int:
    """
    Approximate chat.completions message tokens for trimming control.
    """
    model = model or EVAL_MODEL
    try:
        encoding = get_encoder(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def build_prompts(history_text: str, query_text: str, dim_key: str):
    if dim_key == "dim3_query":
        sys_prompt = (
            "You are a helpful assistant. Review the following conversation history "
            "with the user, then respond to the user's latest query directly."
        )
        user_prompt = f"[Conversation History]\n{history_text}\n\n[Latest Query]\n{query_text}"
    else:
        sys_prompt = (
            "You are a helpful assistant. Review the following conversation history "
            "with the user, then accurately answer the question."
        )
        user_prompt = f"[Conversation History]\n{history_text}\n\n[Question]\n{query_text}"

    return sys_prompt, user_prompt


def estimate_request_tokens(history_text: str, query_text: str, dim_key: str) -> int:
    sys_prompt, user_prompt = build_prompts(history_text, query_text, dim_key)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return num_tokens_from_messages(messages, EVAL_MODEL)


def trim_middle_noise_to_token_limit(item, dim_key, query_text, token_limit=None):
    """
    Trim each dimension separately because query lengths differ.
    The target is full request input tokens, not only history_text.
    """
    token_limit = token_limit or INPUT_TOKEN_LIMIT
    sessions = copy.deepcopy(item["haystack_session"])
    timestamps = copy.deepcopy(item.get("timestamps", []))

    idx_old, idx_new = item["relevant_session_index"]

    history_text = format_haystack(sessions, timestamps)
    cur_tokens = estimate_request_tokens(history_text, query_text, dim_key)

    if cur_tokens <= token_limit:
        return sessions, timestamps, cur_tokens, 0

    removed_count = 0

    # Phase 1: remove up to two head sessions without touching S_old.
    head_remove_budget = min(2, idx_old)
    for _ in range(head_remove_budget):
        if cur_tokens <= token_limit:
            break

        sessions.pop(0)
        if timestamps:
            timestamps.pop(0)

        removed_count += 1
        idx_old -= 1
        idx_new -= 1

        history_text = format_haystack(sessions, timestamps)
        cur_tokens = estimate_request_tokens(history_text, query_text, dim_key)

    if cur_tokens <= token_limit:
        return sessions, timestamps, cur_tokens, removed_count

    # Phase 2: remove up to two tail sessions without touching S_new.
    tail_available = len(sessions) - 1 - idx_new
    tail_remove_budget = min(2, tail_available)

    for _ in range(tail_remove_budget):
        if cur_tokens <= token_limit:
            break

        sessions.pop()
        if timestamps:
            timestamps.pop()

        removed_count += 1

        history_text = format_haystack(sessions, timestamps)
        cur_tokens = estimate_request_tokens(history_text, query_text, dim_key)

    if cur_tokens <= token_limit:
        return sessions, timestamps, cur_tokens, removed_count

    # Phase 3: randomly remove sessions between S_old and S_new.
    middle_indices = list(range(idx_old + 1, idx_new))
    if not middle_indices:
        return sessions, timestamps, cur_tokens, removed_count

    while cur_tokens > token_limit and middle_indices:
        drop_idx = random.choice(middle_indices)

        sessions.pop(drop_idx)
        if timestamps and drop_idx < len(timestamps):
            timestamps.pop(drop_idx)

        removed_count += 1
        idx_new -= 1

        middle_indices = list(range(idx_old + 1, idx_new))

        history_text = format_haystack(sessions, timestamps)
        cur_tokens = estimate_request_tokens(history_text, query_text, dim_key)

    return sessions, timestamps, cur_tokens, removed_count


async def fetch_eval_response(semaphore, uid, dim_key, sys_prompt, user_prompt):
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            start_time = time.perf_counter()
            try:
                request_kwargs = {
                    "model": EVAL_MODEL,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                if MAX_OUTPUT_TOKENS is not None:
                    request_kwargs["max_tokens"] = MAX_OUTPUT_TOKENS

                response = await aclient.chat.completions.create(**request_kwargs)

                elapsed = time.perf_counter() - start_time
                answer = response.choices[0].message.content or ""

                usage_obj = getattr(response, "usage", None)
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None) if usage_obj else None,
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None) if usage_obj else None,
                    "total_tokens": getattr(usage_obj, "total_tokens", None) if usage_obj else None,
                }

                return uid, dim_key, answer, elapsed, usage

            except Exception as e:
                print(f"[Retry {attempt + 1}/{MAX_RETRIES}] uid={uid} dim={dim_key}: {e}")
                await asyncio.sleep(2 ** attempt)

        return uid, dim_key, "ERROR: Failed after multiple retries.", None, {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }


async def run_async_evaluation(icds_path: str, output_path: str, enable_trim: bool = False):
    if aclient is None:
        raise RuntimeError("Evaluation client is not configured. Set the provider API key or pass --provider.")

    print("Loading dataset...")
    with open(icds_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []

    print("Building async tasks...")
    for item in dataset:
        uid = item["uid"]
        queries = item["probing_queries"]
        timestamps = item.get("timestamps", [])
        full_history_text = format_haystack(item["haystack_session"], timestamps)

        for dim_key in ["dim1_query", "dim2_query", "dim3_query"]:
            query_text = queries.get(dim_key)
            if not query_text:
                continue

            request_tokens = None
            removed_count = 0
            if enable_trim:
                trimmed_sessions, trimmed_timestamps, request_tokens, removed_count = (
                    trim_middle_noise_to_token_limit(item, dim_key, query_text)
                )
                history_text = format_haystack(trimmed_sessions, trimmed_timestamps)
            else:
                history_text = full_history_text

            sys_prompt, user_prompt = build_prompts(history_text, query_text, dim_key)

            if enable_trim and request_tokens > INPUT_TOKEN_LIMIT:
                print(
                    f"[WARN] uid={uid} dim={dim_key} still exceeds input limit "
                    f"after trimming: {request_tokens} > {INPUT_TOKEN_LIMIT}"
                )
            elif enable_trim and removed_count > 0:
                print(
                    f"[Trimmed] uid={uid} dim={dim_key}, removed={removed_count}, "
                    f"request_tokens={request_tokens}"
                )

            tasks.append(fetch_eval_response(semaphore, uid, dim_key, sys_prompt, user_prompt))

    print(f"Firing {len(tasks)} requests with concurrency limit of {CONCURRENCY_LIMIT}...")
    results = await tqdm.gather(*tasks, desc="Evaluating API calls")

    print("\nMerging results back into JSON...")
    results_dict = {}
    for uid, dim_key, answer, elapsed, usage in results:
        if uid not in results_dict:
            results_dict[uid] = {}

        results_dict[uid][dim_key] = {
            "answer": answer,
            "elapsed_seconds": elapsed,
            "usage": usage,
        }

    answer_collection = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_elapsed = 0.0
    counted_calls = 0

    for item in dataset:
        current_uid = item["uid"]
        uid_result = results_dict.get(current_uid, {})

        dim1_info = uid_result.get("dim1_query", {})
        dim2_info = uid_result.get("dim2_query", {})
        dim3_info = uid_result.get("dim3_query", {})

        a_answer = {
            "uid": current_uid,
            "target_model": EVAL_MODEL,
            "target_model_responses": {
                "dim1_response": dim1_info.get("answer", ""),
                "dim2_response": dim2_info.get("answer", ""),
                "dim3_response": dim3_info.get("answer", ""),
            },
            "target_model_meta": {
                "dim1_meta": {
                    "elapsed_seconds": dim1_info.get("elapsed_seconds"),
                    "usage": dim1_info.get("usage", {}),
                },
                "dim2_meta": {
                    "elapsed_seconds": dim2_info.get("elapsed_seconds"),
                    "usage": dim2_info.get("usage", {}),
                },
                "dim3_meta": {
                    "elapsed_seconds": dim3_info.get("elapsed_seconds"),
                    "usage": dim3_info.get("usage", {}),
                },
            },
        }
        answer_collection.append(a_answer)

        for dim_info in [dim1_info, dim2_info, dim3_info]:
            if not dim_info:
                continue

            elapsed = dim_info.get("elapsed_seconds")
            usage = dim_info.get("usage", {})

            if elapsed is not None:
                total_elapsed += elapsed
                counted_calls += 1

            if usage.get("prompt_tokens") is not None:
                total_prompt_tokens += usage["prompt_tokens"]
            if usage.get("completion_tokens") is not None:
                total_completion_tokens += usage["completion_tokens"]
            if usage.get("total_tokens") is not None:
                total_tokens += usage["total_tokens"]

    summary = {
        "target_model": EVAL_MODEL,
        "num_items": len(dataset),
        "num_calls": counted_calls,
        "total_elapsed_seconds": total_elapsed,
        "avg_elapsed_seconds": (total_elapsed / counted_calls) if counted_calls else None,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_prompt_tokens": (total_prompt_tokens / counted_calls) if counted_calls else None,
        "avg_completion_tokens": (total_completion_tokens / counted_calls) if counted_calls else None,
        "avg_total_tokens": (total_tokens / counted_calls) if counted_calls else None,
        "trim": {
            "enabled": enable_trim,
            "tokenizer_model": EVAL_MODEL if enable_trim else None,
            "encoding": getattr(ENCODER, "name", None) if enable_trim and ENCODER is not None else None,
            "token_limit": TOKEN_LIMIT if enable_trim else None,
            "reserved_output_tokens": RESERVED_OUTPUT_TOKENS if enable_trim else None,
            "safety_margin": SAFETY_MARGIN if enable_trim else None,
            "input_token_limit": INPUT_TOKEN_LIMIT if enable_trim else None,
        },
    }

    output_obj = {
        "summary": summary,
        "data": answer_collection,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)

    print(f"Evaluation complete. Results saved to {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run target-model responses for a STALE ICDS file.")
    parser.add_argument("--icds-path", required=True, help="Path to the generated MAIN/ICDS dataset JSON.")
    parser.add_argument("--output-path", required=True, help="Path to save target-model responses.")
    parser.add_argument("--model", default=EVAL_MODEL, help="Target model used to answer probing queries.")
    parser.add_argument("--provider", default=EVAL_PROVIDER, help="Provider name resolved by environment variables.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Useful for vLLM, for example http://127.0.0.1:8000/v1.")
    parser.add_argument("--api-key", help="API key for --base-url. vLLM often accepts a dummy value such as EMPTY.")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY_LIMIT, help="Concurrent target-model API calls.")
    parser.add_argument("--max-output-tokens", type=int, help="Optional max_tokens for target-model responses.")
    parser.add_argument("--enable-trim", action="store_true", help="Enable token-aware trimming before sending requests.")
    parser.add_argument("--token-limit", type=int, default=TOKEN_LIMIT, help="Total context token budget used when --enable-trim is set.")
    parser.add_argument("--reserved-output-tokens", type=int, default=RESERVED_OUTPUT_TOKENS, help="Output token reserve used when --enable-trim is set.")
    parser.add_argument("--safety-margin", type=int, default=SAFETY_MARGIN, help="Token safety margin used when --enable-trim is set.")
    return parser.parse_args()


def build_async_client(provider: str, base_url: str = None, api_key: str = None):
    if base_url:
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=api_key or "EMPTY", base_url=base_url)
    return get_Async_client(provider)


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    args = parse_args()
    EVAL_MODEL = args.model
    EVAL_PROVIDER = args.provider
    CONCURRENCY_LIMIT = args.concurrency
    MAX_OUTPUT_TOKENS = args.max_output_tokens
    ENABLE_TRIM = args.enable_trim
    TOKEN_LIMIT = args.token_limit
    RESERVED_OUTPUT_TOKENS = args.reserved_output_tokens
    SAFETY_MARGIN = args.safety_margin
    INPUT_TOKEN_LIMIT = TOKEN_LIMIT - RESERVED_OUTPUT_TOKENS - SAFETY_MARGIN
    aclient = build_async_client(EVAL_PROVIDER, base_url=args.base_url, api_key=args.api_key)
    ENCODER = get_encoder(EVAL_MODEL) if ENABLE_TRIM else None

    asyncio.run(run_async_evaluation(args.icds_path, args.output_path, enable_trim=ENABLE_TRIM))
