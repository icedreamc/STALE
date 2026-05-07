from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR.parent / "STALE" / "outputs" / "demo_T1_MAIN.json"
DEFAULT_EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "runs"
DEFAULT_ENV_FILE = BASE_DIR.parent / "STALE" / ".env"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_key:
            continue
        if normalized_value.startswith(("'", '"')) and normalized_value.endswith(("'", '"')) and len(normalized_value) >= 2:
            normalized_value = normalized_value[1:-1]
        os.environ.setdefault(normalized_key, normalized_value)


def load_default_env_file() -> List[Path]:
    if not DEFAULT_ENV_FILE.exists():
        return []
    load_env_file(DEFAULT_ENV_FILE)
    return [DEFAULT_ENV_FILE.resolve()]


def get_first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def parse_optional_bool(value: str) -> bool | None:
    normalized = str(value or "").strip().lower()
    if not normalized or normalized == "auto":
        return None
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def normalize_base_url(value: str) -> str:
    normalized = str(value or "").strip()
    return normalized or DEFAULT_OPENAI_BASE_URL


def resolve_data_path(explicit_path: str = "") -> Path:
    candidate_text = str(explicit_path or "").strip()
    if candidate_text:
        return Path(candidate_text).resolve()
    return DEFAULT_DATA_PATH.resolve()


def resolve_embedding_model_path(explicit_path: str = "") -> Path:
    candidate_text = str(explicit_path or "").strip()
    if candidate_text:
        return Path(candidate_text).resolve()
    if DEFAULT_EMBEDDING_MODEL_PATH.exists():
        return DEFAULT_EMBEDDING_MODEL_PATH.resolve()
    raise ValueError(
        "embedding_model_path is required. Pass --embedding-model-path to the "
        "local sentence-embedding model directory.\n"
        f"Suggested local location: {DEFAULT_EMBEDDING_MODEL_PATH.resolve()}"
    )


def load_dataset_records(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError(f"Unsupported dataset format in {path}")


def build_llm_client(
    *,
    model: str,
    api_key: str,
    base_url: str,
    cache_dir: Path,
    wire_api: str = "",
    chat_supported: bool | None = None,
):
    from cup_mem import LLMClient, ResponsesLLMClient

    normalized_wire_api = str(wire_api or "").strip().lower()
    use_responses = normalized_wire_api == "responses" or chat_supported is False
    if use_responses:
        return ResponsesLLMClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            cache_dir=cache_dir,
        )
    return LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        cache_dir=cache_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CUP-Mem sample.")
    parser.add_argument("--data-path", type=str, default="", help="Dataset JSON path. Defaults to STALE/outputs/demo_T1_MAIN.json.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--session-mode", type=str, default="relevant_only", choices=["relevant_only", "full"])
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--embedding-model-path",
        type=str,
        default="",
        help="Local sentence-embedding model directory. Defaults to cup_mem/models/all-MiniLM-L6-v2 when present.",
    )
    parser.add_argument("--embedding-device", type=str, default="cpu")
    parser.add_argument("--wire-api", type=str, default="auto", choices=["auto", "chat", "responses"], help="Transport mode override.")
    parser.add_argument("--chat-supported", type=str, default="auto", choices=["auto", "true", "false"], help="Force chat support on or off.")
    parser.add_argument("--enable-debug-trace", action="store_true", help="Include auxiliary debug/trace-only fields in outputs.")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    loaded_env_files = load_default_env_file()
    model = get_first_env("TARGET_MODEL")
    api_key = get_first_env("OPENAI_API_KEY")
    base_url = normalize_base_url(get_first_env("OPENAI_BASE_URL"))
    wire_api = "" if args.wire_api == "auto" else args.wire_api
    chat_supported_value = "" if args.chat_supported == "auto" else args.chat_supported
    chat_supported = parse_optional_bool(chat_supported_value)
    if not model:
        raise ValueError("No model configured. Set TARGET_MODEL in STALE/.env.")
    if not api_key:
        raise ValueError("No API key configured. Set OPENAI_API_KEY in STALE/.env.")
    return {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "wire_api": wire_api,
        "chat_supported": chat_supported,
        "loaded_env_files": [str(path) for path in loaded_env_files],
    }


def validate_inputs(data_path: Path, embedding_model_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset JSON not found.\n"
            f"- requested: {data_path}\n"
            f"- expected STALE output: {DEFAULT_DATA_PATH.resolve()}"
        )
    if not embedding_model_path.exists():
        raise FileNotFoundError(
            "Embedding model directory not found.\n"
            f"- requested: {embedding_model_path}\n"
            f"- suggested local location: {DEFAULT_EMBEDDING_MODEL_PATH.resolve()}\n"
            "Pass --embedding-model-path to a local sentence-embedding model."
        )


def main() -> None:
    args = parse_args()

    from cup_mem import CupMemEngine, PipelineThresholds, TraceConfig
    from cup_mem.llm_layer.client import summarize_call_records

    runtime = build_runtime_config(args)
    data_path = resolve_data_path(args.data_path)
    output_root = Path(args.output_root).resolve()
    embedding_model_path = resolve_embedding_model_path(args.embedding_model_path)
    validate_inputs(data_path, embedding_model_path)

    data = load_dataset_records(data_path)
    sample_index = int(args.sample_index)
    if sample_index < 0 or sample_index >= len(data):
        raise IndexError(f"sample_index {sample_index} is out of range for dataset size {len(data)}")
    item = data[sample_index]

    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(run_dir)

    llm = build_llm_client(
        model=str(runtime["model"]),
        api_key=str(runtime["api_key"]),
        base_url=str(runtime["base_url"]),
        cache_dir=run_dir / ".cache",
        wire_api=str(runtime["wire_api"]),
        chat_supported=runtime["chat_supported"],
    )
    engine = CupMemEngine(
        llm=llm,
        embedding_model_path=str(embedding_model_path),
        embedding_device=str(args.embedding_device),
        thresholds=PipelineThresholds(),
        trace_config=TraceConfig(enable_debug_trace=bool(args.enable_debug_trace)),
    )
    result = engine.run_sample(
        item,
        sample_index=sample_index,
        session_mode=str(args.session_mode),
    )
    call_records = llm.get_call_records() if hasattr(llm, "get_call_records") else []
    usage_summary = summarize_call_records(call_records)
    summary = {
        "uid": item.get("uid"),
        "sample_index": sample_index,
        "session_mode": str(args.session_mode),
        "session_count": len(item.get("haystack_session", [])),
        "relevant_session_index": item.get("relevant_session_index"),
        "data_path": str(data_path),
        "output_dir": str(run_dir),
        "embedding_model_path": str(embedding_model_path),
        "model": runtime["model"],
        "wire_api": runtime["wire_api"] or "chat",
        "loaded_env_files": runtime["loaded_env_files"],
        "answers": {k: v.get("answer", {}).get("answer", "") for k, v in result.get("query_logs", {}).items()},
        "verdicts": {k: v.get("verdict", {}).get("status", "") for k, v in result.get("query_logs", {}).items()},
        "usage_summary": usage_summary,
    }

    (run_dir / "trace.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "llm_calls.json").write_text(json.dumps(call_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
