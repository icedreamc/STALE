from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


def extract_json_block(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    if text.startswith("```"):
        start = text.find("\n")
        end = text.rfind("```")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end].strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            return json.loads(candidate)

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

    return json.loads(text)


def _iter_json_candidates(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    candidates: List[str] = []

    def _push(candidate: str) -> None:
        normalized = str(candidate or "").strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    if text.startswith("```"):
        start = text.find("\n")
        end = text.rfind("```")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end].strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            _push(candidate)

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            _push(text[start : end + 1])

    _push(text)
    return candidates


def _repair_common_json_glitches(candidate: str) -> str:
    repaired = str(candidate or "").strip()
    if not repaired:
        return repaired

    repaired = re.sub(r";(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def parse_json_object_with_repair(text: str) -> tuple[Dict[str, Any], str]:
    last_error: Exception | None = None
    for candidate in _iter_json_candidates(text):
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got: {type(data).__name__}")
            return data, "raw"
        except Exception as exc:
            last_error = exc

        repaired = _repair_common_json_glitches(candidate)
        if repaired == candidate:
            continue
        try:
            data = json.loads(repaired)
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got: {type(data).__name__}")
            return data, "repaired"
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("Empty LLM response")


def build_json_retry_messages(messages: List[Dict[str, str]], raw_text: str) -> List[Dict[str, str]]:
    retry_messages = copy.deepcopy(messages)
    invalid_json = _truncate_excerpt(raw_text, limit=1600)
    retry_messages.append({"role": "assistant", "content": raw_text})
    retry_messages.append(
        {
            "role": "user",
            "content": (
                "Your previous answer was invalid JSON. "
                "Please rewrite the same answer as one valid JSON object only. "
                "Preserve the intended fields and values when possible. "
                "Do not add markdown fences, comments, or trailing punctuation after the closing brace.\n\n"
                f"Previous invalid JSON:\n{invalid_json}"
            ),
        }
    )
    return retry_messages


def _truncate_excerpt(text: Any, *, limit: int = 400) -> str:
    excerpt = str(text or "").strip()
    if not excerpt:
        return ""
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[:limit] + "...<truncated>"


def _attach_llm_debug(exc: BaseException, *, raw_response_excerpt: str = "") -> None:
    if raw_response_excerpt and not getattr(exc, "raw_response_excerpt", None):
        setattr(exc, "raw_response_excerpt", raw_response_excerpt)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _jsonable_model_dump(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except Exception:
            try:
                dumped = value.model_dump()
            except Exception:
                dumped = None
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return None


def normalize_usage(usage: Any) -> Dict[str, Any]:
    raw = _jsonable_model_dump(usage) or {}
    prompt_tokens = raw.get("prompt_tokens")
    completion_tokens = raw.get("completion_tokens")
    total_tokens = raw.get("total_tokens")

    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "completion_tokens", None)
    if total_tokens is None:
        total_tokens = getattr(usage, "total_tokens", None)

    normalized: Dict[str, Any] = {
        "prompt_tokens": _safe_int(prompt_tokens),
        "completion_tokens": _safe_int(completion_tokens),
        "total_tokens": _safe_int(total_tokens),
    }
    for key, value in raw.items():
        if key not in normalized:
            normalized[key] = value
    return normalized


def _zero_usage_like(usage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_tokens": 0 if usage.get("prompt_tokens") is not None else None,
        "completion_tokens": 0 if usage.get("completion_tokens") is not None else None,
        "total_tokens": 0 if usage.get("total_tokens") is not None else None,
    }


def _accumulate_usage(target: Dict[str, int], usage: Dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value


def _empty_usage_counter() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def summarize_call_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_calls": len(records),
        "cache_hits": 0,
        "cache_misses": 0,
        "logical_usage": _empty_usage_counter(),
        "billed_usage": _empty_usage_counter(),
        "by_phase": {},
        "by_query_label": {},
    }

    for record in records:
        cache_hit = bool(record.get("cache_hit", False))
        if cache_hit:
            summary["cache_hits"] += 1
        else:
            summary["cache_misses"] += 1

        logical_usage = record.get("usage") or {}
        billed_usage = record.get("billed_usage") or {}
        _accumulate_usage(summary["logical_usage"], logical_usage)
        _accumulate_usage(summary["billed_usage"], billed_usage)

        phase = str(record.get("phase", "")).strip() or "unknown"
        phase_bucket = summary["by_phase"].setdefault(
            phase,
            {
                "calls": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "logical_usage": _empty_usage_counter(),
                "billed_usage": _empty_usage_counter(),
            },
        )
        phase_bucket["calls"] += 1
        if cache_hit:
            phase_bucket["cache_hits"] += 1
        else:
            phase_bucket["cache_misses"] += 1
        _accumulate_usage(phase_bucket["logical_usage"], logical_usage)
        _accumulate_usage(phase_bucket["billed_usage"], billed_usage)

        query_label = str(record.get("query_label", "")).strip()
        if query_label:
            query_bucket = summary["by_query_label"].setdefault(
                query_label,
                {
                    "calls": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "logical_usage": _empty_usage_counter(),
                    "billed_usage": _empty_usage_counter(),
                },
            )
            query_bucket["calls"] += 1
            if cache_hit:
                query_bucket["cache_hits"] += 1
            else:
                query_bucket["cache_misses"] += 1
            _accumulate_usage(query_bucket["logical_usage"], logical_usage)
            _accumulate_usage(query_bucket["billed_usage"], billed_usage)

    return summary


def _count_message_chars(messages: List[Dict[str, str]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content", "")
        total += len(str(content or ""))
    return total


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        cache_dir: Optional[Path] = None,
        openai_cls=OpenAI,
    ):
        self.model = model
        self.client = openai_cls(api_key=api_key, base_url=base_url)
        self.cache_dir = cache_dir
        self._call_records: List[Dict[str, Any]] = []
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def reset_usage_tracking(self) -> None:
        self._call_records = []

    def get_call_records(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._call_records)

    def get_usage_summary(self) -> Dict[str, Any]:
        return summarize_call_records(self._call_records)

    def _cache_path(self, messages: List[Dict[str, str]]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            json.dumps(
                {
                    "model": self.model,
                    "messages": messages,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _record_call(
        self,
        *,
        messages: List[Dict[str, str]],
        text: str,
        usage: Dict[str, Any],
        cache_hit: bool,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "provider": "chat.completions",
            "model": self.model,
            "cache_hit": cache_hit,
            "message_count": len(messages),
            "message_chars": _count_message_chars(messages),
            "response_chars": len(text or ""),
            "usage": usage,
            "billed_usage": _zero_usage_like(usage) if cache_hit else usage,
        }
        if extra_meta:
            record.update(extra_meta)
        self._call_records.append(record)

    def call_text(
        self,
        messages: List[Dict[str, str]],
        *,
        max_retries: int = 3,
        extra_meta: Optional[Dict[str, Any]] = None,
        extra_request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        cache_path = None
        if not extra_request_kwargs:
            cache_path = self._cache_path(messages)
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            text = str(cached.get("text", ""))
            if text.strip():
                usage = normalize_usage(cached.get("usage"))
                self._record_call(
                    messages=messages,
                    text=text,
                    usage=usage,
                    cache_hit=True,
                    extra_meta=extra_meta,
                )
                return text
            try:
                cache_path.unlink()
            except Exception:
                pass

        last_error: Exception | None = None
        last_raw_excerpt = ""
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **(extra_request_kwargs or {}),
                )
                text = resp.choices[0].message.content or ""
                if not str(text).strip():
                    raise ValueError("Empty LLM response")
                usage = normalize_usage(getattr(resp, "usage", None))
                if cache_path is not None:
                    cache_path.write_text(
                        json.dumps(
                            {
                                "text": text,
                                "usage": usage,
                                "model": self.model,
                                "messages": messages,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                self._record_call(
                    messages=messages,
                    text=text,
                    usage=usage,
                    cache_hit=False,
                    extra_meta=extra_meta,
                )
                return text
            except Exception as exc:  # pragma: no cover - network/runtime path
                last_error = exc
                raw_excerpt = _truncate_excerpt(locals().get("text", ""))
                if raw_excerpt:
                    last_raw_excerpt = raw_excerpt
                if attempt == max_retries - 1:
                    _attach_llm_debug(
                        exc,
                        raw_response_excerpt=raw_excerpt or last_raw_excerpt,
                    )
                    raise
                time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        return ""

    def call_json(
        self,
        system_prompt: str,
        user_payload: str,
        *,
        max_retries: int = 3,
        extra_meta: Optional[Dict[str, Any]] = None,
        extra_request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
        raw = self.call_text(
            messages,
            max_retries=max_retries,
            extra_meta=extra_meta,
            extra_request_kwargs=extra_request_kwargs,
        )
        try:
            data, _parse_mode = parse_json_object_with_repair(raw)
            return data
        except Exception as exc:
            raw_excerpt = _truncate_excerpt(raw)

            retry_messages = build_json_retry_messages(messages, raw)
            try:
                retry_raw = self.call_text(
                    retry_messages,
                    max_retries=1,
                    extra_meta=extra_meta,
                    extra_request_kwargs=extra_request_kwargs,
                )
                data, _parse_mode = parse_json_object_with_repair(retry_raw)
                return data
            except Exception as retry_exc:
                retry_excerpt = _truncate_excerpt(locals().get("retry_raw", ""))
                _attach_llm_debug(
                    retry_exc,
                    raw_response_excerpt=retry_excerpt or raw_excerpt,
                )
                raise

    def call_json_with_schema(
        self,
        system_prompt: str,
        user_payload: str,
        *,
        json_schema: Dict[str, Any],
        schema_name: str = "structured_output",
        max_retries: int = 3,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
        try:
            raw = self.call_text(
                messages,
                max_retries=max_retries,
                extra_meta=extra_meta,
                extra_request_kwargs={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "schema": json_schema,
                            "strict": True,
                        },
                    }
                },
            )
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got: {type(data).__name__}")
            return data
        except Exception:
            fallback_system_prompt = (
                f"{system_prompt}\n\n"
                "Output JSON only. The JSON must conform to this schema:\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            )
            return self.call_json(
                fallback_system_prompt,
                user_payload,
                max_retries=max_retries,
                extra_meta=extra_meta,
            )
