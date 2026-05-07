from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .client import (
    _attach_llm_debug,
    _truncate_excerpt,
    build_json_retry_messages,
    extract_json_block,
    normalize_usage,
    parse_json_object_with_repair,
    summarize_call_records,
)


def _normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if str(part).strip())
    return str(content)


def _messages_to_responses_payload(messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    instructions: List[str] = []
    input_items: List[Dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        content = _normalize_message_content(message.get("content", "")).strip()
        if not content:
            continue

        if role == "system":
            instructions.append(content)
            continue

        input_items.append(
            {
                "role": role if role in {"user", "assistant", "developer"} else "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": content,
                    }
                ],
            }
        )

    payload: Dict[str, Any] = {
        "model": model,
        "input": input_items if input_items else "",
    }
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    return payload


def _parse_sse_events(raw_text: str) -> List[Tuple[str, str]]:
    events: List[Tuple[str, str]] = []
    current_event = "message"
    data_lines: List[str] = []

    for line in raw_text.splitlines():
        stripped = line.rstrip("\n")
        if not stripped:
            if data_lines:
                events.append((current_event, "\n".join(data_lines)))
            current_event = "message"
            data_lines = []
            continue

        if stripped.startswith("event:"):
            current_event = stripped[len("event:") :].strip() or "message"
        elif stripped.startswith("data:"):
            data_lines.append(stripped[len("data:") :].strip())

    if data_lines:
        events.append((current_event, "\n".join(data_lines)))
    return events


def _extract_text_from_response_obj(obj: Any) -> str:
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    if isinstance(obj, list):
        parts = [_extract_text_from_response_obj(item) for item in obj]
        return "".join(part for part in parts if part)

    if not isinstance(obj, dict):
        return ""

    if isinstance(obj.get("output_text"), str) and obj.get("output_text"):
        return obj["output_text"]

    if "response" in obj:
        nested = _extract_text_from_response_obj(obj["response"])
        if nested:
            return nested

    output = obj.get("output")
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
        if parts:
            return "".join(parts)

    content = obj.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "".join(parts)

    for key in ("text", "delta"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value

    return ""


def extract_text_from_responses_body(body: str) -> str:
    body = (body or "").strip()
    if not body:
        raise ValueError("Empty responses body")

    try:
        payload = json.loads(body)
        text = _extract_text_from_response_obj(payload)
        if text:
            return text
    except Exception:
        pass

    events = _parse_sse_events(body)
    if not events:
        return body

    delta_parts: List[str] = []
    done_texts: List[str] = []
    completed_payloads: List[Any] = []

    for event_name, data_text in events:
        if data_text == "[DONE]":
            continue
        parsed: Any = data_text
        try:
            parsed = json.loads(data_text)
        except Exception:
            pass

        if event_name.endswith("output_text.delta"):
            delta = _extract_text_from_response_obj(parsed)
            if delta:
                delta_parts.append(delta)
                continue

        if event_name.endswith("output_text.done"):
            done_text = _extract_text_from_response_obj(parsed)
            if done_text:
                done_texts.append(done_text)
                continue

        if event_name.endswith("completed") or event_name == "response.completed":
            completed_payloads.append(parsed)

    for payload in reversed(completed_payloads):
        text = _extract_text_from_response_obj(payload)
        if text:
            return text.strip()

    if done_texts:
        return done_texts[-1].strip()

    if delta_parts:
        return "".join(delta_parts).strip()

    last_data = events[-1][1]
    return last_data.strip()


def extract_usage_from_responses_body(body: str) -> Dict[str, Any]:
    body = (body or "").strip()
    if not body:
        return normalize_usage(None)

    try:
        payload = json.loads(body)
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if usage is not None:
            return normalize_usage(usage)
    except Exception:
        pass

    events = _parse_sse_events(body)
    if not events:
        return normalize_usage(None)

    for event_name, data_text in reversed(events):
        if data_text == "[DONE]":
            continue
        parsed: Any = data_text
        try:
            parsed = json.loads(data_text)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        usage = parsed.get("usage")
        if usage is not None:
            return normalize_usage(usage)
        if event_name.endswith("completed") or event_name == "response.completed":
            nested = parsed.get("response")
            if isinstance(nested, dict) and nested.get("usage") is not None:
                return normalize_usage(nested.get("usage"))

    return normalize_usage(None)


class ResponsesLLMClient:
    """Adapter for /responses-compatible gateways."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        cache_dir: Optional[Path] = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.timeout = timeout
        self._call_records: List[Dict[str, Any]] = []
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def reset_usage_tracking(self) -> None:
        self._call_records = []

    def get_call_records(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._call_records)

    def get_usage_summary(self) -> Dict[str, Any]:
        return summarize_call_records(self._call_records)

    def _cache_path(
        self,
        messages: List[Dict[str, str]],
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            json.dumps(
                {
                    "adapter": "responses",
                    "model": self.model,
                    "messages": messages,
                    "extra_payload": extra_payload or {},
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _post_responses(self, payload: Dict[str, Any]) -> str:
        url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.post(url, json=payload)
        response.raise_for_status()
        return response.text

    def _record_call(
        self,
        *,
        messages: List[Dict[str, str]],
        text: str,
        usage: Dict[str, Any],
        cache_hit: bool,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        billed_usage = {
            "prompt_tokens": 0 if usage.get("prompt_tokens") is not None and cache_hit else usage.get("prompt_tokens"),
            "completion_tokens": 0 if usage.get("completion_tokens") is not None and cache_hit else usage.get("completion_tokens"),
            "total_tokens": 0 if usage.get("total_tokens") is not None and cache_hit else usage.get("total_tokens"),
        }
        record: Dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "provider": "responses",
            "model": self.model,
            "cache_hit": cache_hit,
            "message_count": len(messages),
            "response_chars": len(text or ""),
            "usage": usage,
            "billed_usage": billed_usage,
        }
        if extra_meta:
            record.update(extra_meta)
        self._call_records.append(record)

    def call_text(
        self,
        messages: List[Dict[str, str]],
        *,
        max_retries: int = 3,
        extra_payload: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        cache_path = self._cache_path(messages, extra_payload=extra_payload)
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

        payload = _messages_to_responses_payload(messages, self.model)
        if extra_payload:
            payload.update(extra_payload)
        last_error: Exception | None = None
        last_raw_excerpt = ""
        for attempt in range(max_retries):
            try:
                raw_body = self._post_responses(payload)
                text = extract_text_from_responses_body(raw_body)
                if not str(text).strip():
                    raise ValueError("Empty LLM response")
                usage = extract_usage_from_responses_body(raw_body)
                if cache_path is not None:
                    cache_path.write_text(
                        json.dumps({"text": text, "usage": usage, "payload": payload}, ensure_ascii=False, indent=2),
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
            except Exception as exc:  # pragma: no cover - runtime/network path
                last_error = exc
                raw_excerpt = _truncate_excerpt(locals().get("raw_body", ""))
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
        raw = self.call_text(
            messages,
            max_retries=max_retries,
            extra_payload={
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": json_schema,
                        "strict": True,
                    }
                }
            },
            extra_meta=extra_meta,
        )
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got: {type(data).__name__}")
        return data

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
            extra_payload=extra_request_kwargs,
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
                    extra_payload=extra_request_kwargs,
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
