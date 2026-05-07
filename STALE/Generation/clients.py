import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _env_prefix(provider: str) -> str:
    return provider.upper().replace("-", "_")


def _client_kwargs(provider: str) -> Optional[dict]:
    prefix = _env_prefix(provider)
    api_key = os.getenv(f"{prefix}_API_KEY")
    base_url = os.getenv(f"{prefix}_BASE_URL")

    if not api_key and prefix != "OPENAI":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        return None

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def get_client(provider: str = "OPENAI"):
    kwargs = _client_kwargs(provider)
    if kwargs is None:
        return None
    from openai import OpenAI

    return OpenAI(**kwargs)


def get_async_client(provider: str = "OPENAI"):
    kwargs = _client_kwargs(provider)
    if kwargs is None:
        return None
    from openai import AsyncOpenAI

    return AsyncOpenAI(**kwargs)


get_Async_client = get_async_client
