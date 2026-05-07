import os
from pathlib import Path

from clients import get_Async_client, get_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("IC_DATA_DIR", PROJECT_ROOT / "data"))

M_OLD_GEN_MODEL = os.getenv("M_OLD_GEN_MODEL", "qwen3.5-plus")
M_NEW_GEN_MODEL = os.getenv("M_NEW_GEN_MODEL", "gpt-5.2")
GEN_EVAL_MODEL = os.getenv("GEN_EVAL_MODEL", "gpt-5.2")
QA_ANS_GEN_MODEL = os.getenv("QA_ANS_GEN_MODEL", "gpt-5.2")
MODEL_FAST = os.getenv("MODEL_FAST", "gemini-3.1-flash-lite-preview")
USER_SIDE_MODEL = os.getenv("USER_SIDE_MODEL", "gpt-5.2")
ASSISTANT_SIDE_MODEL = os.getenv("ASSISTANT_SIDE_MODEL", "gpt-5.1-chat-latest")

QWEN_PROVIDER = os.getenv("QWEN_PROVIDER", "QWEN")
GPT_PROVIDER = os.getenv("GPT_PROVIDER", "OPENAI")
ASYNC_PROVIDER = os.getenv("ASYNC_PROVIDER", GPT_PROVIDER)

qwen_client = get_client(QWEN_PROVIDER)
gpt_client = get_client(GPT_PROVIDER)
gpt_async_client = get_Async_client(ASYNC_PROVIDER)

NOISE_DATASET_DIR = os.getenv("NOISE_DATASET_DIR", str(DATA_DIR / "noise_sessions.json"))
ROOT_DATASET_DIR = os.getenv("ROOT_DATASET_DIR", str(PROJECT_ROOT / "outputs"))

TARGET_CONFLICTPAIR_AMOUNT_PER_SEED = int(os.getenv("TARGET_CONFLICTPAIR_AMOUNT_PER_SEED", "3"))
MAX_ROUND_RESPONSE = int(os.getenv("MAX_ROUND_RESPONSE", "10"))
MAX_ROUND_IN_SESSION = int(os.getenv("MAX_ROUND_IN_SESSION", "6"))
NOISE_SESSION_INSERT_CONCURRENCY_LIMIT = int(os.getenv("NOISE_SESSION_INSERT_CONCURRENCY_LIMIT", "10"))
HAYSTACK_SESSION_AMOUNT = int(os.getenv("HAYSTACK_SESSION_AMOUNT", "50"))
TASK_CONCURRENCY_LIMIT = int(os.getenv("TASK_CONCURRENCY_LIMIT", "5"))
