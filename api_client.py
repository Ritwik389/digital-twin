"""
api_client.py
Centralized Gemini API client with round-robin key rotation.
Reads GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3 from .env.local.
Falls back to a single GOOGLE_API_KEY if numbered keys are absent.
On 429 / ResourceExhausted, automatically rotates to the next key and retries.

This is the ONLY place the Gemini API is called in the whole project —
embeddings are computed locally and offline (see chroma_utils.py), so
gemini_generate() is invoked exactly once per chat turn, for the final
answer.
"""

import os
import re
import json
import time
import threading
from dotenv import load_dotenv
from google import genai

load_dotenv(".env.local")


def _collect_keys() -> list[str]:
    """Gather all available API keys from environment."""
    keys = []
    for i in range(1, 4):
        k = os.environ.get(f"GOOGLE_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    if not keys:
        fallback = os.environ.get("GOOGLE_API_KEY", "").strip()
        if fallback:
            keys.append(fallback)
    if not keys:
        raise RuntimeError(
            "No API keys found. Set GOOGLE_API_KEY or GOOGLE_API_KEY_1/2/3 in .env.local"
        )
    return keys


API_KEYS = _collect_keys()
_lock = threading.Lock()
_current_index = 0
_clients: dict[str, genai.Client] = {}

MODEL = "models/gemini-2.5-flash"
MAX_RETRIES = len(API_KEYS)  # try each key once


def _get_client(key: str) -> genai.Client:
    """Cache one genai.Client per key."""
    if key not in _clients:
        _clients[key] = genai.Client(api_key=key)
    return _clients[key]


def _next_key() -> tuple[str, genai.Client]:
    """Round-robin: return (key, client) and advance the pointer."""
    global _current_index
    with _lock:
        key = API_KEYS[_current_index % len(API_KEYS)]
        _current_index += 1
    return key, _get_client(key)


def _is_quota_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit / quota-exhausted error."""
    msg = str(exc).lower()
    return any(tok in msg for tok in [
        "429", "resource exhausted", "quota", "rate limit",
        "too many requests", "resourceexhausted",
    ])


# ── Public helpers ────────────────────────────────────────────────────────────

def gemini_generate(prompt: str, model: str = MODEL) -> str:
    """
    Generate text with Gemini. Rotates API keys on quota errors.
    Returns the stripped text response.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        key, client = _next_key()
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            last_exc = e
            if _is_quota_error(e):
                print(f"[api] key ...{key[-6:]} quota hit, rotating (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(1)
                continue
            raise
    raise RuntimeError(f"All {MAX_RETRIES} API keys exhausted: {last_exc}")


def parse_json(text: str) -> dict | list | None:
    """Strip markdown fences and parse JSON."""
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def get_key_count() -> int:
    """Return how many API keys are loaded."""
    return len(API_KEYS)
