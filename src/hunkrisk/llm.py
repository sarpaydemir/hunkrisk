"""OpenRouter chat client with a response cache and a hard spend cap.

Cached responses cost nothing, so an interrupted labeling run resumes where it stopped and a
rerun never pays twice for the same prompt.
"""

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://openrouter.ai/api/v1"


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Price:
    prompt: float  # USD per token
    completion: float


def model_price(model: str) -> Price:
    response = httpx.get(f"{BASE_URL}/models", timeout=30)
    response.raise_for_status()
    for entry in response.json()["data"]:
        if entry["id"] == model:
            pricing = entry["pricing"]
            return Price(float(pricing["prompt"]), float(pricing["completion"]))
    raise ValueError(f"OpenRouter has no model {model!r}")


def _api_key() -> str:
    load_dotenv()
    return os.environ["OPENROUTER_API_KEY"]


def key_remaining_usd() -> float | None:
    """Remaining spend on the API key, or None when the key has no limit."""
    response = httpx.get(
        f"{BASE_URL}/key", headers={"Authorization": f"Bearer {_api_key()}"}, timeout=30
    )
    response.raise_for_status()
    return response.json()["data"].get("limit_remaining")


class Client:
    def __init__(self, model: str, cache_dir: Path, max_usd: float) -> None:
        self.model = model
        self.price = model_price(model)
        self.cache_dir = cache_dir / model.replace("/", "__")
        self.max_usd = max_usd
        self.spent = 0.0
        self._lock = threading.Lock()
        self._client = OpenAI(base_url=BASE_URL, api_key=_api_key(), max_retries=3, timeout=120)

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 300) -> dict[str, Any]:
        """Return the reply text with token counts and cost; `cached` marks free replays."""
        key = hashlib.sha256(json.dumps([self.model, messages, max_tokens]).encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text()) | {"cached": True}

        with self._lock:
            if self.spent >= self.max_usd:
                raise BudgetExceeded(f"spent ${self.spent:.4f} of the ${self.max_usd} cap")
        response = self._client.chat.completions.create(
            model=self.model,
            messages=self._with_prompt_caching(messages),  # type: ignore[arg-type]
            temperature=0,
            max_tokens=max_tokens,
            # Reasoning tokens are billed as output; a one-line JSON label doesn't need them.
            extra_body={"usage": {"include": True}, "reasoning": {"enabled": False}},
        )
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        reported = (usage.model_extra or {}).get("cost") if usage else None
        cost = (
            float(reported)
            if reported is not None
            else prompt_tokens * self.price.prompt + completion_tokens * self.price.completion
        )
        with self._lock:
            self.spent += cost

        result = {
            "text": response.choices[0].message.content or "",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(result))
        partial.replace(path)
        return result | {"cached": False}

    def _with_prompt_caching(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Mark the shared system prompt as cacheable for Anthropic models.

        Other providers on OpenRouter cache repeated prefixes automatically; Anthropic needs an
        explicit breakpoint. Cached reads cost a tenth of the input price.
        """
        if not self.model.startswith("anthropic/"):
            return list(messages)
        return [
            {
                "role": m["role"],
                "content": [
                    {"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}
                ],
            }
            if m["role"] == "system"
            else m
            for m in messages
        ]

    def estimate_usd(self, prompt_chars: int, n_requests: int, completion_tokens: int) -> float:
        """Upper-end estimate: ~3.2 characters per token for code, plus 15% headroom."""
        prompt_tokens = prompt_chars / 3.2
        total = prompt_tokens * self.price.prompt
        total += n_requests * completion_tokens * self.price.completion
        return total * 1.15
