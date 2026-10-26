"""Labels from decision models (e.g. TypeSafe Jev) through OpenRouter's Decisions API.

Decision models answer typed questions with probabilities instead of writing text, so the
labeling guide is mapped onto questions: one yes/no question for risk, using the whole guide
as instructions, and one per category, using that category's row of the guide.
"""

import hashlib
import json
import re
import threading
from functools import cache
from pathlib import Path
from typing import Any

import httpx

from hunkrisk.llm import PROVIDER_ROUTING, BudgetExceeded, _api_key
from hunkrisk.prompt import guide, user_message
from hunkrisk.schema import CATEGORIES

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
THRESHOLD = 0.5

_TABLE_ROW = re.compile(r"^\| `(\w+)` \| (.+?) \| (.*?) \|$", re.MULTILINE)


@cache
def category_definitions() -> dict[str, str]:
    """Each category's definition and examples, read from the guide's category table."""
    rows = {name: (what, examples) for name, what, examples in _TABLE_ROW.findall(guide())}
    return {
        c: f"`high` when the hunk changes {rows[c][0]}. Examples: {rows[c][1]}." for c in CATEGORIES
    }


@cache
def questions() -> dict[str, Any]:
    risk = {
        "type": "noul",
        "instructions": guide(),
        "criteria": {
            "true": "Risk is high: a reviewer must read this hunk closely before it ships.",
            "false": "Risk is low: no behavior change in auth, money, schema, data or secrets.",
        },
    }
    per_category = {
        category: {
            "type": "noul",
            "instructions": f"Does this hunk change behavior in the `{category}` area? {text}",
            "criteria": {
                "true": f"It affects {category}.",
                "false": f"It doesn't affect {category}.",
            },
        }
        for category, text in category_definitions().items()
    }
    return {"high_risk": risk, **per_category}


def to_label(answers: dict[str, Any]) -> dict[str, Any]:
    """Turn per-question probabilities into a label that satisfies the output schema."""
    p_high = float(answers["high_risk"]["noul"])
    p_categories = {c: float(answers[c]["noul"]) for c in CATEGORIES}
    if p_high < THRESHOLD:
        return {"risk": "low", "categories": ["none"], "reason": None}
    chosen = [c for c in CATEGORIES if p_categories[c] >= THRESHOLD]
    return {
        "risk": "high",
        "categories": chosen or [max(p_categories, key=p_categories.__getitem__)],
        "reason": None,  # decision models don't explain themselves
    }


class DecisionClient:
    def __init__(self, model: str, cache_dir: Path, max_usd: float) -> None:
        self.model = model
        self.cache_dir = cache_dir / model.replace("/", "__")
        self.max_usd = max_usd
        self.spent = 0.0
        self._lock = threading.Lock()
        self._http = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {_api_key()}"})

    def decide(self, file: str, hunk: str) -> dict[str, Any]:
        body = {"model": self.model, "state": user_message(file, hunk), "questions": questions()}
        request = body | {"provider": PROVIDER_ROUTING}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text()) | {"cached": True}

        with self._lock:
            if self.spent >= self.max_usd:
                raise BudgetExceeded(f"spent ${self.spent:.4f} of the ${self.max_usd} cap")
        for attempt in range(4):
            response = self._http.post(ENDPOINT, json=request)
            if response.status_code not in (429, 500, 502, 503, 524, 529):
                break
            threading.Event().wait(2**attempt)
        response.raise_for_status()
        data = response.json()
        result = {
            "resolved_model": data["model"],
            "answers": data["answers"],
            "prompt_tokens": data["usage"]["input_tokens"],
            "completion_tokens": data["usage"]["output_tokens"],
            "cost": float(data["usage"].get("cost") or 0.0),
        }
        with self._lock:
            self.spent += result["cost"]
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(result))
        partial.replace(path)
        return result | {"cached": False}
