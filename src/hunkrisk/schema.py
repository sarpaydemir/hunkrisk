"""The output contract: one JSON object per hunk, validated exactly as the release bar defines."""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

CATEGORIES = ("auth", "money", "schema", "data", "secrets")
ALLOWED_CATEGORIES = (*CATEGORIES, "none")

Risk = Literal["high", "low"]

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class InvalidOutput(ValueError):
    """The text isn't a valid label. The message says which rule it broke."""


@dataclass(frozen=True)
class Label:
    risk: Risk
    categories: tuple[str, ...]
    reason: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self) | {"categories": list(self.categories)}


def parse_label(text: str, *, strict: bool = True) -> Label:
    """Parse and validate model output.

    strict: the whole output must be the JSON object; this is what the release bar measures.
    Otherwise the first {...} span is used, which suits API teachers that add a code fence.
    """
    if not strict:
        match = _JSON_OBJECT.search(text)
        text = match.group(0) if match else text
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidOutput(f"not JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise InvalidOutput("not a JSON object")
    return validate(obj)


def validate(obj: dict[str, Any]) -> Label:
    if set(obj) != {"risk", "categories", "reason"}:
        raise InvalidOutput(f"expected keys risk, categories, reason; got {sorted(obj)}")
    risk, categories = validate_decision(obj["risk"], obj["categories"])
    reason = obj["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise InvalidOutput("reason must be a non-empty string")
    return Label(risk, categories, reason.strip())


def validate_decision(risk: Any, categories: Any) -> tuple[Risk, tuple[str, ...]]:
    """Check risk and categories together; shared by model outputs and human gold labels."""
    if risk not in ("high", "low"):
        raise InvalidOutput(f"risk must be high or low, got {risk!r}")
    if not isinstance(categories, list) or not categories:
        raise InvalidOutput("categories must be a non-empty list")
    if unknown := [c for c in categories if c not in ALLOWED_CATEGORIES]:
        raise InvalidOutput(f"unknown categories {unknown}")
    if len(set(categories)) != len(categories):
        raise InvalidOutput("repeated category")
    if risk == "low" and categories != ["none"]:
        raise InvalidOutput('low risk must have categories ["none"]')
    if risk == "high" and "none" in categories:
        raise InvalidOutput("high risk can't include none")
    return risk, tuple(c for c in ALLOWED_CATEGORIES if c in categories)  # canonical order
