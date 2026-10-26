"""Prompts, built from the labeling guide so the teacher and the human apply the same rules."""

import hashlib
from functools import cache
from pathlib import Path

GUIDE_PATH = Path(__file__).parents[2] / "docs" / "labeling-guide.md"

INSTRUCTION = (
    "You triage code changes for review. Classify the diff hunk the user sends by following "
    "the guide below. Reply with exactly one JSON object with the keys risk, categories and "
    "reason, and nothing else: no code fence, no commentary."
)


@cache
def guide() -> str:
    """The guide from its "## Task" section on; the preamble is about the file, not the task."""
    text = GUIDE_PATH.read_text()
    return text[text.index("## Task") :].strip()


@cache
def system_prompt() -> str:
    return f"{INSTRUCTION}\n\n{guide()}"


def user_message(file: str, hunk: str) -> str:
    return f"{file}\n{hunk}"


def messages(file: str, hunk: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_message(file, hunk)},
    ]


def prompt_version() -> str:
    """Short hash of the system prompt, stored with every label for provenance."""
    return hashlib.sha256(system_prompt().encode()).hexdigest()[:12]
