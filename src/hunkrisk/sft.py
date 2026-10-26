"""Turn labeled hunks into tokenized supervised examples. Loss is computed on the answer only."""

import json
import random
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from hunkrisk.prompt import messages, short_messages

ANSWER_PREFIX = '{"risk": "'


def answer_text(label: dict[str, Any]) -> str:
    """The exact target string; key order puts `risk` first so P(high) is one token away."""
    return json.dumps(
        {"risk": label["risk"], "categories": label["categories"], "reason": label["reason"]},
        ensure_ascii=False,
    )


def prompt_text(tokenizer: PreTrainedTokenizerBase, file: str, hunk: str, full_guide: bool) -> str:
    convo = messages(file, hunk) if full_guide else short_messages(file, hunk)
    return tokenizer.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def load_labeled(hunks_path: Path, labels_path: Path) -> list[dict[str, Any]]:
    """Hunks joined with their labels; hunks whose label was invalid are dropped."""
    labels = {r["id"]: r["label"] for r in map(json.loads, labels_path.open())}
    return [
        r | {"label": labels[r["id"]]}
        for r in map(json.loads, hunks_path.open())
        if labels.get(r["id"])
    ]


def rebalance(records: list[dict[str, Any]], high_share: float, seed: int) -> list[dict[str, Any]]:
    """Downsample `low` records until `high` makes up `high_share` of the set."""
    high = [r for r in records if r["label"]["risk"] == "high"]
    low = [r for r in records if r["label"]["risk"] == "low"]
    keep = min(len(low), round(len(high) * (1 - high_share) / high_share))
    kept = high + random.Random(seed).sample(low, keep)
    return sorted(kept, key=lambda r: r["id"])


def tokenize(
    records: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> list[dict[str, list[int]]]:
    examples = []
    for r in records:
        prompt = tokenizer(
            prompt_text(tokenizer, r["file"], r["hunk"], full_guide=False), add_special_tokens=False
        ).input_ids
        answer = tokenizer(
            answer_text(r["label"]) + tokenizer.eos_token, add_special_tokens=False
        ).input_ids
        if len(prompt) + len(answer) > max_length:
            continue
        examples.append({"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer})
    return examples


class Collator:
    """Right-pads a batch; padded label positions are ignored by the loss."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (width - len(values))

        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_token_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
        }
