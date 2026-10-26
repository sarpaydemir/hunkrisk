import json
from types import SimpleNamespace

from hunkrisk.sft import ANSWER_PREFIX, Collator, answer_text, rebalance, tokenize


class FakeTokenizer:
    """Whitespace tokenizer with a trivial chat template, enough to test masking."""

    eos_token = "<eos>"

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def apply_chat_template(self, convo, **_):
        return " ".join(m["content"] for m in convo) + " <assistant>"

    def __call__(self, text, add_special_tokens=False):
        ids = [self.vocab.setdefault(t, len(self.vocab) + 1) for t in text.split()]
        return SimpleNamespace(input_ids=ids)


LABEL = {"risk": "high", "categories": ["auth"], "reason": "Drops a check."}


def record(i, risk="low"):
    label = LABEL if risk == "high" else {"risk": "low", "categories": ["none"], "reason": "Safe."}
    return {"id": f"h{i}", "file": "a.py", "hunk": "@@ -1 +1 @@\n+x", "label": label}


def test_answer_starts_with_the_risk_field():
    assert answer_text(LABEL).startswith(ANSWER_PREFIX + "high")
    assert json.loads(answer_text(LABEL)) == LABEL


def test_loss_is_masked_on_the_prompt():
    [example] = tokenize([record(0, "high")], FakeTokenizer(), max_length=512)
    answer_start = example["labels"].index(next(t for t in example["labels"] if t != -100))
    assert all(t == -100 for t in example["labels"][:answer_start])
    assert example["labels"][answer_start:] == example["input_ids"][answer_start:]


def test_examples_over_the_length_limit_are_dropped():
    assert tokenize([record(0)], FakeTokenizer(), max_length=5) == []


def test_rebalance_reaches_the_requested_high_share():
    records = [record(i, "high") for i in range(10)] + [record(i + 10) for i in range(90)]
    kept = rebalance(records, high_share=0.25, seed=1)
    assert sum(r["label"]["risk"] == "high" for r in kept) == 10
    assert len(kept) == 40


def test_collator_pads_inputs_and_ignores_padding_in_the_loss():
    batch = Collator(pad_token_id=0)(
        [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [5]}]
    )
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [5, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
