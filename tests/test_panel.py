import json

from hunkrisk.panel import load_gold, merge

HIGH_AUTH = {"risk": "high", "categories": ["auth"]}
HIGH_DATA = {"risk": "high", "categories": ["data", "money"]}
LOW = {"risk": "low", "categories": ["none"]}


def test_agreeing_judges_merge_with_the_union_of_categories():
    assert merge(HIGH_AUTH, HIGH_DATA) == {"risk": "high", "categories": ["auth", "money", "data"]}
    assert merge(LOW, LOW) == LOW


def test_disagreeing_judges_need_a_human():
    assert merge(HIGH_AUTH, LOW) is None


def test_human_labels_override_the_panel(tmp_path):
    panel, human = tmp_path / "panel.jsonl", tmp_path / "human.jsonl"
    panel.write_text(json.dumps({"id": "a", "label": LOW}) + "\n")
    human.write_text(
        json.dumps({"id": "a", "label": HIGH_AUTH}) + "\n" + json.dumps({"id": "b", "label": LOW})
    )
    gold = load_gold(panel, human)
    assert gold["a"] == HIGH_AUTH | {"source": "human"}
    assert gold["b"]["source"] == "human"
