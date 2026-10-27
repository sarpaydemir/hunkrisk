import json

from hunkrisk.panel import load_gold, majority, merge

HIGH_AUTH = {"risk": "high", "categories": ["auth"]}
HIGH_DATA = {"risk": "high", "categories": ["data", "money"]}
LOW = {"risk": "low", "categories": ["none"]}


def test_agreeing_judges_merge_with_the_union_of_categories():
    assert merge(HIGH_AUTH, HIGH_DATA) == {"risk": "high", "categories": ["auth", "money", "data"]}
    assert merge(LOW, LOW) == LOW


def test_disagreeing_judges_are_left_out():
    assert merge(HIGH_AUTH, LOW) is None


def test_a_third_judge_settles_a_disagreement_by_majority():
    assert majority([HIGH_AUTH, LOW, LOW]) == LOW
    # Only the judges that said high contribute categories.
    assert majority([HIGH_AUTH, LOW, HIGH_DATA]) == {
        "risk": "high",
        "categories": ["auth", "money", "data"],
    }


def test_a_missing_third_label_leaves_the_tie_unsettled():
    assert majority([HIGH_AUTH, LOW, None]) is None


def test_later_sources_override_earlier_ones(tmp_path):
    panel, tiebreak, human = (tmp_path / f"{n}.jsonl" for n in ("panel", "tiebreak", "human"))
    panel.write_text(json.dumps({"id": "a", "label": LOW}) + "\n")
    tiebreak.write_text(json.dumps({"id": "c", "label": LOW}) + "\n")
    human.write_text(
        json.dumps({"id": "a", "label": HIGH_AUTH}) + "\n" + json.dumps({"id": "b", "label": LOW})
    )
    gold = load_gold({"panel": panel, "tiebreak": tiebreak, "human": human})
    assert gold["a"] == HIGH_AUTH | {"source": "human"}
    assert gold["b"]["source"] == "human"
    assert gold["c"]["source"] == "tiebreak"
