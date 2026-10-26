import json

import pytest

from hunkrisk.gold import GoldStore
from hunkrisk.schema import InvalidOutput


@pytest.fixture
def hunks(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = [
        {"id": f"h{i}", "repo": "o/r", "stratum": "targeted", "subject": "Fix auth bypass",
         "file": f"f{i}.py", "hunk": "@@ -1 +1 @@\n-a\n+b"}
        for i in range(5)
    ]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_items_expose_only_what_the_model_sees(hunks, tmp_path):
    store = GoldStore(hunks, tmp_path / "gold.jsonl")
    assert all(set(item) == {"id", "file", "hunk"} for item in store.items)


def test_order_is_shuffled_but_stable(hunks, tmp_path):
    first = [i["id"] for i in GoldStore(hunks, tmp_path / "gold.jsonl").items]
    second = [i["id"] for i in GoldStore(hunks, tmp_path / "gold.jsonl").items]
    assert first == second
    assert sorted(first) == [f"h{i}" for i in range(5)]


def test_saved_labels_persist_and_the_latest_wins(hunks, tmp_path):
    gold = tmp_path / "gold.jsonl"
    store = GoldStore(hunks, gold)
    store.save({"id": "h1", "risk": "low", "categories": ["none"]})
    store.save({"id": "h1", "risk": "high", "categories": ["money", "auth"], "seconds": 7.25})
    reloaded = GoldStore(hunks, gold)
    assert reloaded.labels["h1"]["label"] == {"risk": "high", "categories": ["auth", "money"]}
    assert reloaded.labels["h1"]["seconds"] == 7.2


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "nope", "risk": "low", "categories": ["none"]},
        {"id": "h1", "risk": "high", "categories": []},
        {"id": "h1", "risk": "low", "categories": ["auth"]},
    ],
)
def test_rejects_invalid_labels(hunks, tmp_path, payload):
    with pytest.raises(InvalidOutput):
        GoldStore(hunks, tmp_path / "gold.jsonl").save(payload)


def test_ids_filter_restricts_items_and_labels(hunks, tmp_path):
    gold = tmp_path / "gold.jsonl"
    GoldStore(hunks, gold).save({"id": "h3", "risk": "low", "categories": ["none"]})
    store = GoldStore(hunks, gold, ids={"h1", "h2"})
    assert sorted(i["id"] for i in store.items) == ["h1", "h2"]
    assert store.labels == {}
