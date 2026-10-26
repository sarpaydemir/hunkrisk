import json

import pytest

from hunkrisk.schema import InvalidOutput, Label, parse_label


def dump(risk="high", categories=("auth",), reason="Drops the ownership check."):
    return json.dumps({"risk": risk, "categories": list(categories), "reason": reason})


def test_parses_a_valid_label_into_canonical_category_order():
    label = parse_label(dump(categories=["data", "schema"]))
    assert label == Label("high", ("schema", "data"), "Drops the ownership check.")


def test_low_with_none_is_valid():
    assert parse_label(dump("low", ["none"], "Test-only change.")).risk == "low"


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("not json", "not JSON"),
        ("[1, 2]", "not a JSON object"),
        (dump(risk="medium"), "risk must be"),
        (dump(categories=[]), "non-empty list"),
        (dump(categories=["auth", "billing"]), "unknown categories"),
        (dump(categories=["auth", "auth"]), "repeated"),
        (dump("low", ["auth"]), "low risk"),
        (dump("high", ["auth", "none"]), "high risk"),
        (dump(reason="  "), "reason"),
        (json.dumps({"risk": "low", "categories": ["none"]}), "expected keys"),
    ],
)
def test_rejects_outputs_that_break_a_rule(text, rule):
    with pytest.raises(InvalidOutput, match=rule):
        parse_label(text)


def test_strict_mode_rejects_a_code_fence():
    fenced = f"```json\n{dump()}\n```"
    with pytest.raises(InvalidOutput):
        parse_label(fenced)
    assert parse_label(fenced, strict=False).risk == "high"
