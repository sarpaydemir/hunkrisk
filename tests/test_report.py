import pytest

from hunkrisk.report import md_table, score, tune_threshold


def label(risk, categories=None):
    return {"label": {"risk": risk, "categories": categories or ["none"]}}


def test_threshold_is_the_highest_that_keeps_target_recall():
    reference = {f"h{i}": label("high", ["auth"]) for i in range(10)} | {"l0": label("low")}
    val = {f"h{i}": {"p_high": (i + 1) / 10} for i in range(10)} | {"l0": {"p_high": 0.95}}
    # 9 of 10 positives must clear it, so the threshold sits at the second-lowest positive.
    assert tune_threshold(val, reference) == pytest.approx(0.2)


def test_precision_at_base_rate_corrects_for_oversampling():
    # 2 strata: random has 1 high in 10 (base rate 10%); targeted is half high.
    gold = {f"r{i}": {"risk": "high" if i == 0 else "low", "categories": []} for i in range(10)}
    gold |= {f"t{i}": {"risk": "high" if i < 5 else "low", "categories": []} for i in range(10)}
    strata = {i: "random" if i.startswith("r") else "targeted" for i in gold}
    # A system that catches every high and false-alarms on 3 of the 14 lows.
    false_alarms = {"r1", "r2", "t5"}
    predicted = {
        i: {"risk": "high" if g["risk"] == "high" or i in false_alarms else "low", "categories": []}
        for i, g in gold.items()
    }
    m = score(gold, predicted, strata)
    assert m["recall"] == 1.0
    assert m["precision"] == pytest.approx(6 / 9)
    fpr = 3 / 14
    assert m["precision_natural"] == pytest.approx(0.1 / (0.1 + fpr * 0.9))


def test_md_table_renders_header_separator_and_rows():
    assert md_table(["a", "b"], [["1", "2"]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_threshold_only_counts_hunks_the_reference_labels():
    reference = {f"h{i}": label("high", ["auth"]) for i in range(10)}
    val = {f"h{i}": {"p_high": (i + 1) / 10} for i in range(10)} | {"x": {"p_high": 0.01}}
    # "x" has no reference label (a panel disagreement), so it can't drag the threshold down.
    assert tune_threshold(val, reference) == pytest.approx(0.2)
