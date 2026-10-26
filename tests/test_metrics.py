import pytest

from hunkrisk.compare import compare
from hunkrisk.metrics import category_f1, risk_metrics


def test_risk_metrics_treat_high_as_positive():
    reference = ["high", "high", "low", "low"]
    predicted = ["high", "low", "high", "low"]
    m = risk_metrics(reference, predicted)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["agreement"] == pytest.approx(0.5)
    assert m["kappa"] == pytest.approx(0.0)


def test_perfect_agreement_on_a_single_class_has_kappa_one():
    assert risk_metrics(["low", "low"], ["low", "low"])["kappa"] == 1.0


def test_category_f1_scores_each_category_independently():
    reference = [["auth"], ["auth", "money"], ["none"]]
    predicted = [["auth"], ["auth"], ["none"]]
    scores = category_f1(reference, predicted)
    assert scores["auth"] == pytest.approx(1.0)
    assert scores["money"] == pytest.approx(0.0)
    assert scores["schema"] is None


def test_compare_counts_invalid_candidate_outputs_separately():
    high = {"risk": "high", "categories": ["auth"], "reason": "r"}
    low = {"risk": "low", "categories": ["none"], "reason": "r"}
    reference = {"a": high, "b": low, "c": low}
    candidate = {"a": high, "b": None, "c": low}
    m = compare(reference, candidate)
    assert m["invalid"] == 1
    assert m["n"] == 2
    assert m["recall"] == pytest.approx(1.0)
