from collections import Counter

from hunkrisk.label import balanced_sample


def records(repo, n):
    return [{"id": f"{repo}-{i}", "repo": repo} for i in range(n)]


def test_balanced_sample_spreads_across_repos():
    pool = records("a/big", 40) + records("b/small", 3) + records("c/mid", 10)
    picked = balanced_sample(pool, 9, seed=1)
    assert Counter(r["repo"] for r in picked) == {"a/big": 3, "b/small": 3, "c/mid": 3}


def test_balanced_sample_is_reproducible_and_unique():
    pool = records("a/x", 20) + records("b/y", 20)
    first = balanced_sample(pool, 15, seed=7)
    assert first == balanced_sample(pool, 15, seed=7)
    assert len({r["id"] for r in first}) == 15


def test_decision_questions_cover_risk_and_every_category():
    from hunkrisk.decisions import category_definitions, questions
    from hunkrisk.schema import CATEGORIES

    assert set(questions()) == {"high_risk", *CATEGORIES}
    assert all(q["type"] == "noul" for q in questions().values())
    assert "persisted data structure" in category_definitions()["schema"]


def test_decision_probabilities_become_a_valid_label():
    from hunkrisk.decisions import to_label
    from hunkrisk.schema import validate_decision

    def answers(p_high, **p):
        cats = {c: {"noul": p.get(c, 0.1)} for c in ("auth", "money", "schema", "data", "secrets")}
        return {"high_risk": {"noul": p_high}, **cats}

    assert to_label(answers(0.2, auth=0.9))["categories"] == ["none"]
    assert to_label(answers(0.8, auth=0.9, data=0.6))["categories"] == ["auth", "data"]
    weak = to_label(answers(0.8, money=0.3, auth=0.2))  # high, but no category clears 0.5
    assert weak["categories"] == ["money"]
    for label in (weak, to_label(answers(0.1))):
        validate_decision(label["risk"], label["categories"])
