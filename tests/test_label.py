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
