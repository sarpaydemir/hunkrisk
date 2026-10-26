from pathlib import Path

import pytest

from hunkrisk.config import SPLITS, load_collect_config, load_repos

CONFIGS = Path(__file__).parents[1] / "configs"
STACKS = {"laravel", "django", "fastapi", "go", "ts"}


def test_every_split_covers_every_stack():
    repos = load_repos(CONFIGS / "repos.yaml")
    for split in SPLITS:
        assert {r.stack for r in repos if r.split == split} == STACKS, split


def test_collect_config_has_a_quota_per_split():
    cfg = load_collect_config(CONFIGS / "collect.yaml")
    assert set(cfg.hunks_per_repo) == set(SPLITS)


def test_rejects_a_repo_listed_twice(tmp_path):
    entry = "{repo: a/b, stack: go, license: MIT, split: train}"
    (tmp_path / "repos.yaml").write_text(f"repos:\n  - {entry}\n  - {entry}\n")
    with pytest.raises(ValueError, match="more than once"):
        load_repos(tmp_path / "repos.yaml")


def test_rejects_a_non_permissive_license(tmp_path):
    (tmp_path / "repos.yaml").write_text(
        "repos:\n  - {repo: a/b, stack: go, license: AGPL-3.0, split: train}\n"
    )
    with pytest.raises(ValueError, match="not permissive"):
        load_repos(tmp_path / "repos.yaml")
