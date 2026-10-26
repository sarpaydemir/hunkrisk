"""Typed, validated access to the YAML configs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

Split = Literal["train", "val", "test"]

# Collection order. Test goes first so that a hunk also present in a train repo stays in test.
SPLITS: tuple[Split, ...] = ("test", "val", "train")

PERMISSIVE_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}


@dataclass(frozen=True)
class Repo:
    repo: str
    stack: str
    license: str
    split: Split


@dataclass(frozen=True)
class CollectConfig:
    seed: int
    since: str
    until: str
    hunks_per_repo: dict[str, int]
    targeted_share: float
    targeted_paths_per_category: int
    max_hunks_per_commit: int
    max_files_per_commit: int
    max_tokens: int
    tokenizer: str


def load_repos(path: Path) -> list[Repo]:
    repos = [Repo(**entry) for entry in yaml.safe_load(path.read_text())["repos"]]
    names = [r.repo for r in repos]
    if duplicates := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"repos listed more than once: {duplicates}")
    for r in repos:
        if r.split not in SPLITS:
            raise ValueError(f"{r.repo}: unknown split {r.split!r}")
        if r.license not in PERMISSIVE_LICENSES:
            raise ValueError(f"{r.repo}: license {r.license!r} is not permissive")
    return repos


def load_collect_config(path: Path) -> CollectConfig:
    return CollectConfig(**yaml.safe_load(path.read_text()))
