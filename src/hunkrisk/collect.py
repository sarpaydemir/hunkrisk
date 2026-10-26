"""Collect diff hunks from the source repositories into one JSONL file per split.

    uv run python -m hunkrisk.collect                       # full run
    uv run python -m hunkrisk.collect --scale 0.1 --only krayin/laravel-crm

Each repository has a hunk quota, filled from two strata:

- targeted: commits that touch risk-looking paths (see `sampling.py`)
- random:   commits drawn from random pages of the history

Randomness is seeded per repository and API responses are cached, so a rerun reproduces
the same dataset, and adding a repository doesn't reshuffle the others.
"""

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from hunkrisk.config import SPLITS, CollectConfig, Repo, load_collect_config, load_repos
from hunkrisk.github import GitHub, last_page
from hunkrisk.hunks import Hunk, dedupe_key, has_meaningful_change, is_excluded, render, split_patch
from hunkrisk.sampling import candidate_paths, interleave

BOT_MARKERS = ("[bot]", "dependabot", "renovate", "github-actions")
COMMITS_PER_PAGE = 10  # cap per history page, so random commits spread across time
FETCH_BATCH = 16


class Collector:
    def __init__(self, gh: GitHub, cfg: CollectConfig, tokenizer: PreTrainedTokenizerBase):
        self.gh = gh
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.seen: set[str] = set()  # dedupe keys across all repos and splits
        self.dropped: Counter[str] = Counter()

    def collect(self, repo: Repo, quota: int) -> list[dict[str, Any]]:
        rng = random.Random(f"{self.cfg.seed}:{repo.repo}")
        used: set[str] = set()
        records: list[dict[str, Any]] = []
        targeted_quota = round(quota * self.cfg.targeted_share)
        # The random stratum fills up to the full quota, which also covers a targeted shortfall.
        strata = [
            ("targeted", self._targeted_commits(repo.repo, rng), targeted_quota),
            ("random", self._random_commits(repo.repo, rng), quota),
        ]
        for stratum, shas, target in strata:
            if len(records) >= target:
                continue
            for commit in self._fetch(repo.repo, shas, used):
                for hunk, n_tokens in self._pick(commit, rng, limit=target - len(records)):
                    records.append(_record(repo, stratum, commit, hunk, n_tokens))
                if len(records) >= target:
                    break
        return records

    def _window(self) -> dict[str, str]:
        return {"since": self.cfg.since, "until": self.cfg.until}

    def _random_commits(self, repo: str, rng: random.Random) -> Iterator[str]:
        _, link = self.gh.get(f"/repos/{repo}/commits", per_page=100, page=1, **self._window())
        pages = list(range(1, last_page(link) + 1))
        rng.shuffle(pages)
        for page in pages:
            listing, _ = self.gh.get(
                f"/repos/{repo}/commits", per_page=100, page=page, **self._window()
            )
            shas = [c["sha"] for c in listing if _is_reviewable(c)]
            rng.shuffle(shas)
            yield from shas[:COMMITS_PER_PAGE]

    def _targeted_commits(self, repo: str, rng: random.Random) -> Iterator[str]:
        meta, _ = self.gh.get(f"/repos/{repo}")
        tree, _ = self.gh.get(f"/repos/{repo}/git/trees/{meta['default_branch']}", recursive=1)
        files = [
            e["path"] for e in tree["tree"] if e["type"] == "blob" and not is_excluded(e["path"])
        ]
        by_category: list[list[str]] = []
        for paths in candidate_paths(files).values():
            shas: list[str] = []
            for path in rng.sample(paths, min(len(paths), self.cfg.targeted_paths_per_category)):
                listing, _ = self.gh.get(
                    f"/repos/{repo}/commits", path=path, per_page=100, **self._window()
                )
                shas.extend(c["sha"] for c in listing if _is_reviewable(c))
            rng.shuffle(shas)
            by_category.append(shas)
        yield from interleave(by_category)

    def _fetch(self, repo: str, shas: Iterator[str], used: set[str]) -> Iterator[dict[str, Any]]:
        """Fetch commit details in parallel batches, yielding them in input order."""
        with ThreadPoolExecutor(FETCH_BATCH) as pool:
            while True:
                batch = []
                for sha in shas:
                    if sha not in used:
                        used.add(sha)
                        batch.append(sha)
                    if len(batch) == FETCH_BATCH:
                        break
                if not batch:
                    return
                yield from pool.map(
                    lambda sha: self.gh.get(f"/repos/{repo}/commits/{sha}")[0], batch
                )

    def _pick(
        self, commit: dict[str, Any], rng: random.Random, limit: int
    ) -> list[tuple[Hunk, int]]:
        files = commit.get("files") or []
        if len(files) > self.cfg.max_files_per_commit:
            self.dropped["commit_too_large"] += 1
            return []
        eligible: dict[str, tuple[Hunk, int]] = {}
        for file in files:
            if is_excluded(file["filename"]):
                self.dropped["excluded_file"] += 1
                continue
            if "patch" not in file:  # binary, or too large for the API to render
                self.dropped["no_patch"] += 1
                continue
            for hunk in split_patch(file["filename"], file["patch"]):
                key = dedupe_key(hunk)
                if not has_meaningful_change(hunk):
                    self.dropped["whitespace_only"] += 1
                elif key in self.seen or key in eligible:
                    self.dropped["duplicate"] += 1
                elif (
                    n_tokens := len(self.tokenizer(render(hunk)).input_ids)
                ) > self.cfg.max_tokens:
                    self.dropped["too_long"] += 1
                else:
                    eligible[key] = (hunk, n_tokens)
        k = min(len(eligible), self.cfg.max_hunks_per_commit, limit)
        chosen = rng.sample(sorted(eligible), k)
        self.seen.update(chosen)
        return [eligible[key] for key in chosen]


def _is_reviewable(commit: dict[str, Any]) -> bool:
    """Skip merges (their changes appear in the merged commits) and dependency bots."""
    login = (commit.get("author") or {}).get("login") or ""
    author = f"{login} {commit['commit']['author'].get('name') or ''}"
    return len(commit["parents"]) == 1 and not any(m in author.lower() for m in BOT_MARKERS)


def _record(
    repo: Repo, stratum: str, commit: dict[str, Any], hunk: Hunk, n_tokens: int
) -> dict[str, Any]:
    uid = f"{repo.repo}@{commit['sha']}:{hunk.file}#{hunk.index}"
    return {
        "id": hashlib.sha1(uid.encode()).hexdigest()[:12],
        "repo": repo.repo,
        "stack": repo.stack,
        "license": repo.license,
        "split": repo.split,
        "stratum": stratum,
        "commit": commit["sha"],
        "committed_at": commit["commit"]["committer"]["date"],
        "subject": (commit["commit"]["message"].splitlines() or [""])[0][:200],
        "file": hunk.file,
        "hunk_index": hunk.index,
        "hunk": hunk.text,
        "n_tokens": n_tokens,
    }


def _git_revision() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    return head.strip() + ("-dirty" if dirty.strip() else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repos", type=Path, default=Path("configs/repos.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/collect.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/hunks"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/github"))
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every quota")
    parser.add_argument("--only", nargs="*", help="restrict to these repos (owner/name)")
    args = parser.parse_args()

    cfg = load_collect_config(args.config)
    repos = [r for r in load_repos(args.repos) if not args.only or r.repo in args.only]
    collector = Collector(GitHub(args.cache), cfg, AutoTokenizer.from_pretrained(cfg.tokenizer))
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"revision": _git_revision(), "config": asdict(cfg)}
    manifest |= {"scale": args.scale, "splits": {}}
    for split in SPLITS:
        records: list[dict[str, Any]] = []
        for repo in tqdm([r for r in repos if r.split == split], desc=split, unit="repo"):
            quota = max(1, round(cfg.hunks_per_repo[split] * args.scale))
            got = collector.collect(repo, quota)
            if len(got) < quota:
                tqdm.write(f"warning: {repo.repo} yielded {len(got)}/{quota} hunks")
            records.extend(got)
        path = args.out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        manifest["splits"][split] = {
            "hunks": len(records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "by_stratum": Counter(r["stratum"] for r in records),
            "by_repo": Counter(r["repo"] for r in records),
        }
    manifest["dropped"] = collector.dropped
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({s: v["hunks"] for s, v in manifest["splits"].items()}))


if __name__ == "__main__":
    main()
