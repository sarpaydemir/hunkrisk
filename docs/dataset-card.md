# Dataset card: hunkrisk

Single diff hunks from public open-source applications, each labeled with whether it needs close human review (`high` or `low`), which risk categories it touches, and a one-sentence reason.

## Contents

| Split | Hunks | Repositories | Targeted / random | Labels from |
|---|---|---|---|---|
| train | 2,993 | 20 | 1,714 / 1,279 | teacher model |
| train, doubled | 5,737 | 20 | 3,051 / 2,686 | teacher model |
| validation | 300 | 5 | 180 / 120 | teacher model |
| test | 300 | 5 | 180 / 120 | two-model panel, disagreements settled by a third model |

The doubled train set comes from a follow-up run (see [decisions](decisions.md)): the same 20 repositories with a quota of 300 hunks each instead of 150, the same seed and filters. Two repositories run out of eligible hunks (django-oscar 164, fastapi-users 173). Validation and test come out byte-identical in that run, so train is deduplicated against exactly the same hunks. It overlaps the first train set by 1,835 hunks, since a larger quota draws commits in a different order.

Each record has the hunk itself (file path plus unified-diff hunk) and its provenance: repository, stack, license, commit SHA and date, commit subject, sampling stratum and token count. The commit subject is kept for analysis only. Models never see it, and neither does the blind labeling tool.

Hunk length, in Qwen3 tokens with path included: median 136, mean 214, 95th percentile 725, hard cap 1,500 (train).

## Sources

30 permissively licensed projects (15 MIT, 12 Apache-2.0, 3 BSD-3-Clause) across Laravel, Django, FastAPI, Go and TypeScript, listed with their split in [`configs/repos.yaml`](../configs/repos.yaml). Each split covers all five stacks. The split is **by repository**, so no project's code or conventions appear in more than one split.

Commits are drawn from 2022-01-01 to a fixed cutoff of 2026-10-26, from two strata:

- **Targeted (60%):** commits that touch paths that look risk-related, sampled per category (auth, money, schema, secrets, data) and interleaved so no single category dominates.
- **Random (40%):** commits from random pages of history. This stratum estimates the natural base rate.

## Filtering

Dropped before labeling, with counts from the first collection run:

- Commits: merges, bot commits, and 305 commits that touch more than 50 files.
- Files: 1,225 lockfiles, vendored or generated files, translations, snapshots and assets, plus 205 binary or unrendered patches.
- Hunks: 263 whitespace-only, 580 over 1,500 tokens, and 3,992 near-duplicates (same changed lines ignoring whitespace, path and context, deduplicated across splits with test first).

At most three hunks are kept per commit.

## Labels

The rules are in the [labeling guide](labeling-guide.md): `high` when the hunk changes behavior in auth, money, schema, data or secrets, and a mistake could cost money, weaken access control, corrupt or lose data, or expose a secret. Ties go to `high`.

- **Train and validation:** DeepSeek V4.1 Flash (open weights, MIT), prompted with the full guide, temperature 0, reasoning off. All 3,293 labels were valid. 18.7% of train and 18.3% of validation are `high`. Labeling cost $0.26 in total. On the doubled train set, 5,735 of 5,737 labels were valid (the other two are dropped) and 18.0% are `high`; it cost $0.33, of which $0.19 was new.
- **Validation, second reference:** the same two-model panel as test also labels validation, and its 270 agreements (36 `high`) are what decision thresholds are tuned on. Its 30 disagreements are left out; nobody adjudicates them.
- **Test:** Claude Sonnet 5 and GPT-5.6 Terra label independently with the same prompt. They agreed on risk for 277 of 300 hunks (κ 0.75), and those agreements are the gold labels; when both say `high`, the categories are the union of theirs. The 23 disagreements go to a third judge, Mistral Large 4, from a company with no other role in the project. It labels them with the same prompt without seeing the other labels, and the majority of the three decides; it called 10 of them `high`. In all, 56 of the 300 test hunks are `high` (18.7%). A blind tool that shows only the path and the hunk can replace any label with a human one; no human labels are in the current set.

`high` hunks by category in train: auth 256, data 174, schema 129, secrets 84, money 55. In the doubled set: auth 431, data 353, schema 247, secrets 146, money 111.

## Known limitations

- **One hunk, no surrounding code.** Some risk only shows with context from other files, for example a harmless-looking helper that every payment path calls.
- **Teacher noise in train.** On the test set, the teacher's `high` recall against the gold labels is in the [results](../reports/results.md).
- **Money is rare:** 55 train examples (111 in the doubled set) and about 16 in test, so its per-category score is noisy.
- **Base rate is about 18% `high`,** below the 30% the design aimed for. Path-targeted commits turned out less risky than expected, about 23% `high` against 12% for random commits.
- **A model-built answer key.** Test labels come from two frontier models, with a third settling their disagreements, and no human label. Hunks where the models are wrong in the same way go unnoticed, and a system that thinks like the judges scores higher.

## Reproducing

```sh
uv run python -m hunkrisk.collect          # needs a GitHub token (`gh auth login`)
uv run python -m hunkrisk.label --model deepseek/deepseek-v4.1-flash \
    --input data/hunks/train.jsonl --max-usd 0.45
uv run python -m hunkrisk.collect --config configs/collect-train-x2.yaml --out data/hunks-x2
```

Collection is seeded per repository and cut off at a fixed date. Rebuilt from the same API responses, the splits come out byte-identical; the SHA-256 hashes are in `data/hunks/manifest.json`. Gold test labels are committed in [`data/gold/`](../data/gold/).

## License

The hunks keep the licenses of their source repositories, all permissive and listed per repository. Labels and code are MIT.
