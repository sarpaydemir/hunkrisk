# Decisions

A running log of design choices and the reasons behind them, newest at the bottom. Items marked **open** are settled by an experiment noted in the entry.

## 2026-10-26: before data collection

### Data

- **Split by repository, fixed before any data is pulled.** Hunks from one repo share naming, conventions and near-duplicate code. A random hunk-level split would leak that into test and overstate quality. The split lives in [`configs/repos.yaml`](../configs/repos.yaml).
- **Diffs come from the GitHub API, not clones.** Only sampled commits are needed. Listing commits (with `path=` filters for targeted sampling) and fetching per-commit diffs avoids cloning multi-gigabyte repositories.
- **Sampling mixes targeted and random commits.** Targeted commits touch paths or keywords associated with risk (migrations, auth, payments, env and config). Random commits are drawn uniformly from history. Sampling only by keyword would make the keyword baseline look artificially strong and miss the hard cases, like a pricing change in `utils.py` or a permission check inside a controller. The keyword baseline's recall is reported separately on both parts.
- **The test set is at least 30% `high`.** With ~300 hunks that gives 90+ positives, enough for a usable recall estimate. Since oversampling inflates precision, precision is also reported reweighted to the natural base rate measured on the random sample. *(Superseded by the base-rate entry below.)*
- **Collection parameters** live in [`configs/collect.yaml`](../configs/collect.yaml):
  - **Window:** commits from 2022-01-01 to a fixed cutoff of 2026-10-26, so reruns see the same history.
  - **Quotas:** 60 hunks per test and validation repo, 150 per train repo, for 300 / 300 / 3,000 in total. Train is collected above the 2,400 target so it can be rebalanced once teacher labels exist.
  - **Strata:** 60% targeted, 40% random. Targeted commits come from listings filtered by up to six sampled risk-looking paths per category, with the categories interleaved so a busy one (e.g. `config/`) can't crowd out the rest. Random commits come from random history pages, at most ten per page so they spread across time.
  - **Diversity:** at most three hunks per commit.
  - **Reproducibility:** randomness is seeded per repository, and API responses are cached on disk.
- **Dropped before labeling:**
  - merge commits, bot commits (Dependabot, Renovate) and commits touching more than 50 files
  - lockfiles, vendored and generated code, translations, snapshots and binary diffs
  - whitespace-only hunks and near-duplicates (same changed lines, ignoring whitespace and path)
  - hunks over 1,500 tokens, measured with the Qwen3 tokenizer (the more verbose of the two candidates)
- **Dedupe runs across splits, test first.** If a hunk appears in both a test and a train repo, it stays in test only.

### Labels

- **One rule, written down first.** Money, auth, schema, data and secrets changes get reviewed first. The [labeling guide](labeling-guide.md) defines each category and is the source of the teacher prompt, so the teacher and the human apply the same rules.
- **Scope calls.** `schema` covers persisted data only; API contract changes are a different kind of risk. CI, Docker and infrastructure files are `high` only through an existing category. Dependency bumps are `low`; supply-chain risk is better handled by dedicated scanners. Ties go to `high`, matching the release bar's emphasis on recall.
- **Gold test labels are human and blind.** The labeler never sees teacher output. Reviewing teacher labels instead would anchor the reviewer and inflate agreement. Teacher–human agreement (Cohen's κ) is reported.

### Teacher and budget

- **All API calls fit in $2,** enforced as a spend limit on the API key.
- **Teacher: a flash-tier model through OpenRouter.** Labeling ~2,700 train and validation hunks is about 3M input tokens. That's roughly $0.20 at flash-tier prices versus $6+ for a frontier model. Prefer a model whose terms allow training on its outputs. Candidates, OpenRouter list prices in $/M tokens in/out on 2026-10-26:

  | Model | In | Out |
  |---|---|---|
  | `deepseek/deepseek-v4.1-flash` | 0.048 | 0.191 |
  | `z-ai/glm-5.3` | 0.07 | 1.40 |
  | `anthropic/claude-haiku-5.5` | 0.10 | 0.50 |
  | `qwen/qwen3.8-flash` | 0.15 | 0.47 |
  | `openai/gpt-5.6-luna` | 0.20 | 1.20 |

  **Settled below:** DeepSeek V4.1 Flash, picked by a 50-hunk pilot on validation repos.
- **Ceiling model: a frontier model, on the test set only.** `anthropic/claude-sonnet-5.5` ($2 / $10 per M tokens) costs about $0.90 for 300 hunks. It sets the quality ceiling and the cost reference in the release bar. Its outputs are never used as training data. *(Model changed to Sonnet 5 below.)*
- **No local teacher.** Models that fit in 6 GB aren't strong enough to label. The base-model zero-shot baseline runs through the same stack used for training.

### Evaluation

- **F1 in the release bar is F1 of the `high` class.**
- **Recall is set with a threshold tuned on validation.** `risk` is the first field of the output, so P(`high`) can be read from the logits at that position. The threshold that reaches 0.90 recall on validation is frozen and applied to test.
- **Baselines share the test set and the prompt:** a keyword/path heuristic, the base model zero-shot, the teacher zero-shot and the ceiling model zero-shot.

### Base model

- **Candidates:**
  - Qwen3.5-2B: current generation, Apache-2.0, hybrid linear attention, 248k vocabulary, plus a vision tower that stays frozen
  - Qwen3-1.7B: text-only, mature tooling
  - Qwen3.5-0.8B: fallback
- **Qwen3.5 trains with 16-bit LoRA, not QLoRA.** Unsloth advises against 4-bit training on Qwen3.5 because of "higher than normal quantization differences". Qwen3-1.7B uses QLoRA.
- **All three fit on the reference GPU.** Smoke test with [`scripts/smoke_vram.py`](../scripts/smoke_vram.py): LoRA r=16, one 2,048-token sequence per step, batch 1, Unsloth gradient checkpointing, torch 2.14.1+cu130, unsloth 2026.10.9.

  | Model | Method | Peak VRAM | s/step |
  |---|---|---|---|
  | Qwen3-1.7B | QLoRA 4-bit | 2.08 GB | 1.17 |
  | Qwen3.5-2B | 16-bit LoRA | 4.89 GB | 1.50 |
  | Qwen3.5-0.8B | 16-bit LoRA | 2.23 GB | 0.80 |

  **Open:** train Qwen3.5-2B (16-bit LoRA) and Qwen3-1.7B (QLoRA), and pick on validation.

## 2026-10-26: teacher pilot and base rate

- **Teacher: `deepseek/deepseek-v4.1-flash`** (open weights, MIT license). In a pilot on 50 validation hunks, balanced across repositories and compared against Claude Sonnet 5:

  | Candidate | Agreement | κ | Recall `high` | Precision `high` | Cost / 50 |
  |---|---|---|---|---|---|
  | DeepSeek V4.1 Flash | 0.94 | 0.81 | 0.89 | 0.80 | $0.0065 |
  | Qwen3.8 Flash | 0.92 | 0.75 | 0.89 | 0.73 | $0.0036 |

  Only 9 of the 50 hunks were `high` per the reference, so the gap is about one hunk. The license settles it. GLM-5.3 was dropped: OpenRouter requires reasoning for it, which makes it the most expensive of the cheap candidates.
- **Ceiling model: `anthropic/claude-sonnet-5`, reasoning off.** On OpenRouter, Sonnet 5.5 requires reasoning, which is billed as output and would put the 300-hunk ceiling run close to the whole API budget. Sonnet 5 has the same list price and can run without reasoning. That's also the fairer comparison, because the fine-tuned model doesn't reason either.
- **Base rate: teacher labels put validation at 18% `high`** (23% of targeted hunks, 12% of random ones). The ≥ 30% target assumed path-targeted commits would be richer in risky hunks than they are. The test set is kept as collected rather than re-selected with a model's help, which would bias it toward hunks a model already recognizes. That means roughly 50–60 positives in test, so recall comes with a bootstrap 95% CI (about ±8 points at that size) reported next to the point estimate.

## 2026-10-27: answer key, ceiling and baselines

- **Gold test labels: a two-model panel plus human adjudication.** Labeling all 300 test hunks by hand was cut for time. Instead, Claude Sonnet 5 and GPT-5.6 Terra label them independently with the same prompt and reasoning off. Neither comes from the teacher's company (DeepSeek) or the student's (Qwen). Where they agree on risk, that's the gold label; when both say `high`, the categories are the union of theirs. They agreed on 277 of 300 (κ 0.75). The 23 disagreements are labeled by a human in the blind tool, and human labels always win. Results are marked provisional until those 23 are in.
  - Considered and rejected: a single model as the answer key, which would measure agreement with that model rather than correctness, and TypeSafe Jev, see below.
- **Ceiling model changed from Claude Sonnet 5 to Kimi K3.1 Pro**, before any fine-tuned results existed. Sonnet became a panel member, and a panel member scores perfectly on every hunk the panel agreed on, so its F1 is inflated by construction. Kimi K3.1 Pro (Moonshot AI) is a flagship model from a fifth company, can run without reasoning, and had no part in the answer key. Its pilot agreement with Sonnet was 0.92 (κ 0.73). Sonnet and GPT stay in the results table, marked as panel members.
- **TypeSafe Jev 1.14 is a baseline, not a judge.** It's a decision model that returns probabilities for typed questions; the guide maps onto one yes/no question for risk and one per category. On the 50-hunk pilot it ranked hunks reasonably (AUC 0.85 against Sonnet) but caught only 3 of 9 risky hunks at the default threshold, and no threshold matched the chat models. A shorter, choice-style prompt did worse (AUC 0.79). It's too weak for an answer key, but at about $0.09 per 1,000 hunks it's a useful comparison point for a purpose-built decision API.
- **API calls go to the cheapest provider.** One model can cost 10x more depending on which provider serves it (GPT-5.6 Terra: $1 to $4 per M input tokens; DeepSeek V4.1 Flash: $0.048 to $0.60). Requests ask OpenRouter for the cheapest provider first, and each reply records which provider served it.
- **Keyword/path heuristic** ([`heuristic.py`](../src/hunkrisk/heuristic.py)): a hunk is `high` when its path or changed lines match a category's keywords, except in test, doc and fixture files. It's the deterministic floor.

## 2026-10-27: first results and one follow-up iteration

- **The first fine-tuned models miss the release bar,** on the provisional answer key (277 of 300 hunks). Qwen3.5-2B reaches recall 0.85 at precision 0.49 (F1 0.62), against Kimi's 0.82 F1. Qwen3-1.7B does worse (F1 0.48). Both are published as the pre-registered result.
- **Two causes, both visible without the test set:**
  - *The threshold is tuned against teacher labels.* On test, the teacher itself finds only 85% of the `high` hunks, so a student tuned to find 90% of the teacher's `high` hunks on validation lands near 85% against the answer key. Validation needs labels of the same kind as test.
  - *The student overfits.* Training loss falls to 0.37 while validation loss bottoms out at 0.61 in the second epoch, with 559 `high` training examples.
- **Follow-up, one iteration, chosen on validation only:**
  - Label validation with the same two-model panel as test (about $0.90). Their agreements become the reference the threshold is tuned on; disagreements are left out, as on the provisional test set. The release bar only says the threshold is chosen on validation, so this stays within it, but it's a change of protocol made after seeing test results, so both versions are reported.
  - Double the training data: 300 hunks per train repository instead of 150, same seed, same filters, labeled by the same teacher. Validation and test are collected first and must come out byte-identical, so deduplication against them is unchanged.
  - The model, recipe and hyperparameters stay the same, so the comparison isolates data size.
- **The first cause didn't hold up on validation.** The panel agreed on 270 of 300 validation hunks (36 `high`), and the teacher finds 35 of those 36 (recall 0.97, κ 0.89). So teacher labels aren't what pulls test recall down; the shortfall on test comes from the test repositories themselves (see the error analysis). Panel-labeled validation is kept as the threshold reference because it matches how test is labeled. It lowers the 2B model's threshold from 0.148 to 0.107. Results are shown under both references.
