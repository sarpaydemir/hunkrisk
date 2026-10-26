# Decisions

A running log of design choices and the reasons behind them, newest at the bottom. Items marked **open** are settled by an experiment noted in the entry.

## 2026-10-26: before data collection

### Data

- **Split by repository, fixed before any data is pulled.** Hunks from one repo share naming, conventions and near-duplicate code. A random hunk-level split would leak that into test and overstate quality. The split lives in [`configs/repos.yaml`](../configs/repos.yaml).
- **Diffs come from the GitHub API, not clones.** Only sampled commits are needed. Listing commits (with `path=` filters for targeted sampling) and fetching per-commit diffs avoids cloning multi-gigabyte repositories.
- **Sampling mixes targeted and random commits.** Targeted commits touch paths or keywords associated with risk (migrations, auth, payments, env and config). Random commits are drawn uniformly from history. Sampling only by keyword would make the keyword baseline look artificially strong and miss the hard cases, like a pricing change in `utils.py` or a permission check inside a controller. The keyword baseline's recall is reported separately on both parts.
- **The test set is at least 30% `high`.** With ~300 hunks that gives 90+ positives, enough for a usable recall estimate. Since oversampling inflates precision, precision is also reported reweighted to the natural base rate measured on the random sample.
- **Dropped before labeling:** lockfiles, generated and vendored files, binary diffs, near-duplicate hunks, and hunks over 1,500 tokens.

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

  **Open:** picked by a 50-hunk pilot on validation repos.
- **Ceiling model: a frontier model, on the test set only.** `anthropic/claude-sonnet-5.5` ($2 / $10 per M tokens) costs about $0.90 for 300 hunks. It sets the quality ceiling and the cost reference in the release bar. Its outputs are never used as training data.
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
