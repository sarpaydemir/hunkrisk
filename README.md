# hunkrisk

A 2B-parameter model, fine-tuned on a laptop GPU, that reads one diff hunk and says whether a human needs to review it closely, and why.

```diff
app/Http/Controllers/InvoiceController.php
-        if ($invoice->user_id !== $request->user()->id) {
-            abort(403);
-        }
         $invoice->delete();
```

```json
{"risk": "high", "categories": ["auth"], "reason": "Removes the invoice-owner authorization check, so any authenticated user can now delete any invoice."}
```

<!-- verdict:start -->
**Release bar: met.** The published adapter (Qwen3.5-2B, fine-tuned, 2× data) has recall 0.91, 100% valid JSON, 33× lower cost than the ceiling and F1 within 7.8 points of the ceiling (0.69 vs 0.77).
<!-- verdict:end -->

Recall clears the bar by one hunk; [Results](#results) has the details.

## Why

Review attention is finite. Changes to money, auth, schema, data and secrets deserve a careful read; most hunks don't. A frontier model can triage hunks well, but running one on every change costs money and sends code to a third party. This project measures how close a small model, fine-tuned for this one task, gets to a frontier model while running locally, and holds it to a bar set before any training.

## Results

The fine-tuned 2B model runs on a 6 GB laptop GPU, at about $0.04 per 1,000 hunks when batched at rented-GPU prices: 33× less than the ceiling model. Doubling the training data was the one follow-up iteration. It raised average precision on validation from 0.71 to 0.82, so that model, chosen on validation before its test results existed, is the published one.

It meets the release bar, narrowly. Recall is 0.91 against a bar of 0.90, one hunk above the line: settled another way, the 23 hunks the answer key's two judges disagree on can pull it to 0.89 ([sensitivity](reports/results.md#release-bar-sensitivity-to-the-tie-break)). Its F1 is 7.8 points below the ceiling's, inside the 10 allowed, but on those 23 hunks the adapter and the ceiling make the same number of mistakes, which costs the more precise ceiling more F1. On the 277 hunks the judges agree on, the gap is 10.4 points.

At 0.90 recall its precision is the weak point. Most false alarms are harmless changes inside sensitive code: tests, deprecations, CI and config in auth or payment areas. The teacher, a cheap open-weights API model, is still more accurate than its student and cheaper per hunk. What the local model offers is that code never leaves the machine.

<!-- results:start -->
Scored on 300 test hunks from 5 held-out repositories (56 `high`). The answer key is two frontier models' agreement on 277 and a third model's vote (Mistral Large 4) on the 23 they disagree on. Local models use a P(high) threshold tuned on panel-labeled validation; API models are scored as returned. † Panel member: it helped write the answer key, so its scores are inflated.

| System | Recall (95% CI) | Precision | Precision at base rate | F1 (95% CI) | Valid JSON | Cost / 1k hunks | Latency (GPU) |
|---|---|---|---|---|---|---|---|
| Keyword/path heuristic | 0.73 (0.62–0.84) | 0.35 | 0.35 | 0.48 (0.38–0.57) | 100.0% | $0 | n/a |
| Qwen3-1.7B, zero-shot | 0.95 (0.88–1.00) | 0.23 | 0.23 | 0.37 (0.31–0.45) | 0.0% | $0.120 | 2.01 s |
| Qwen3.5-2B, zero-shot | 0.88 (0.78–0.95) | 0.24 | 0.24 | 0.38 (0.30–0.46) | 98.7% | $0.087 | 1.42 s |
| **Qwen3-1.7B, fine-tuned** | 0.93 (0.86–0.98) | 0.33 | 0.32 | 0.49 (0.40–0.57) | 100.0% | $0.044 | 2.71 s |
| **Qwen3.5-2B, fine-tuned** | 0.88 (0.78–0.95) | 0.47 | 0.47 | 0.61 (0.52–0.69) | 100.0% | $0.037 | 2.17 s |
| **Qwen3.5-2B, fine-tuned, 2× data** | 0.91 (0.83–0.97) | 0.55 | 0.55 | 0.69 (0.60–0.77) | 100.0% | $0.036 | 2.19 s |
| Jev 1.14 (decision API) | 0.71 (0.60–0.83) | 0.75 | 0.75 | 0.73 (0.64–0.82) | n/a | $0.088 | n/a |
| DeepSeek V4.1 Flash (teacher) | 0.79 (0.68–0.89) | 0.85 | 0.84 | 0.81 (0.73–0.89) | n/a | $0.025 | n/a |
| Kimi K3.1 Pro (ceiling) | 0.91 (0.83–0.98) | 0.66 | 0.66 | 0.77 (0.68–0.84) | n/a | $1.19 | n/a |
| Claude Sonnet 5 † | 0.93 (0.85–0.99) | 0.93 | 0.93 | 0.93 (0.87–0.98) | n/a | $1.78 | n/a |
| GPT-5.6 Terra † | 0.89 (0.80–0.97) | 0.85 | 0.84 | 0.87 (0.80–0.93) | n/a | $1.22 | n/a |

Full breakdown: [reports/results.md](reports/results.md).
<!-- results:end -->

The [error analysis](reports/error-analysis.md) reads every test error of both fine-tuned Qwen3.5-2B models.

## How it works

1. **Data.** Hunks from 30 permissively licensed open-source applications (Laravel, Django, FastAPI, Go, TypeScript), [split by repository](configs/repos.yaml), so no project's code appears in more than one split. 60% come from commits that touch risk-looking paths, 40% from random commits. Lockfiles, generated code, whitespace-only and duplicate hunks are dropped. See the [dataset card](docs/dataset-card.md).
2. **Labels.** The rules are in the [labeling guide](docs/labeling-guide.md). DeepSeek V4.1 Flash (open weights, MIT) labels train and validation as the teacher. The test set is labeled independently by two frontier models from different companies. Where they agree, that's the answer. On the 23 hunks where they disagree, a third model from another company (Mistral Large 4) labels blind and the majority decides, under a rule committed before it ran.
3. **Training.** LoRA on one RTX 3060 Laptop GPU (6 GB): Qwen3.5-2B with 16-bit LoRA and Qwen3-1.7B with QLoRA, then Qwen3.5-2B again on twice the data; completion-only loss, early stopping on validation loss. Each run writes its config, git revision, dataset hashes and metrics to [`runs/`](runs/).
4. **Decision.** The model answers in JSON with `risk` first, so P(`high`) is read straight from the logits of that token. The decision threshold is the highest one that reaches 0.90 recall on validation, frozen before test.
5. **Evaluation.** The same 300 test hunks for every system: a keyword/path heuristic, both base models zero-shot, the fine-tuned models, the teacher, a purpose-built decision API (Jev), and a frontier model from a company with no part in the answer key (Kimi K3.1 Pro) as the ceiling. Bootstrap confidence intervals, precision reweighted to the natural base rate, per-category F1, validity, latency and cost per 1,000 hunks.

Every choice, including the ones that changed after measuring, is in the [decision log](docs/decisions.md).

## Release bar

[Pre-registered](docs/release-bar.md) before the first training run. All three must hold on the test set:

- recall ≥ 0.90 on `high`
- ≥ 98% valid JSON with unconstrained decoding
- ≥ 10× cheaper than the ceiling model, at most 10 points lower F1

**Changed from the plan:** the bar assumes a test set labeled blind by a human. The answer key is model-made instead: two frontier judges, and a third settling their disagreements. Scores measure agreement with a three-model majority, not with a person. The [decision log](docs/decisions.md) has the reasons.

## Use it

The published adapter is [`sarpaydemir/hunkrisk-qwen3.5-2b-lora`](https://huggingface.co/sarpaydemir/hunkrisk-qwen3.5-2b-lora) ([model card](docs/model-card.md)).

```sh
git diff main... | uv run python -m hunkrisk.predict --model sarpaydemir/hunkrisk-qwen3.5-2b-lora
```

On [`examples/sample.diff`](examples/sample.diff), three hunks from an invented Laravel app:

```json
{"file": "app/Http/Controllers/InvoiceController.php", "hunk": "@@ -38,10 +38,7 @@ class InvoiceController extends Controller", "risk": "high", "p_high": 0.99683, "answer": {"risk": "high", "categories": ["auth"], "reason": "Removes the invoice-owner authorization check, so any authenticated user can now delete any invoice."}}
{"file": "database/migrations/2026_10_20_000000_drop_legacy_totals_from_orders.php", "hunk": "@@ -0,0 +1,15 @@", "risk": "high", "p_high": 0.99753, "answer": {"risk": "high", "categories": ["schema", "data"], "reason": "Drops the legacy_total and legacy_currency columns from the orders table, which removes persisted data and requires a migration to backfill existing rows."}}
{"file": "resources/views/invoices/index.blade.php", "hunk": "@@ -12,7 +12,7 @@", "risk": "low", "p_high": 0.00026, "answer": {"risk": "low", "categories": ["none"], "reason": "Only swaps a hardcoded column header for a translation key in a Blade table, with no change to data, access, or money behavior."}}
```

Each line has P(`high`) and the model's own answer. `--threshold` sets the flagging point: the default 0.5 follows the model's own answer, and 0.148 is the evaluated operating point with 0.90 recall on validation.

## Documents

| | |
|---|---|
| [Release bar](docs/release-bar.md) | pass criteria, fixed before training |
| [Decision log](docs/decisions.md) | every design choice and change of plan, dated, with reasons |
| [Labeling guide](docs/labeling-guide.md) | the rules, used verbatim as the teacher prompt |
| [Dataset card](docs/dataset-card.md) | sources, filtering, label statistics, limitations |
| [Model card](docs/model-card.md) | the published adapter: usage, training, evaluation |
| [Results](reports/results.md) | every table: per category, per stratum, thresholds, release bar |
| [Error analysis](reports/error-analysis.md) | every test error, read and grouped |
| [How it was built](HOW-IT-WAS-BUILT.md) | how this was directed with a coding agent |

## Repository layout

```
configs/    source repositories and splits, collection and training configs
docs/       release bar, labeling guide, decision log, dataset and model cards
scripts/    one-off tools (GPU smoke test)
src/        the hunkrisk package: collection, labeling, training, evaluation, prediction
tests/      unit tests for the pure logic (no network, no GPU)
data/       local datasets (gitignored), except the gold test labels
runs/       per-run configs and metrics
reports/    results and error analysis
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, for training or local inference, an NVIDIA GPU with ≥ 6 GB VRAM.

```sh
uv sync                    # or `uv sync --no-group train` without a GPU
cp .env.example .env       # OPENROUTER_API_KEY for labeling; collection uses `gh auth token`
```

The commands for each step, from collection to the report, are listed in [`CLAUDE.md`](CLAUDE.md).

## License

Code and labels are MIT. Hunks in the dataset keep the licenses of their source repositories, listed in [`configs/repos.yaml`](configs/repos.yaml). The adapter is Apache-2.0, like its base model.
