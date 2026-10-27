# hunkrisk

A small model that runs on a laptop GPU, reads one diff hunk, and decides whether a human needs to review it closely, and why.

> **Status: in progress.** The evaluation criteria are [pre-registered](docs/release-bar.md); results land below as runs complete.

```diff
app/Http/Controllers/InvoiceController.php
-        if ($invoice->user_id !== $request->user()->id) {
-            abort(403);
-        }
         $invoice->delete();
```

```json
{"risk": "high", "categories": ["auth", "money"], "reason": "Removes the ownership check, so any user can delete any invoice."}
```

## Why

Review attention is finite. Changes to money, auth, schema, data and secrets deserve a careful read; most hunks don't. A frontier model can triage hunks well, but running one on every change is expensive and means sending code to a third party. This project measures how close a ~2B model, fine-tuned on this one task, can get to a frontier model while running locally at a fraction of the cost.

## Approach

1. **Data.** About 3,000 hunks from 30 permissively licensed open-source projects (Laravel, Django, FastAPI, Go, TypeScript), [split by repository](configs/repos.yaml) into train, validation and test.
2. **Labels.** An API model labels train and validation by following the [labeling guide](docs/labeling-guide.md). The 300-hunk test set is labeled by hand, blind to any model output.
3. **Baselines.** A keyword/path heuristic, the base model zero-shot, and a frontier model zero-shot, all on the same test set.
4. **Training.** LoRA on a single RTX 3060 Laptop GPU (6 GB), comparing Qwen3.5-2B (16-bit LoRA) and Qwen3-1.7B (QLoRA).
5. **Evaluation.** Recall and precision on `high`, per-category F1, output validity, latency and cost per 1,000 hunks, plus an error taxonomy built by reading every test error.

Design choices and their reasons are logged in [`docs/decisions.md`](docs/decisions.md).

## Release bar

Set before training ([full text](docs/release-bar.md)):

- recall ≥ 0.90 on `high`
- ≥ 98% valid outputs with unconstrained decoding
- ≥ 10× cheaper than the frontier model, at most 10 points lower F1

## Results

<!-- results:start -->
**Provisional:** 23 hunks where the two judges disagreed still need a human label and are left out. Scored on 277 test hunks from 5 held-out repositories (46 `high`). Local models use a P(high) threshold tuned on panel-labeled validation; API models are scored as returned. † Panel member: it helped write the answer key, so its scores are inflated.

| System | Recall (95% CI) | Precision | Precision at base rate | F1 (95% CI) | Valid JSON | Cost / 1k hunks | Latency (GPU) |
|---|---|---|---|---|---|---|---|
| Keyword/path heuristic | 0.78 (0.65–0.90) | 0.35 | 0.33 | 0.48 (0.37–0.57) | 100.0% | $0 | n/a |
| Qwen3-1.7B, zero-shot | 0.96 (0.88–1.00) | 0.21 | 0.20 | 0.35 (0.26–0.42) | 0.0% | $0.120 | 2.01 s |
| Qwen3.5-2B, zero-shot | 0.87 (0.76–0.96) | 0.22 | 0.20 | 0.35 (0.26–0.43) | 98.7% | $0.087 | 1.42 s |
| **Qwen3-1.7B, fine-tuned** | 0.93 (0.86–1.00) | 0.32 | 0.30 | 0.47 (0.37–0.56) | 100.0% | $0.044 | 2.71 s |
| **Qwen3.5-2B, fine-tuned** | 0.87 (0.76–0.96) | 0.48 | 0.45 | 0.62 (0.50–0.70) | 100.0% | $0.037 | 2.17 s |
| Jev 1.14 (decision API) | 0.80 (0.69–0.91) | 0.84 | 0.83 | 0.82 (0.73–0.90) | n/a | $0.088 | n/a |
| DeepSeek V4.1 Flash (teacher) | 0.85 (0.74–0.95) | 0.93 | 0.92 | 0.89 (0.81–0.95) | n/a | $0.025 | n/a |
| Kimi K3.1 Pro (ceiling) | 0.91 (0.83–0.98) | 0.74 | 0.72 | 0.82 (0.72–0.89) | n/a | $1.19 | n/a |
| Claude Sonnet 5 † | 1.00 (1.00–1.00) | 1.00 | 1.00 | 1.00 (1.00–1.00) | n/a | $1.78 | n/a |
| GPT-5.6 Terra † | 1.00 (1.00–1.00) | 1.00 | 1.00 | 1.00 (1.00–1.00) | n/a | $1.22 | n/a |

Full breakdown: [reports/results.md](reports/results.md).
<!-- results:end -->

## Repository layout

```
configs/    source repositories and splits, training configs
docs/       release bar, labeling guide, decision log
scripts/    one-off tools (GPU smoke test)
src/        the hunkrisk package: collection, labeling, training, evaluation
tests/      unit tests for the pure logic (no network, no GPU)
data/       local datasets (gitignored), except the human gold labels
runs/       per-run configs and metrics
reports/    evaluation tables and error analysis
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, for training, an NVIDIA GPU with ≥ 6 GB VRAM.

```sh
uv sync                    # or `uv sync --no-group train` without a GPU
cp .env.example .env       # add OPENROUTER_API_KEY for labeling and API baselines
```

## License

Code is MIT. Hunks in the dataset keep the licenses of their source repositories, listed in [`configs/repos.yaml`](configs/repos.yaml).
