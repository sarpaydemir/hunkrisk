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

Pending.

## Repository layout

```
configs/    source repositories and splits, training configs
docs/       release bar, labeling guide, decision log
scripts/    one-off tools (GPU smoke test)
src/        the hunkrisk package: collection, labeling, training, evaluation
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
