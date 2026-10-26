# hunkrisk

A small fine-tuned model that reads one diff hunk and decides whether it needs close human review, and why. `README.md` has the overview; `docs/` has the design.

## Invariants

These protect the evaluation. Breaking one silently invalidates the results.

- **Split by repository.** `configs/repos.yaml` assigns every source repo to exactly one of train, val or test, and was fixed before any data was pulled. Nothing from test repos is used for training, prompt design, threshold tuning or model selection.
- **Gold labels are human and blind.** Test labels in `data/gold/` are written by a person who hasn't seen teacher or model output for that hunk. Tooling must never show model output before the human label is saved. Agents never edit gold labels.
- **The release bar is pre-registered.** `docs/release-bar.md` was committed before the first training run. Don't edit it; record any change in `docs/decisions.md` with the reason, and call it out in the README.
- **Permissive licenses only.** Every source repo is MIT, Apache-2.0 or BSD, and is listed with its license in `configs/repos.yaml`.
- **Runs are reproducible.** Each training or eval run writes `runs/<run_id>/` with the resolved config, git commit, dataset hash and metrics. Never overwrite a run directory.
- **Paid API calls are capped.** A script that calls a paid API prints its estimated cost, enforces `--max-usd`, and is tried on ≤ 50 items before a full run.

## Commands

```sh
uv sync                                              # everything, incl. torch + unsloth
uv sync --no-group train                             # CPU-only: data, labeling, eval tooling
uv run ruff check . && uv run ruff format --check . && uv run pytest
uv run python -m hunkrisk.collect                    # hunks -> data/hunks/{test,val,train}.jsonl
uv run python -m hunkrisk.collect --scale 0.1 --only krayin/laravel-crm --out data/smoke
uv run python -m hunkrisk.label --model deepseek/deepseek-v4.1-flash --input data/hunks/val.jsonl --max-usd 0.05
uv run python -m hunkrisk.compare <reference.jsonl> <candidate.jsonl>...   # agreement table
uv run python -m hunkrisk.panel <judge-a.jsonl> <judge-b.jsonl>   # panel gold + disagreements
uv run python -m hunkrisk.gold --ids data/gold/disagreements.txt  # blind adjudication UI on :8765
uv run python -m hunkrisk.heuristic --split test
uv run python -m hunkrisk.train configs/train/qwen3.5-2b-lora.yaml
uv run python -m hunkrisk.evaluate --model runs/<run>/adapter --split test --name ft-qwen3.5-2b
uv run python -m hunkrisk.report                     # reports/results.md + README results
uv run python scripts/smoke_vram.py Qwen/Qwen3.5-2B --bf16
```

## Conventions

- Python 3.12, pinned in `.python-version`.
- In training code, import `unsloth` before `transformers`, `trl` and `peft` so its patches apply. The xformers warning at startup is harmless; PyTorch SDPA is used.
- Secrets come from `.env` (see `.env.example`) and never reach code, logs or run directories.
- Record non-obvious choices in `docs/decisions.md` when they're made, not afterwards.
- Comments explain why, not what.

## Reference hardware

RTX 3060 Laptop GPU (6 GB VRAM, 5.7 GB usable), 14 GB RAM.
