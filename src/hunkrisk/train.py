"""Fine-tune a base model with LoRA on teacher-labeled hunks.

    uv run python -m hunkrisk.train configs/train/qwen3.5-2b-lora.yaml

Each run writes runs/<run_id>/ with the resolved config, git revision, dataset hashes, loss
history, final metrics and the LoRA adapter. Run directories are never overwritten.
"""

from unsloth import FastModel  # isort: skip  # must load before transformers to patch it

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from hunkrisk.sft import Collator, load_labeled, rebalance, tokenize


@dataclass(frozen=True)
class TrainConfig:
    name: str
    model: str
    load_in_4bit: bool
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 16
    epochs: float = 2
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 16
    warmup: float = 0.03
    eval_every: int = 40
    patience: int = 3
    seed: int = 13
    high_share: float | None = None  # rebalance train to this share of `high`, or keep as is
    train_hunks: str = "data/hunks/train.jsonl"
    train_labels: str = "data/labels/teacher/train.jsonl"
    val_hunks: str = "data/hunks/val.jsonl"
    val_labels: str = "data/labels/teacher/val.jsonl"


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    return head.strip() + ("-dirty" if dirty.strip() else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--max-steps", type=int, default=-1, help="cap steps, for smoke runs")
    args = parser.parse_args()

    cfg = TrainConfig(**yaml.safe_load(args.config.read_text()))
    run_dir = args.runs / f"{datetime.now():%Y%m%d-%H%M}-{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    model, processor = FastModel.from_pretrained(
        cfg.model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        full_finetuning=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    model = FastModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    train_records = load_labeled(Path(cfg.train_hunks), Path(cfg.train_labels))
    if cfg.high_share:
        train_records = rebalance(train_records, cfg.high_share, cfg.seed)
    val_records = load_labeled(Path(cfg.val_hunks), Path(cfg.val_labels))
    train = tokenize(train_records, tokenizer, cfg.max_seq_length)
    val = tokenize(val_records, tokenizer, cfg.max_seq_length)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            num_train_epochs=cfg.epochs,
            max_steps=args.max_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=cfg.warmup,
            optim="adamw_8bit",
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=cfg.eval_every,
            save_strategy="steps",
            save_steps=cfg.eval_every,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            train_sampling_strategy="group_by_length",
            remove_unused_columns=False,
            seed=cfg.seed,
            report_to="none",
        ),
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(val),
        data_collator=Collator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train()
    minutes = (time.perf_counter() - started) / 60

    model.save_pretrained(run_dir / "adapter")
    tokenizer.save_pretrained(run_dir / "adapter")
    high = sum(r["label"]["risk"] == "high" for r in train_records)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    (run_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    metrics = {
        "revision": git_revision(),
        "datasets": {k: sha256(getattr(cfg, k)) for k in
                     ("train_hunks", "train_labels", "val_hunks", "val_labels")},
        "train_examples": len(train),
        "train_high_share": round(high / len(train_records), 4),
        "val_examples": len(val),
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "steps": trainer.state.global_step,
        "minutes": round(minutes, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }  # fmt: skip
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"run": str(run_dir)} | metrics))


if __name__ == "__main__":
    main()
