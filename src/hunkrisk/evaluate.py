"""Run a local model (base or fine-tuned) over a split and record its predictions.

    uv run python -m hunkrisk.evaluate --model runs/<run>/adapter --name ft-qwen3.5-2b
    uv run python -m hunkrisk.evaluate --model Qwen/Qwen3.5-2B --full-guide \\
        --name qwen3.5-2b-zeroshot

Per hunk it stores the raw generation, the strictly parsed label (or the validation error),
P(high) and latency. P(high) is read from the generation's own logits at the risk token, so it
costs nothing extra; only when a model wanders off the format is a forced-prefix pass used.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from hunkrisk.schema import InvalidOutput, parse_label
from hunkrisk.sft import ANSWER_PREFIX, prompt_text


def load(model_name: str, device: str, load_in_4bit: bool) -> tuple[Any, Any]:
    """Load a base model or a LoRA adapter directory, ready for inference."""
    if device == "cuda":
        from unsloth import FastModel  # GPU only; patches transformers on import

        model, processor = FastModel.from_pretrained(
            model_name, max_seq_length=4096, load_in_4bit=load_in_4bit
        )
        FastModel.for_inference(model)
    else:
        # transformers picks flash-linear-attention's kernels at import time whenever the package
        # is installed, but they are Triton (GPU-only); hiding it selects the PyTorch path.
        sys.modules["fla"] = None  # type: ignore[assignment]
        from peft import PeftConfig, PeftModel
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
        )

        is_adapter = (Path(model_name) / "adapter_config.json").exists()
        base = model_name
        if is_adapter:
            base = PeftConfig.from_pretrained(model_name).base_model_name_or_path
        # Qwen3.5 is multimodal, and its causal-LM class is a text-only variant with different
        # module names: a LoRA trained on the full model silently fails to attach to it.
        multimodal = hasattr(AutoConfig.from_pretrained(base), "vision_config")
        loader = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
        model = loader.from_pretrained(base, dtype=torch.bfloat16)
        if is_adapter:
            model = PeftModel.from_pretrained(model, model_name)
        model.eval()
        processor = AutoTokenizer.from_pretrained(model_name)
    return model, getattr(processor, "tokenizer", processor)


def single_token(tokenizer: Any, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not a single token for this tokenizer: {ids}")
    return ids[0]


class Predictor:
    def __init__(self, model: Any, tokenizer: Any, device: str, full_guide: bool) -> None:
        self.model, self.tokenizer, self.device, self.full_guide = (
            model,
            tokenizer,
            device,
            full_guide,
        )
        self.prefix = tokenizer(ANSWER_PREFIX, add_special_tokens=False).input_ids
        self.high = single_token(tokenizer, "high")
        self.low = single_token(tokenizer, "low")

    def _p_high(self, logits: torch.Tensor) -> float:
        pair = torch.stack([logits[self.high], logits[self.low]]).float()
        return torch.softmax(pair, dim=0)[0].item()

    def _sync(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def predict(self, file: str, hunk: str, max_new_tokens: int) -> dict[str, Any]:
        prompt = prompt_text(self.tokenizer, file, hunk, self.full_guide)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        self._sync()
        started = time.perf_counter()
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self._sync()
        latency = time.perf_counter() - started
        new = out.sequences[0, enc.input_ids.shape[1] :].tolist()
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()

        k = len(self.prefix)
        if new[:k] == self.prefix and len(out.logits) > k:
            p_high, source = self._p_high(out.logits[k][0]), "generation"
        else:  # off-format output: score the risk token under a forced answer prefix
            forced = self.tokenizer(prompt + ANSWER_PREFIX, return_tensors="pt",
                                    add_special_tokens=False).to(self.device)  # fmt: skip
            p_high, source = self._p_high(self.model(**forced).logits[0, -1]), "forced"

        try:
            label, error = parse_label(text).to_json(), None
        except InvalidOutput as e:
            label, error = None, str(e)
        return {
            "label": label,
            "error": error,
            "raw": text,
            "p_high": round(p_high, 5),
            "p_source": source,
            "latency_s": round(latency, 4),
            "prompt_tokens": enc.input_ids.shape[1],
            "new_tokens": len(new),
        }


def length_batches(lengths: list[int], max_size: int, max_tokens: int) -> list[list[int]]:
    """Group indices, shortest first, so each padded batch stays within a token budget."""
    batches: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        batch = batches[-1] if batches else None
        fits = batch and len(batch) < max_size and lengths[i] * (len(batch) + 1) <= max_tokens
        if fits:
            batch.append(i)
        else:
            batches.append([i])
    return batches


@torch.inference_mode()
def throughput(predictor: Predictor, records: list[dict[str, Any]], batch_size: int,
               max_new_tokens: int, max_batch_tokens: int = 4096) -> float:  # fmt: skip
    """GPU-seconds per hunk when processed in length-sorted batches, as a batch job would."""
    tokenizer = predictor.tokenizer
    prompts = [prompt_text(tokenizer, r["file"], r["hunk"], predictor.full_guide) for r in records]
    lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
    tokenizer.padding_side = "left"
    predictor._sync()
    started = time.perf_counter()
    for batch in length_batches(lengths, batch_size, max_batch_tokens):
        enc = tokenizer([prompts[i] for i in batch], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(predictor.device)  # fmt: skip
        predictor.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)  # fmt: skip
    predictor._sync()
    tokenizer.padding_side = "right"
    return (time.perf_counter() - started) / len(prompts)


def run_split(predictor: Predictor, args: argparse.Namespace, split: str) -> dict[str, Any]:
    path = Path(f"data/hunks/{split}.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records = records[: args.limit] if args.limit else records
    suffix = f"-{args.device}" if args.device != "cuda" else ""
    out = Path("data/predictions") / args.name / f"{split}{suffix}.jsonl"
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    predictor.predict(records[0]["file"], records[0]["hunk"], 8)  # warm-up: compile, allocate

    if args.reuse and out.exists():
        rows = [json.loads(line) for line in out.read_text().splitlines()]
    else:
        rows = []
        for r in tqdm(records, desc=f"{args.name}/{split}", unit="hunk"):
            prediction = predictor.predict(r["file"], r["hunk"], args.max_new_tokens)
            rows.append({"id": r["id"]} | prediction)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    latencies = sorted(r["latency_s"] for r in rows)
    summary = {
        "model": args.model,
        "split": split,
        "device": args.device,
        "full_guide": args.full_guide,
        "n": len(rows),
        "valid": sum(r["label"] is not None for r in rows) / len(rows),
        "p_from_generation": sum(r["p_source"] == "generation" for r in rows) / len(rows),
        "latency_mean_s": round(statistics.mean(latencies), 4),
        "latency_p50_s": round(latencies[len(latencies) // 2], 4),
        "latency_p95_s": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 4),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)
        if args.device == "cuda"
        else None,
    }
    if args.throughput_batch:
        summary["throughput_batch"] = args.throughput_batch
        summary["throughput_s_per_hunk"] = round(
            throughput(predictor, records, args.throughput_batch, args.max_new_tokens), 4
        )
    out.with_name(f"{split}{suffix}.summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="hub id or a LoRA adapter directory")
    parser.add_argument("--name", required=True, help="output folder under data/predictions")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--full-guide", action="store_true", help="zero-shot with the full guide")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--limit", type=int, help="first N hunks only (CPU latency runs)")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--throughput-batch", type=int, default=0, help="also time batched generation (cost)"
    )
    parser.add_argument("--reuse", action="store_true", help="keep predictions already saved")
    args = parser.parse_args()

    model, tokenizer = load(args.model, args.device, args.load_in_4bit)
    predictor = Predictor(model, tokenizer, args.device, args.full_guide)
    for split in args.splits:
        print(json.dumps(run_split(predictor, args, split)))


if __name__ == "__main__":
    main()
