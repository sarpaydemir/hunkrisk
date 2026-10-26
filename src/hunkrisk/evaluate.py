"""Run a local model (base or fine-tuned) over a split and record its predictions.

    uv run python -m hunkrisk.evaluate --model runs/<run>/adapter --split test --name ft-qwen3.5-2b
    uv run python -m hunkrisk.evaluate --model Qwen/Qwen3.5-2B --full-guide --split test \\
        --name qwen3.5-2b-zeroshot

Per hunk it stores the raw generation, the strictly parsed label (or the validation error),
P(high) and latency. P(high) is read from the generation's own logits at the risk token, so it
costs nothing extra; only when a model wanders off the format is a forced-prefix pass used.
"""

import argparse
import json
import statistics
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
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer

        is_adapter = (Path(model_name) / "adapter_config.json").exists()
        loader = AutoPeftModelForCausalLM if is_adapter else AutoModelForCausalLM
        model = loader.from_pretrained(model_name, dtype=torch.bfloat16).eval()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="hub id or a LoRA adapter directory")
    parser.add_argument("--name", required=True, help="output folder under data/predictions")
    parser.add_argument("--split", default="test")
    parser.add_argument("--full-guide", action="store_true", help="zero-shot with the full guide")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--limit", type=int, help="first N hunks only (CPU latency runs)")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    records = [
        json.loads(x) for x in Path(f"data/hunks/{args.split}.jsonl").read_text().splitlines()
    ]
    records = records[: args.limit] if args.limit else records
    model, tokenizer = load(args.model, args.device, args.load_in_4bit)
    predictor = Predictor(model, tokenizer, args.device, args.full_guide)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = []
    for r in tqdm(records, desc=f"{args.name}/{args.split}", unit="hunk"):
        rows.append({"id": r["id"]} | predictor.predict(r["file"], r["hunk"], args.max_new_tokens))

    suffix = f"-{args.device}" if args.device != "cuda" else ""
    out = Path("data/predictions") / args.name / f"{args.split}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    latencies = sorted(r["latency_s"] for r in rows)
    summary = {
        "model": args.model,
        "split": args.split,
        "device": args.device,
        "full_guide": args.full_guide,
        "n": len(rows),
        "valid": sum(r["label"] is not None for r in rows) / len(rows),
        "p_from_generation": sum(r["p_source"] == "generation" for r in rows) / len(rows),
        "latency_mean_s": round(statistics.mean(latencies), 4),
        "latency_p50_s": round(latencies[len(latencies) // 2], 4),
        "latency_p95_s": round(latencies[int(len(latencies) * 0.95) - 1], 4),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)
        if args.device == "cuda"
        else None,
    }
    out.with_name(f"{args.split}{suffix}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
