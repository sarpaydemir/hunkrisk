"""VRAM and speed smoke test for a base-model candidate.

Loads the model with LoRA r=16 (4-bit unless --bf16), runs a few training steps on
full-length sequences, and prints peak VRAM and seconds per step as one JSON line.

    uv run python scripts/smoke_vram.py unsloth/Qwen3-1.7B-unsloth-bnb-4bit
    uv run python scripts/smoke_vram.py Qwen/Qwen3.5-2B --bf16
"""

import argparse
import json
import time

import torch
from unsloth import FastModel  # before transformers, so Unsloth's patches apply

DIFF = """--- a/app/Http/Controllers/InvoiceController.php
+++ b/app/Http/Controllers/InvoiceController.php
@@ -41,9 +41,6 @@ class InvoiceController extends Controller
     public function destroy(Request $request, Invoice $invoice)
     {
-        if ($invoice->user_id !== $request->user()->id) {
-            abort(403);
-        }
         $invoice->lines()->delete();
         $invoice->delete();
         return redirect()->route('invoices.index');
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--bf16", action="store_true", help="LoRA on bf16 weights instead of 4-bit")
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--steps", type=int, default=4)
    args = p.parse_args()

    torch.cuda.reset_peak_memory_stats()
    model, tok = FastModel.from_pretrained(
        args.model,
        max_seq_length=args.seq,
        load_in_4bit=not args.bf16,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )
    load_gb = torch.cuda.max_memory_allocated() / 2**30

    text_tok = getattr(tok, "tokenizer", tok)
    ids = text_tok(DIFF * 40, return_tensors="pt").input_ids[:, : args.seq].cuda()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-4)

    model.train()
    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)

    print(
        json.dumps(
            {
                "model": args.model,
                "quant": "bf16" if args.bf16 else "4bit",
                "seq": ids.shape[1],
                "load_gb": round(load_gb, 2),
                "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                "sec_per_step": round(sum(times[1:]) / max(len(times) - 1, 1), 2),
                "loss": round(loss.item(), 3),
            }
        )
    )


if __name__ == "__main__":
    main()
