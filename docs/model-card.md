---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- lora
- code-review
- diff
- risk-classification
---

# hunkrisk: Qwen3.5-2B LoRA

A LoRA adapter for [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) that reads one diff hunk and decides whether a reviewer must read it closely before it ships. It names the risk categories it touches (auth, money, schema, data, secrets) and gives a one-sentence reason.

```json
{"risk": "high", "categories": ["auth"], "reason": "Removes the invoice-owner authorization check, so any authenticated user can now delete any invoice."}
```

Code, data pipeline and the full evaluation: [github.com/sarpaydemir/hunkrisk](https://github.com/sarpaydemir/hunkrisk).

## Results

**Provisional:** scored on 277 of the 300 test hunks, from five held-out repositories (46 `high`). The other 23 are hunks where the two judges of the answer key disagreed, and they are waiting for a human label. The [results page](https://github.com/sarpaydemir/hunkrisk/blob/main/reports/results.md) has the current numbers.

| System | Recall | Precision | F1 | Cost / 1k hunks |
|---|---|---|---|---|
| **This adapter**, P(high) ≥ 0.148 | 0.91 | 0.58 | 0.71 | $0.036 |
| This adapter, its own answer | 0.67 | 0.84 | 0.75 | $0.036 |
| Kimi K3.1 Pro (ceiling) | 0.91 | 0.74 | 0.82 | $1.19 |
| DeepSeek V4.1 Flash (teacher) | 0.85 | 0.93 | 0.89 | $0.025 |
| Keyword/path heuristic | 0.78 | 0.35 | 0.48 | $0 |

- **Threshold:** 0.148 is the highest that reaches 0.90 recall on validation, frozen before test.
- **Validation ranking:** average precision 0.82 and ROC AUC 0.95 for `high`, on 300 hunks from five other repositories.
- **Validity:** 100% valid JSON with unconstrained decoding.
- **Latency:** 2.19 s per hunk on an RTX 3060 Laptop GPU one at a time, 0.33 s batched. On the same laptop's CPU (Ryzen 7 5800H, bf16) it takes about 41 s, because Qwen3.5's linear-attention layers fall back to PyTorch's reference implementation without a GPU.
- **Cost:** batched GPU time at an assumed $0.40 per GPU-hour.
- **[Release bar](https://github.com/sarpaydemir/hunkrisk/blob/main/docs/release-bar.md), pre-registered:** not met on the provisional answer key. Recall and validity pass, and the adapter is 33× cheaper than the ceiling, but its F1 is 10.4 points below the ceiling's, where the bar allows 10.

## How to use

The input is the file path, a newline, then the hunk, under a short system prompt. Thinking is off. `risk` is the first field of the answer, so P(`high`) is one forward pass away.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

repo = "sarpaydemir/hunkrisk-qwen3.5-2b-lora"
THRESHOLD = 0.148  # 0.90 recall on validation; 0.5 follows the model's own answer

# Qwen3.5 is multimodal. AutoModelForCausalLM builds a text-only variant with different module
# names, and the adapter would silently fail to attach to it.
base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-2B", dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, repo).eval()
tokenizer = AutoTokenizer.from_pretrained(repo)

SYSTEM = (
    "Classify the diff hunk for code review risk. Reply with one JSON object: "
    '{"risk": "high" or "low", "categories": [...], "reason": "<one sentence>"}. '
    'Categories: auth, money, schema, data, secrets; use ["none"] when risk is low.'
)
file = "app/Http/Controllers/InvoiceController.php"
hunk = """@@ -40,9 +40,6 @@ public function destroy(Request $request, Invoice $invoice)
-        if ($invoice->user_id !== $request->user()->id) {
-            abort(403);
-        }
         $invoice->delete();"""

messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{file}\n{hunk}"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)

# P(high): score the risk token right after the answer prefix.
high, low = (tokenizer.encode(t, add_special_tokens=False)[0] for t in ("high", "low"))
inputs = tokenizer(prompt + '{"risk": "', return_tensors="pt", add_special_tokens=False)
with torch.inference_mode():
    logits = model(**inputs.to(model.device)).logits[0, -1]
p_high = torch.softmax(logits[[high, low]].float(), dim=0)[0].item()
print(f"P(high) = {p_high:.3f}, flag = {p_high >= THRESHOLD}")

# The full answer, with categories and reason.
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))
```

The evaluation ran with Unsloth's kernels, so scores from plain transformers can differ in the second decimal. From the repository, `git diff | uv run python -m hunkrisk.predict --model sarpaydemir/hunkrisk-qwen3.5-2b-lora` scores every hunk of a diff.

## Training

Trained on 5,735 hunks from 20 repositories, labeled by the teacher (18% `high`), and validated on 300 hunks from five other repositories. It ran 718 steps (two epochs) in 62 minutes. Validation loss was still falling at the end (0.588), so the published weights are the last checkpoint.

- **Method:** 16-bit LoRA (Unsloth advises against 4-bit training on Qwen3.5), r = 16, α = 16, no dropout, on the attention, linear-attention and MLP projections of the language model. The vision tower is frozen and unused.
- **Optimization:** AdamW 8-bit, learning rate 2e-4 with cosine decay and 3% warmup, effective batch 16, up to 2 epochs, bf16. Loss on the answer only.
- **Selection:** validation loss every 40 steps, early stopping with patience 3, best checkpoint kept.
- **Hardware:** one RTX 3060 Laptop GPU (6 GB).

## Data

Single hunks from 30 permissively licensed open-source applications (Laravel, Django, FastAPI, Go, TypeScript), split by repository into train (20 repositories), validation (5) and test (5). Train labels come from DeepSeek V4.1 Flash (MIT license) following a written [labeling guide](https://github.com/sarpaydemir/hunkrisk/blob/main/docs/labeling-guide.md). Test labels come from two frontier models from different companies, with a human settling their disagreements. About 18% of hunks are `high`. Details are in the [dataset card](https://github.com/sarpaydemir/hunkrisk/blob/main/docs/dataset-card.md).

## Limitations

- **One hunk, no surrounding code.** The model judges what the hunk shows. Some risk only appears with context from elsewhere in the file or codebase.
- **It confuses sensitive areas with risky changes.** At the 0.90-recall threshold, most false alarms are harmless edits in auth or payment code: tests, refactors, deprecations, CI. See the [error analysis](https://github.com/sarpaydemir/hunkrisk/blob/main/reports/error-analysis.md).
- **Learned from a teacher model.** The labels it was trained on come from an API model and carry that model's mistakes.
- **Five held-out repositories.** Test results come with wide confidence intervals and may not transfer to other stacks or domains.
- **Triage, not approval.** A `low` answer is not a statement that a change is safe.

## License

Apache-2.0, like the base model.
