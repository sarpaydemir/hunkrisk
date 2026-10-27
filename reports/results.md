# Results

**Release bar (provisional): not met.** The published adapter (Qwen3.5-2B, fine-tuned, 2× data) has recall 0.91, 100% valid JSON and 33× lower cost than the ceiling, but its F1 is 10.4 points below the ceiling's (0.71 vs 0.82), where the bar allows 10.

**Provisional:** 23 hunks where the two judges disagreed still need a human label and are left out. Scored on 277 test hunks from 5 held-out repositories (46 `high`). Local models use a P(high) threshold tuned on panel-labeled validation; API models are scored as returned. † Panel member: it helped write the answer key, so its scores are inflated.

| System | Recall (95% CI) | Precision | Precision at base rate | F1 (95% CI) | Valid JSON | Cost / 1k hunks | Latency (GPU) |
|---|---|---|---|---|---|---|---|
| Keyword/path heuristic | 0.78 (0.65–0.90) | 0.35 | 0.33 | 0.48 (0.37–0.57) | 100.0% | $0 | n/a |
| Qwen3-1.7B, zero-shot | 0.96 (0.88–1.00) | 0.21 | 0.20 | 0.35 (0.26–0.42) | 0.0% | $0.120 | 2.01 s |
| Qwen3.5-2B, zero-shot | 0.87 (0.76–0.96) | 0.22 | 0.20 | 0.35 (0.26–0.43) | 98.7% | $0.087 | 1.42 s |
| **Qwen3-1.7B, fine-tuned** | 0.93 (0.86–1.00) | 0.32 | 0.30 | 0.47 (0.37–0.56) | 100.0% | $0.044 | 2.71 s |
| **Qwen3.5-2B, fine-tuned** | 0.87 (0.76–0.96) | 0.48 | 0.45 | 0.62 (0.50–0.70) | 100.0% | $0.037 | 2.17 s |
| **Qwen3.5-2B, fine-tuned, 2× data** | 0.91 (0.83–1.00) | 0.58 | 0.56 | 0.71 (0.61–0.80) | 100.0% | $0.036 | 2.19 s |
| Jev 1.14 (decision API) | 0.80 (0.69–0.91) | 0.84 | 0.83 | 0.82 (0.73–0.90) | n/a | $0.088 | n/a |
| DeepSeek V4.1 Flash (teacher) | 0.85 (0.74–0.95) | 0.93 | 0.92 | 0.89 (0.81–0.95) | n/a | $0.025 | n/a |
| Kimi K3.1 Pro (ceiling) | 0.91 (0.83–0.98) | 0.74 | 0.72 | 0.82 (0.72–0.89) | n/a | $1.19 | n/a |
| Claude Sonnet 5 † | 1.00 (1.00–1.00) | 1.00 | 1.00 | 1.00 (1.00–1.00) | n/a | $1.78 | n/a |
| GPT-5.6 Terra † | 1.00 (1.00–1.00) | 1.00 | 1.00 | 1.00 (1.00–1.00) | n/a | $1.22 | n/a |

## Per-category F1

| System | auth | money | schema | data | secrets |
|---|---|---|---|---|---|
| Keyword/path heuristic | 0.47 | 0.48 | 0.65 | 0.20 | 0.11 |
| Qwen3-1.7B, zero-shot | 0.20 | 0.32 | 0.56 | 0.21 | 0.00 |
| Qwen3.5-2B, zero-shot | 0.32 | 0.12 | 0.43 | 0.08 | 0.00 |
| **Qwen3-1.7B, fine-tuned** | 0.56 | 0.21 | 0.70 | 0.40 | 0.33 |
| **Qwen3.5-2B, fine-tuned** | 0.56 | 0.32 | 0.71 | 0.49 | 0.60 |
| **Qwen3.5-2B, fine-tuned, 2× data** | 0.65 | 0.58 | 0.59 | 0.51 | 0.29 |
| Jev 1.14 (decision API) | 0.84 | 0.52 | 0.76 | 0.68 | 0.80 |
| DeepSeek V4.1 Flash (teacher) | 0.84 | 0.55 | 0.90 | 0.73 | 0.50 |
| Kimi K3.1 Pro (ceiling) | 0.77 | 0.62 | 0.85 | 0.68 | 0.40 |
| Claude Sonnet 5 | 0.96 | 0.93 | 0.95 | 0.72 | 0.86 |
| GPT-5.6 Terra | 0.94 | 0.86 | 0.95 | 1.00 | 0.67 |

## Recall by sampling stratum

Targeted hunks come from risk-looking paths, which the heuristic also keys on. Random hunks are where it has no head start.

| System | Targeted | Random |
|---|---|---|
| Keyword/path heuristic | 0.76 | 0.82 |
| Qwen3-1.7B, zero-shot | 0.93 | 1.00 |
| Qwen3.5-2B, zero-shot | 0.86 | 0.88 |
| **Qwen3-1.7B, fine-tuned** | 0.93 | 0.94 |
| **Qwen3.5-2B, fine-tuned** | 0.83 | 0.94 |
| **Qwen3.5-2B, fine-tuned, 2× data** | 0.90 | 0.94 |
| Jev 1.14 (decision API) | 0.76 | 0.88 |
| DeepSeek V4.1 Flash (teacher) | 0.76 | 1.00 |
| Kimi K3.1 Pro (ceiling) | 0.86 | 1.00 |
| Claude Sonnet 5 | 1.00 | 1.00 |
| GPT-5.6 Terra | 1.00 | 1.00 |

## Local models: threshold reference

Each threshold is the highest one that reaches 0.90 recall on validation. The first runs tuned it against the teacher's validation labels; it is now tuned against the panel's, which are made the same way as the test labels (see [decisions](../docs/decisions.md)).

| System | Threshold tuned on | Threshold | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Qwen3-1.7B, zero-shot | panel-labeled validation | 1.000 | 0.96 | 0.21 | 0.35 |
| Qwen3-1.7B, zero-shot | teacher-labeled validation | 1.000 | 0.96 | 0.21 | 0.35 |
| Qwen3.5-2B, zero-shot | panel-labeled validation | 0.053 | 0.87 | 0.22 | 0.35 |
| Qwen3.5-2B, zero-shot | teacher-labeled validation | 0.060 | 0.83 | 0.22 | 0.34 |
| **Qwen3-1.7B, fine-tuned** | panel-labeled validation | 0.060 | 0.93 | 0.32 | 0.47 |
| **Qwen3-1.7B, fine-tuned** | teacher-labeled validation | 0.076 | 0.85 | 0.33 | 0.48 |
| **Qwen3.5-2B, fine-tuned** | panel-labeled validation | 0.107 | 0.87 | 0.48 | 0.62 |
| **Qwen3.5-2B, fine-tuned** | teacher-labeled validation | 0.148 | 0.85 | 0.49 | 0.62 |
| **Qwen3.5-2B, fine-tuned, 2× data** | panel-labeled validation | 0.148 | 0.91 | 0.58 | 0.71 |
| **Qwen3.5-2B, fine-tuned, 2× data** | teacher-labeled validation | 0.107 | 0.93 | 0.55 | 0.69 |

## Local models: validation ranking

P(high) against panel-labeled validation, over all thresholds. The published adapter is the fine-tuned model with the highest average precision here.

| System | Average precision | ROC AUC |
|---|---|---|
| Qwen3-1.7B, zero-shot | 0.168 | 0.616 |
| Qwen3.5-2B, zero-shot | 0.421 | 0.811 |
| **Qwen3-1.7B, fine-tuned** | 0.654 | 0.891 |
| **Qwen3.5-2B, fine-tuned** | 0.708 | 0.914 |
| **Qwen3.5-2B, fine-tuned, 2× data** | 0.818 | 0.946 |

## Local models: plain decoding

The model's own `risk` field with no threshold, parsed leniently (a code fence is tolerated); the valid column is strict, as in the release bar.

| System | Recall | Precision | F1 | Valid JSON (strict) |
|---|---|---|---|---|
| Qwen3-1.7B, zero-shot | 0.98 | 0.18 | 0.30 | 0.0% |
| Qwen3.5-2B, zero-shot | 0.20 | 0.64 | 0.30 | 98.7% |
| **Qwen3-1.7B, fine-tuned** | 0.52 | 0.73 | 0.61 | 100.0% |
| **Qwen3.5-2B, fine-tuned** | 0.61 | 0.85 | 0.71 | 100.0% |
| **Qwen3.5-2B, fine-tuned, 2× data** | 0.67 | 0.84 | 0.75 | 100.0% |

## Local models: latency

Seconds per hunk with greedy decoding of the full answer, mean (p95). One hunk: one request at a time, as in interactive use. Batched: length-sorted batches of up to 16, which the cost column is based on. GPU: RTX 3060 Laptop (6 GB). CPU: the same laptop's Ryzen 7 5800H (8 cores), bf16, on the first test hunks; Qwen3.5's linear-attention layers run on PyTorch's reference implementation there, since the fast kernels are GPU-only.

| System | GPU, one hunk | GPU, batched | CPU, one hunk |
|---|---|---|---|
| Qwen3-1.7B, zero-shot | 2.01 s (2.67 s) | 1.08 s | not measured |
| Qwen3.5-2B, zero-shot | 1.42 s (1.74 s) | 0.78 s | not measured |
| **Qwen3-1.7B, fine-tuned** | 2.71 s (3.48 s) | 0.39 s | not measured |
| **Qwen3.5-2B, fine-tuned** | 2.17 s (2.77 s) | 0.33 s | not measured |
| **Qwen3.5-2B, fine-tuned, 2× data** | 2.19 s (2.80 s) | 0.33 s | 40.77 s (52.44 s), 20 hunks |

## Release bar

| System | Threshold tuned on | Recall ≥ 0.90 | Valid ≥ 98% | ≥ 10× cheaper at ≤ 10 pts lower F1 than the ceiling | Pass |
|---|---|---|---|---|---|
| **Qwen3-1.7B, fine-tuned** | panel-labeled validation | yes (0.93) | yes (100.0%) | no (F1 0.47 vs ≥ 0.72) | **no** |
| **Qwen3-1.7B, fine-tuned** | teacher-labeled validation | no (0.85) | yes (100.0%) | no (F1 0.48 vs ≥ 0.72) | **no** |
| **Qwen3.5-2B, fine-tuned** | panel-labeled validation | no (0.87) | yes (100.0%) | no (F1 0.62 vs ≥ 0.72) | **no** |
| **Qwen3.5-2B, fine-tuned** | teacher-labeled validation | no (0.85) | yes (100.0%) | no (F1 0.62 vs ≥ 0.72) | **no** |
| **Qwen3.5-2B, fine-tuned, 2× data** | panel-labeled validation | yes (0.91) | yes (100.0%) | no (F1 0.71 vs ≥ 0.72) | **no** |
| **Qwen3.5-2B, fine-tuned, 2× data** | teacher-labeled validation | yes (0.93) | yes (100.0%) | no (F1 0.69 vs ≥ 0.72) | **no** |
