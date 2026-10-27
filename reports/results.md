# Results

**Release bar: met.** The published adapter (Qwen3.5-2B, fine-tuned, 2× data) has recall 0.91, 100% valid JSON, 33× lower cost than the ceiling and F1 within 7.8 points of the ceiling (0.69 vs 0.77).

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

## Per-category F1

| System | auth | money | schema | data | secrets |
|---|---|---|---|---|---|
| Keyword/path heuristic | 0.47 | 0.44 | 0.63 | 0.21 | 0.10 |
| Qwen3-1.7B, zero-shot | 0.22 | 0.32 | 0.53 | 0.19 | 0.00 |
| Qwen3.5-2B, zero-shot | 0.32 | 0.10 | 0.40 | 0.06 | 0.00 |
| **Qwen3-1.7B, fine-tuned** | 0.49 | 0.17 | 0.72 | 0.40 | 0.20 |
| **Qwen3.5-2B, fine-tuned** | 0.55 | 0.26 | 0.70 | 0.44 | 0.43 |
| **Qwen3.5-2B, fine-tuned, 2× data** | 0.61 | 0.55 | 0.60 | 0.53 | 0.18 |
| Jev 1.14 (decision API) | 0.76 | 0.44 | 0.70 | 0.62 | 0.57 |
| DeepSeek V4.1 Flash (teacher) | 0.80 | 0.46 | 0.87 | 0.68 | 0.40 |
| Kimi K3.1 Pro (ceiling) | 0.75 | 0.52 | 0.83 | 0.69 | 0.25 |
| Claude Sonnet 5 | 0.89 | 0.91 | 0.91 | 0.71 | 0.60 |
| GPT-5.6 Terra | 0.88 | 0.71 | 0.92 | 0.86 | 0.44 |

## Recall by sampling stratum

Targeted hunks come from risk-looking paths, which the heuristic also keys on. Random hunks are where it has no head start.

| System | Targeted | Random |
|---|---|---|
| Keyword/path heuristic | 0.74 | 0.73 |
| Qwen3-1.7B, zero-shot | 0.94 | 0.95 |
| Qwen3.5-2B, zero-shot | 0.88 | 0.86 |
| **Qwen3-1.7B, fine-tuned** | 0.94 | 0.91 |
| **Qwen3.5-2B, fine-tuned** | 0.85 | 0.91 |
| **Qwen3.5-2B, fine-tuned, 2× data** | 0.91 | 0.91 |
| Jev 1.14 (decision API) | 0.71 | 0.73 |
| DeepSeek V4.1 Flash (teacher) | 0.74 | 0.86 |
| Kimi K3.1 Pro (ceiling) | 0.88 | 0.95 |
| Claude Sonnet 5 | 0.88 | 1.00 |
| GPT-5.6 Terra | 0.97 | 0.77 |

## Local models: threshold reference

Each threshold is the highest one that reaches 0.90 recall on validation. The first runs tuned it against the teacher's validation labels; it is now tuned against the panel's, which are made the same way as the test labels (see [decisions](../docs/decisions.md)).

| System | Threshold tuned on | Threshold | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Qwen3-1.7B, zero-shot | panel-labeled validation | 1.000 | 0.95 | 0.23 | 0.37 |
| Qwen3-1.7B, zero-shot | teacher-labeled validation | 1.000 | 0.95 | 0.23 | 0.37 |
| Qwen3.5-2B, zero-shot | panel-labeled validation | 0.053 | 0.88 | 0.24 | 0.38 |
| Qwen3.5-2B, zero-shot | teacher-labeled validation | 0.060 | 0.84 | 0.24 | 0.37 |
| **Qwen3-1.7B, fine-tuned** | panel-labeled validation | 0.060 | 0.93 | 0.33 | 0.49 |
| **Qwen3-1.7B, fine-tuned** | teacher-labeled validation | 0.076 | 0.86 | 0.35 | 0.49 |
| **Qwen3.5-2B, fine-tuned** | panel-labeled validation | 0.107 | 0.88 | 0.47 | 0.61 |
| **Qwen3.5-2B, fine-tuned** | teacher-labeled validation | 0.148 | 0.84 | 0.49 | 0.62 |
| **Qwen3.5-2B, fine-tuned, 2× data** | panel-labeled validation | 0.148 | 0.91 | 0.55 | 0.69 |
| **Qwen3.5-2B, fine-tuned, 2× data** | teacher-labeled validation | 0.107 | 0.93 | 0.53 | 0.68 |

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
| Qwen3-1.7B, zero-shot | 0.98 | 0.20 | 0.33 | 0.0% |
| Qwen3.5-2B, zero-shot | 0.18 | 0.59 | 0.27 | 98.7% |
| **Qwen3-1.7B, fine-tuned** | 0.48 | 0.64 | 0.55 | 100.0% |
| **Qwen3.5-2B, fine-tuned** | 0.55 | 0.74 | 0.63 | 100.0% |
| **Qwen3.5-2B, fine-tuned, 2× data** | 0.64 | 0.77 | 0.70 | 100.0% |

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
| **Qwen3-1.7B, fine-tuned** | panel-labeled validation | yes (0.93) | yes (100.0%) | no (F1 0.49 vs ≥ 0.67) | **no** |
| **Qwen3-1.7B, fine-tuned** | teacher-labeled validation | no (0.86) | yes (100.0%) | no (F1 0.49 vs ≥ 0.67) | **no** |
| **Qwen3.5-2B, fine-tuned** | panel-labeled validation | no (0.88) | yes (100.0%) | no (F1 0.61 vs ≥ 0.67) | **no** |
| **Qwen3.5-2B, fine-tuned** | teacher-labeled validation | no (0.84) | yes (100.0%) | no (F1 0.62 vs ≥ 0.67) | **no** |
| **Qwen3.5-2B, fine-tuned, 2× data** | panel-labeled validation | yes (0.91) | yes (100.0%) | yes (F1 0.69 vs ≥ 0.67) | **yes** |
| **Qwen3.5-2B, fine-tuned, 2× data** | teacher-labeled validation | yes (0.93) | yes (100.0%) | yes (F1 0.68 vs ≥ 0.67) | **yes** |

## Release bar: sensitivity to the tie-break

The two judges disagree on 23 hunks, and Mistral Large 4 settles them. The published adapter against the ceiling if they were settled another way:

| Disputed hunks settled by | `high` hunks | Adapter recall | Adapter F1 | Ceiling F1 | F1 gap (points) | Bar met |
|---|---|---|---|---|---|---|
| Mistral Large 4's vote (used) | 56 | 0.911 | 0.69 | 0.77 | 7.8 | yes |
| Claude Sonnet 5's label | 56 | 0.893 | 0.68 | 0.77 | 9.1 | no |
| GPT-5.6 Terra's label | 59 | 0.915 | 0.72 | 0.78 | 6.4 | yes |
| all `high` | 69 | 0.899 | 0.77 | 0.85 | 7.9 | no |
| all `low` | 46 | 0.913 | 0.61 | 0.68 | 7.4 | yes |
