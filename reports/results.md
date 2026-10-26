# Results

**Provisional:** 23 hunks where the two judges disagreed still need a human label and are left out. Scored on 277 test hunks from 5 held-out repositories (46 `high`). Local models use a P(high) threshold tuned on validation; API models are scored as returned. † Panel member: it helped write the answer key, so its scores are inflated.

| System | Recall (95% CI) | Precision | Precision at base rate | F1 (95% CI) | Valid JSON | Cost / 1k hunks | Latency (GPU) |
|---|---|---|---|---|---|---|---|
| Keyword/path heuristic | 0.78 (0.65–0.90) | 0.35 | 0.33 | 0.48 (0.37–0.57) | 100.0% | $0 | n/a |
| Jev 1.14 (decision API) | 0.80 (0.69–0.91) | 0.84 | 0.83 | 0.82 (0.73–0.90) | n/a | $0.088 | n/a |
| DeepSeek V4.1 Flash (teacher) | 0.85 (0.74–0.95) | 0.93 | 0.92 | 0.89 (0.81–0.95) | n/a | $0.025 | n/a |
| Kimi K3.1 Pro (ceiling) | 0.91 (0.83–0.98) | 0.74 | 0.72 | 0.82 (0.72–0.89) | n/a | $1.19 | n/a |
| Claude Sonnet 5 † | 1.00 (1.00–1.00) | 1.00 | 1.00 | 1.00 (1.00–1.00) | n/a | $1.78 | n/a |
| GPT-5.6 Terra † | 1.00 (1.00–1.00) | 1.00 | 1.00 | 1.00 (1.00–1.00) | n/a | $1.22 | n/a |

## Per-category F1

| System | auth | money | schema | data | secrets |
|---|---|---|---|---|---|
| Keyword/path heuristic | 0.47 | 0.48 | 0.65 | 0.20 | 0.11 |
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
| Jev 1.14 (decision API) | 0.76 | 0.88 |
| DeepSeek V4.1 Flash (teacher) | 0.76 | 1.00 |
| Kimi K3.1 Pro (ceiling) | 0.86 | 1.00 |
| Claude Sonnet 5 | 1.00 | 1.00 |
| GPT-5.6 Terra | 1.00 | 1.00 |

## Local models: threshold vs plain decoding

Thresholds are tuned on validation to reach 0.90 recall.

| System | Threshold | Recall as generated | Valid JSON |
|---|---|---|---|

## Release bar

| System | Recall ≥ 0.90 | Valid ≥ 98% | ≥ 10× cheaper at ≤ 10 pts lower F1 than the ceiling | Pass |
|---|---|---|---|---|
