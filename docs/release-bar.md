# Release bar

Pre-registered on 2026-10-26, before any training run, so the targets can't be fitted to the results. Any later change is logged in [`decisions.md`](decisions.md) and called out in the README.

Everything is measured on the gold test set: about 300 hunks from held-out repositories, labeled blind by a human.

## Pass criteria

All three must hold.

1. **Recall on `high` ≥ 0.90.** A missed risky hunk is the expensive error. The decision threshold on P(`high`) is chosen on the validation set and applied unchanged to test.
2. **Valid output ≥ 98%.** Measured with unconstrained decoding (no grammar or JSON mode). An output is valid when it:
   - parses as JSON
   - has `risk` ∈ {`high`, `low`}
   - has a non-empty `categories` drawn from the allowed set, where `none` appears only alone and only with `low`, and `high` names at least one category
   - has a non-empty one-sentence `reason`
3. **≥ 10× cheaper than the API ceiling model, at most 10 points lower F1.** F1 here is the F1 of the `high` class.
   - API cost: the provider's list price at run time × measured tokens.
   - Local cost: measured GPU-seconds per 1,000 hunks on the reference GPU × $0.40/hour, an assumed rate for a comparable rented GPU.

## Reported regardless of outcome

- Precision on `high`, as measured and reweighted to the natural base rate (estimated from randomly sampled hunks)
- Per-category F1
- Latency per hunk on GPU and CPU
- Bootstrap 95% confidence intervals for recall and F1
- An error taxonomy built by reading every test error

A miss is published the same way as a pass, together with the error analysis.
