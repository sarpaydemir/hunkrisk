# Error analysis

Every test error of the first fine-tuned Qwen3.5-2B was read and grouped by hand. The model was fixed before its test errors were read; the follow-up model was already training when this analysis was written, so nothing here fed back into it.

- **Setting:** threshold 0.107 (the highest that reaches 0.90 recall on panel-labeled validation), scored on the 277 provisional gold hunks, 46 of them `high`.
- **Errors:** 6 missed `high` hunks and 44 false alarms.
- **To reproduce the dump** (hunk, gold label, model answer and P(high) for each error): `uv run python -m hunkrisk.errors data/predictions/ft-qwen3.5-2b --threshold 0.10669`. IDs below are hunk IDs from `data/hunks/test.jsonl`.

## Summary

1. **The false alarms come from the threshold, not the model's judgment.** In 39 of the 44, the model's own answer was `low` with a correct reason. They're flagged only because 0.90 recall needs a threshold of 0.107, so any hunk in sensitive code clears it, even when the change itself is harmless. The teacher called only 2 of the 44 `high`, so the student didn't learn them from its labels.
2. **False alarms cluster where a whole repository is sensitive.** Authelia, an authentication server, has 4 `high` hunks and 10 false alarms. Saleor, a commerce platform, has 10 and 12. In those codebases, almost every path looks like auth or money.
3. **Two of the six misses are misreadings of the diff.** The model's reason describes a change that isn't the one in the hunk. Two more saw the change but under-weighted what it does to data. One misses an explicit guide rule, and one has a debatable gold label.
4. **Two of the six misses were inherited from the teacher,** which labeled them `low` too.

## Missed `high` hunks (6)

| Kind | Count | Hunks |
|---|---|---|
| Misread the diff | 2 | `683ad3f30608`, `b8b2e66fe9f1` |
| Saw the change, under-weighted its effect on data | 2 | `dc0640d89e02`, `ee7a5d630c68` |
| Missed an explicit guide rule | 1 | `1b94e7a1f43d` |
| Debatable gold label | 1 | `7a57e9a4d27d` |

- **`683ad3f30608` (BookStack):** removes `'restricted'` from a model's `$hidden` list, so the attribute now appears in serialized output. The model's reason: "removes a leading comma, a formatting-only change". P(high) 0.009.
- **`b8b2e66fe9f1` (BookStack):** adds a live `iframe_sources` setting whose default allows YouTube, Vimeo and draw.io in the content security policy. The model read the comment lines above it and called it "a commented configuration entry". The teacher made the same call.
- **`dc0640d89e02` (Payload):** comments out the call that writes uploads to local storage, so uploaded files are silently not stored. The model noticed but judged it "only affects file upload behavior".
- **`ee7a5d630c68` (BookStack):** changes which pages a ZIP export includes. A mistake there leaks drafts. The model called it "display/export-only".
- **`1b94e7a1f43d` (Saleor):** deletes a test that checks webhook query validation. The guide makes deleting a test that covers auth, money or data behavior `high`; the model applied the general test-only rule instead.
- **`7a57e9a4d27d` (Polar):** changes the copy and states shown after an automated payment-access review. The guide lists UI copy as `low`, but both judges called it `high`. Agents don't edit gold labels, so it stays, and it's flagged here. The teacher said `low` as well.

## False alarms (44)

| Kind | Count | Example |
|---|---|---|
| CI, build and local dev setup | 11 | `.env.template` with secret-named variables and no values (`7a657e23b5d6`) |
| Harmless edits inside sensitive code | 7 | removing a `# type: ignore` next to a payment gateway call (`1983e5da4214`) |
| Tests that add coverage in sensitive areas | 7 | a new OIDC login test (`508b65c07588`) |
| API surface without behavior change | 7 | a deprecation notice on a GraphQL field (`9f1ac48e9418`) |
| New code outside the five categories | 6 | a gzip middleware (`e3b1267ca424`) |
| Config and validation for non-sensitive settings | 4 | a privacy-policy URL validator (`3d27a0a60691`) |
| Security wording, not security behavior | 2 | reflowed text in a one-time-code email (`e0d6bb5594bc`) |

Only 5 false alarms were the model's own `high` answer:

- deleting a CI packaging script that mentions system users (`9d3264f9368d`)
- moving where a dev script finds its secrets file (`ade056de1709`)
- deprecating export mutations (`c07f58b412ac`)
- removing fields from seed data (`c368b4644061`)
- the one-time-code email above (`e0d6bb5594bc`)

All of these touch the right vocabulary without changing risky behavior.

## What this suggests

- **The ranking is better than the operating point.** Test AUC is 0.94 and the plain decoding is precise, but recall at 0.90 costs half the precision. Better separation between "sensitive area" and "risky change" matters more than a better threshold.
- **Training data in sensitive areas needs harmless contrasts.** Tests, refactors, deprecations and CI changes in auth or payment code, labeled `low`, are the confusions above. The follow-up run doubles the data, which adds more of them. Picking such hunks on purpose would be the next step.
- **One hunk isn't always enough.** `683ad3f30608` needs to know what `$hidden` does, and `dc0640d89e02` needs to know that nothing else stores the upload. A model with a few lines of surrounding file context, or the file's other hunks, would see more of the effect.
