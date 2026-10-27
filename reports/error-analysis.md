# Error analysis

Every test error of the first fine-tuned Qwen3.5-2B was read and grouped by hand. The model was fixed before its test errors were read; the follow-up model was already training when this analysis was written, so nothing here fed back into it. The follow-up model, trained on twice the data and the one that is published, is read the same way [further down](#the-follow-up-model). The 23 hunks the answer key's judges disagree on were settled after both readings, and the errors on them are read [last](#the-23-hunks-settled-by-a-third-judge).

- **Setting:** threshold 0.107 (the highest that reaches 0.90 recall on panel-labeled validation), scored on the 277 hunks the two judges agree on, 46 of them `high`.
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

## The follow-up model

The doubled-data model at its own threshold, 0.148 (0.90 recall on panel-labeled validation), on the same 277 hunks: 4 missed `high` hunks and 30 false alarms, down from 6 and 44. To reproduce: `uv run python -m hunkrisk.errors data/predictions/ft-qwen3.5-2b-x2 --threshold 0.14805`.

- **Misses.** It fixes three of the first model's six: both that under-weighted an effect on data (`dc0640d89e02`, `ee7a5d630c68`) and the debatable label (`7a57e9a4d27d`). The two misreadings and the missed guide rule remain (`683ad3f30608`, `b8b2e66fe9f1`, `1b94e7a1f43d`). One miss is new: `73dcda5459ce` (BookStack) adds a relation from attachments to the permission table, which both the model and the teacher call a read-only relationship. Whether it matters depends on where it's used, and the hunk doesn't show that.
- **False alarms.** 27 of the 30 were false alarms for the first model too, and 3 are new. Grouped as before:

| Kind | First model | Follow-up |
|---|---|---|
| CI, build and local dev setup | 11 | 6 |
| Harmless edits inside sensitive code | 7 | 6 |
| Tests that add coverage in sensitive areas | 7 | 2 |
| API surface without behavior change | 7 | 7 |
| New code outside the five categories | 6 | 2 |
| Config and validation for non-sensitive settings | 4 | 4 |
| Security wording or docs, not security behavior | 2 | 3 |

- **More data fixed tests, CI and new code, not deprecations.** Six of the seven API-surface false alarms deprecate a field or remove a deprecated one. Three of them are the model's own `high` answer, with reasons like "changing which data can be exported" for a deprecation notice. A deprecation changes nothing until the field is removed, and the training data doesn't teach that.
- **The threshold still accounts for most false alarms.** In 24 of the 30, the model's own answer was `low`. The other 6 are its own `high` answers, against 5 for the first model.
- **The ranking improved more than the operating point shows.** Average precision on validation rose from 0.71 to 0.82, and plain decoding (the model's own answers) reaches F1 0.75 on test, against 0.71 at the 0.90-recall threshold. Contrasting examples of harmless changes in sensitive code remain the most direct next step.

## The 23 hunks settled by a third judge

The two judges disagree on 23 test hunks, and a third model settles them by majority (see the [decision log](../docs/decisions.md)). With them, the first model has 7 missed `high` hunks and 55 false alarms on all 300; the follow-up has 5 and 41.

- **Both models make the same 12 mistakes here:** one miss and 11 false alarms. The ceiling, Kimi K3.1 Pro, makes as many, and nine of its false alarms are on the same hunks. Two strong models split on each of these hunks, so their labels are 2-to-1 votes, the least certain in the set.
- **The miss:** `7a2e872ab6cf` (Polar) renames an emitted billing event type from `meter_credited` to `meter.credited`. Anything that stores or matches the old string breaks. Both models call it "only a string literal", the same kind of miss as the under-weighted data effects above. The teacher said `high`.
- **The false alarms,** grouped as before:

| Kind | Count | Example |
|---|---|---|
| Harmless edits inside sensitive code | 5 | truncating a Stripe coupon name to the API's 40-character limit (`fbe28f3c8bdc`) |
| CI, build and local dev setup | 2 | a dev script's secrets-file helper, removed and replaced (`909b7e41545f`, `c0f006d4b1a9`) |
| Tests in sensitive areas | 2 | permission fixtures moved out of a shared fixtures file (`e1e18b3bf10f`) |
| Security docs, not security behavior | 1 | the docblock and return type of an OIDC method (`f68f0b8d19bd`) |
| A non-unique index | 1 | an index on `views.updated_at` (`4d6458ac80a8`) |

- **Several are defensible calls.** The labeling guide lists indexes under schema, but this one can't lose data, and two of the three judges said `low`. Five of the 11 are the follow-up model's own `high` answer, and the teacher said `high` on five as well.
