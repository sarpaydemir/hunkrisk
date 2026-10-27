# How it was built

I built this project by directing a coding agent, Claude Code, rather than writing every line myself. This file covers how that worked: the rules the agent worked under, which decisions were mine, and where measuring something changed the plan. The full chronological log is in [`docs/decisions.md`](docs/decisions.md).

## Setup

- **The task is my own review rule:** money, auth, schema and data changes get read first. I picked it because I can defend every labeling call myself.
- **The agent works under [`CLAUDE.md`](CLAUDE.md),** whose invariants exist to keep the evaluation honest:
  - split by repository
  - blind gold labels
  - a pre-registered release bar
  - permissive licenses only
  - a logged, reproducible run directory for every training run
  - a spend cap on every paid API call
- **Paid API calls had a hard budget,** first $2 and later $7, enforced as a limit on the API key. Every labeling run prints its cost estimate first, refuses to start above `--max-usd`, and is trialed on 50 hunks before a full run. The whole project spent $2.71 on APIs.
- **Hardware:** one laptop GPU, an RTX 3060 with 6 GB.

## Decisions I made

- **The task, the five categories and the rule** behind them.
- **A release bar set before training,** with recall as the headline metric, because a missed risky hunk is the expensive error.
- **Budget limits.** The project has to be cheap enough to run on a personal key.
- **Replacing manual test labeling with a model panel.** Two frontier models from different companies label the test set. I planned to settle their 23 disagreements by hand, then gave them to a third model from another company instead, under a rule committed before it ran. The README says the answer key has no human label in it.
- **Pushing back on a recommendation.** I asked to try TypeSafe's Jev decision model as the labeler. Measuring it showed it was too inaccurate for an answer key, but it became a baseline, so the results include a purpose-built decision API.
- **Scrutinizing judge choices.** I asked why GLM-5.3 and Grok 4.7 weren't judges. Comparing them showed both force reasoning, billed as output, so their cost couldn't be predicted or capped.
- **Publishing:** the GitHub repository stayed private until I had reviewed it, and the adapter went public on Hugging Face.

## Where measurement changed the plan

- **The base rate was 18% `high`, not 30%.** Path-targeted sampling found less risk than assumed. I kept the test set as collected rather than let a model pick test hunks, which would bias it toward what models already catch, and report bootstrap confidence intervals instead.
- **The ceiling model moved.** Claude Sonnet 5 was meant to be the quality ceiling, but became a judge on the answer-key panel, and a judge scores perfectly on everything the panel agreed on. Before any fine-tuned results existed, the ceiling moved to Kimi K3.1 Pro, from a company with no part in the answer key.
- **Forced reasoning ruled out models.** On OpenRouter, Claude Sonnet 5.5, GLM-5.3, Grok 4.7 and Gemini 3.8 Flash all force reasoning, which is billed as output. Every model that labels or judges runs with reasoning off, the same footing as the small model being evaluated.
- **Provider prices vary 10×.** The same model costs very different amounts depending on which provider serves it; DeepSeek V4.1 Flash ranges from $0.048 to $0.60 per million input tokens. Every request now asks for the cheapest provider and records which one served it.
- **The training recipe changed.** Unsloth advises against 4-bit training on Qwen3.5, so it trains with 16-bit LoRA. A smoke test confirmed that fits in 6 GB (4.9 GB peak).
- **A plausible explanation turned out wrong.** The first fine-tuned model missed the recall target, and the obvious suspect was tuning its threshold on the teacher's labels. Labeling validation with the same panel as test showed the teacher finds 35 of the panel's 36 `high` hunks, so that wasn't it. The panel-labeled reference was kept because it matches how test is labeled, and the decision log records that the hypothesis failed.
- **One follow-up iteration, with the selection rule fixed first.** Doubling the training data was the only change, so the comparison isolates data size. Which adapter gets published was decided by a validation metric, written down while the follow-up was still training.
- **A verdict that hinges on one hunk is reported as one.** Settling the disagreements turned "missed by 0.4 points" into "met". But on those 23 hunks the adapter and the ceiling make the same number of mistakes, and the gap closed only because the same errors cost a more precise model more F1. Recall sits one hunk above the line. The report now shows the verdict under every other way of settling them.
- **Checking the published usage code caught a silent bug.** Loading the adapter the standard way, through `AutoPeftModelForCausalLM`, builds a text-only Qwen3.5 whose module names don't match, so the adapter silently fails to attach. Comparing P(high) against the evaluation's numbers exposed it before the model card shipped.

## Working style

- **Small, logical commits**, each passing ruff and the test suite. 95 unit tests cover the pure logic; nothing needs a network or a GPU.
- **Reproducibility.** The data collector caches every API response. Rebuilding the dataset from the cache gave byte-identical files, which I checked before relying on it.
- **Smoke tests before long runs:** collection on one repository, a 6-step training run, a 50-hunk labeling trial for every new model.
- **Kept honest.** The README reports every system, every change to the original plan is logged, and the results stayed marked provisional until every test hunk had a label.
