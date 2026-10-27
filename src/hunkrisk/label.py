"""Label hunks with an API model by following the labeling guide.

    uv run python -m hunkrisk.label --model deepseek/deepseek-v4.1-flash \\
        --input data/hunks/val.jsonl --sample 50 --max-usd 0.05
    uv run python -m hunkrisk.label --model typesafe/jev-1.14 --api decisions \\
        --input data/hunks/test.jsonl --max-usd 0.05
    uv run python -m hunkrisk.label --model mistralai/mistral-large-4-0 \\
        --input data/hunks/test.jsonl --ids data/gold/disagreements.txt --max-usd 0.30

The cost estimate is printed first, and the run refuses to start when the estimate exceeds
--max-usd or the API key's remaining limit.
"""

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from hunkrisk.decisions import DecisionClient, questions, to_label
from hunkrisk.llm import BudgetExceeded, Client, key_remaining_usd, model_price
from hunkrisk.prompt import messages, prompt_version, user_message
from hunkrisk.schema import CATEGORIES, InvalidOutput, parse_label

COMPLETION_TOKENS = 200  # a label is ~60 tokens; leaves room without paying for rambling
WORKERS = 8


def balanced_sample(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Draw n records spread evenly across repositories, reproducibly."""
    rng = random.Random(seed)
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_repo[record["repo"]].append(record)
    queues = [rng.sample(group, len(group)) for _, group in sorted(by_repo.items())]
    picked: list[dict[str, Any]] = []
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    return picked


def label_one(client: Client, record: dict[str, Any]) -> dict[str, Any]:
    """Label one hunk, retrying once with the validation error when the reply is invalid."""
    convo = messages(record["file"], record["hunk"])
    result: dict[str, Any] = {"id": record["id"], "model": client.model}
    result |= {"prompt_version": prompt_version(), "label": None, "error": None}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "new_spend": 0.0}
    for _ in range(2):
        reply = client.complete(convo, max_tokens=COMPLETION_TOKENS)
        for field in ("prompt_tokens", "completion_tokens", "cost"):
            usage[field] += reply[field]
        usage["new_spend"] += 0.0 if reply["cached"] else reply["cost"]
        try:
            result["label"] = parse_label(reply["text"], strict=False).to_json()
            result["error"] = None
            break
        except InvalidOutput as e:
            result["error"], result["raw"] = str(e), reply["text"]
            convo = [
                *convo,
                {"role": "assistant", "content": reply["text"]},
                {"role": "user", "content": f"Invalid: {e}. Reply with only the JSON object."},
            ]
    return result | usage


def decide_one(client: DecisionClient, record: dict[str, Any]) -> dict[str, Any]:
    """Label one hunk with a decision model; probabilities are kept for threshold analysis."""
    reply = client.decide(record["file"], record["hunk"])
    return {
        "id": record["id"],
        "model": reply["resolved_model"],
        "prompt_version": prompt_version(),
        "label": to_label(reply["answers"]),
        "error": None,
        "p_high": reply["answers"]["high_risk"]["noul"],
        "p_categories": {c: reply["answers"][c]["noul"] for c in CATEGORIES},
        "prompt_tokens": reply["prompt_tokens"],
        "completion_tokens": reply["completion_tokens"],
        "cost": reply["cost"],
        "new_spend": 0.0 if reply["cached"] else reply["cost"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--api",
        choices=["chat", "decisions"],
        default="chat",
        help="decisions for decision models such as typesafe/jev-1.14",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default: data/labels/<model>/<input name>")
    parser.add_argument("--ids", type=Path, help="label only the hunk ids listed in this file")
    parser.add_argument("--sample", type=int, help="label a balanced sample of this size")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/llm"))
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.open()]
    if args.ids:
        wanted = set(args.ids.read_text().split())
        records = [r for r in records if r["id"] in wanted]
    if args.sample:
        records = balanced_sample(records, args.sample, args.seed)
    out = args.out or Path("data/labels") / args.model.split("/")[-1] / args.input.name

    client: Client | DecisionClient
    if args.api == "decisions":
        client = DecisionClient(args.model, Path("data/cache/decisions"), args.max_usd)
        price = model_price(args.model)
        fixed = len(json.dumps(questions()))
        chars = sum(fixed + len(user_message(r["file"], r["hunk"])) for r in records)
        estimate = chars / 3.2 * price.prompt * 1.15  # output tokens are free
        label: Callable[[Any, dict[str, Any]], dict[str, Any]] = decide_one
    else:
        client = Client(args.model, args.cache, args.max_usd)
        chars = sum(sum(len(m["content"]) for m in messages(r["file"], r["hunk"])) for r in records)
        estimate = client.estimate_usd(chars, len(records), COMPLETION_TOKENS)
        label = label_one
    remaining = key_remaining_usd()
    print(f"{len(records)} hunks with {args.model}: estimated ≤ ${estimate:.4f} "
          f"(cap ${args.max_usd}, key has ${remaining} left)")  # fmt: skip
    if estimate > args.max_usd or (remaining is not None and estimate > remaining):
        raise SystemExit("estimate exceeds the budget; nothing was sent")

    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(label, client, r) for r in records]
        results = []
        for future in tqdm(futures, desc=args.model.split("/")[-1], unit="hunk"):
            try:
                results.append(future.result())
            except BudgetExceeded as e:
                raise SystemExit(f"stopped: {e}") from e

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    valid = [r for r in results if r["label"]]
    high = sum(r["label"]["risk"] == "high" for r in valid)
    cost = sum(r["cost"] for r in results)
    spent = sum(r["new_spend"] for r in results)
    print(
        f"wrote {out}: {len(valid)}/{len(results)} valid, {high} high; "
        f"labeling cost ${cost:.4f}, newly spent ${spent:.4f}"
    )


if __name__ == "__main__":
    main()
