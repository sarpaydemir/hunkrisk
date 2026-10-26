"""Dump a system's test errors with full context, for reading and grouping by hand.

    uv run python -m hunkrisk.errors data/predictions/ft-qwen3.5-2b --threshold 0.31

Writes data/errors/<name>.md: every missed `high` hunk and every false alarm, with the hunk,
the gold label and its source, and the model's own answer and P(high).
"""

import argparse
import json
from pathlib import Path

from hunkrisk.panel import load_gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", type=Path, help="folder holding test.jsonl")
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    gold = load_gold()
    hunks = {
        r["id"]: r for r in map(json.loads, Path("data/hunks/test.jsonl").read_text().splitlines())
    }
    preds = {
        r["id"]: r
        for r in map(json.loads, (args.predictions / "test.jsonl").read_text().splitlines())
    }

    sections = {
        "Missed (gold high, predicted low)": [],
        "False alarms (gold low, predicted high)": [],
    }
    for i, g in sorted(gold.items()):
        predicted_high = preds[i]["p_high"] >= args.threshold
        if (g["risk"] == "high") == predicted_high:
            continue
        key = next(iter(sections)) if g["risk"] == "high" else list(sections)[1]
        h, p = hunks[i], preds[i]
        sections[key].append(
            f"### {i} · {h['repo']} · {h['stratum']}\n\n"
            f"`{h['file']}`, commit: {h['subject']}\n\n"
            f"- gold: {g['risk']} {g['categories']} ({g['source']})\n"
            f"- model: P(high)={p['p_high']:.3f}, answer: {p['raw']}\n\n"
            f"```diff\n{h['hunk']}\n```\n"
        )

    out = Path("data/errors") / f"{args.predictions.name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"## {title} ({len(items)})\n\n" + "\n".join(items) for title, items in sections.items()
    ]
    out.write_text(f"# Errors: {args.predictions.name}\n\n" + "\n".join(body))
    print(
        f"{out}: " + ", ".join(f"{len(v)} {k.split(' (')[0].lower()}" for k, v in sections.items())
    )


if __name__ == "__main__":
    main()
