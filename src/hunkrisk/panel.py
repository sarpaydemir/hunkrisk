"""Build the gold test labels from two independent judges, and a third for their disagreements.

    uv run python -m hunkrisk.panel data/labels/test/claude-sonnet-5.jsonl \\
        data/labels/test/gpt-5.6-terra.jsonl --tiebreak data/labels/test/mistral-large-4-0.jsonl
    uv run python -m hunkrisk.panel <val judge a> <val judge b> --out data/labels/val-panel

Where the two judges agree on risk, that's the gold label; when both say `high`, the categories
are the union of theirs. Where they disagree, a third judge from another company labels the hunk
blind, and the majority of the three decides. A human label from the gold labeling tool always
wins. Without --tiebreak (validation, for threshold tuning), disagreements are left out.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from hunkrisk.compare import load_labels
from hunkrisk.schema import ALLOWED_CATEGORIES

PANEL_PATH = Path("data/gold/panel.jsonl")
TIEBREAK_PATH = Path("data/gold/tiebreak.jsonl")
HUMAN_PATH = Path("data/gold/test.jsonl")
DISAGREEMENTS_PATH = Path("data/gold/disagreements.txt")
GOLD_SOURCES = {"panel": PANEL_PATH, "tiebreak": TIEBREAK_PATH, "human": HUMAN_PATH}


def majority(labels: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """The judges' majority on risk, or None on a tie. Missing labels don't vote.

    A `high` majority takes the union of the categories of the judges that said `high`.
    """
    votes = [label for label in labels if label]
    high = [label for label in votes if label["risk"] == "high"]
    if 2 * len(high) == len(votes):
        return None
    if 2 * len(high) < len(votes):
        return {"risk": "low", "categories": ["none"]}
    union = set().union(*(label["categories"] for label in high))
    return {"risk": "high", "categories": [c for c in ALLOWED_CATEGORIES if c in union]}


def merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    """The agreed label of two judges, or None when they disagree on risk."""
    return majority([a, b])


def load_gold(sources: dict[str, Path] = GOLD_SOURCES) -> dict[str, dict[str, Any]]:
    """Gold label per hunk id, with its source; later sources win. Unsettled hunks are absent."""
    gold: dict[str, dict[str, Any]] = {}
    for source, path in sources.items():
        if path.exists():
            for entry in map(json.loads, path.open()):  # later lines win: relabels are appended
                gold[entry["id"]] = entry["label"] | {"source": source}
    return gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("judges", type=Path, nargs=2)
    parser.add_argument("--tiebreak", type=Path, help="a third judge's labels for disagreements")
    parser.add_argument("--out", type=Path, default=PANEL_PATH.parent, help="default: data/gold")
    args = parser.parse_args()

    first, second = (load_labels(path) for path in args.judges)
    agreed, disagreements = [], []
    for hunk_id in sorted(first.keys() & second.keys()):
        a, b = first[hunk_id], second[hunk_id]
        label = merge(a, b) if a and b else None
        if label:
            agreed.append({"id": hunk_id, "label": label, "judges": [p.stem for p in args.judges]})
        else:
            disagreements.append(hunk_id)

    panel_path = args.out / PANEL_PATH.name
    disagreements_path = args.out / DISAGREEMENTS_PATH.name
    args.out.mkdir(parents=True, exist_ok=True)
    panel_path.write_text("".join(json.dumps(e) + "\n" for e in agreed))
    disagreements_path.write_text("".join(f"{i}\n" for i in disagreements))
    high = sum(e["label"]["risk"] == "high" for e in agreed)
    print(
        f"{len(agreed)} agreed ({high} high), {len(disagreements)} disagreements "
        f"-> {panel_path}, {disagreements_path}"
    )

    if args.tiebreak:
        third = load_labels(args.tiebreak)
        judges = [p.stem for p in (*args.judges, args.tiebreak)]
        settled = []
        for hunk_id in disagreements:
            label = majority([first[hunk_id], second[hunk_id], third.get(hunk_id)])
            if label:
                settled.append({"id": hunk_id, "label": label, "judges": judges})
        tiebreak_path = args.out / TIEBREAK_PATH.name
        tiebreak_path.write_text("".join(json.dumps(e) + "\n" for e in settled))
        high = sum(e["label"]["risk"] == "high" for e in settled)
        print(f"{len(settled)} settled by {args.tiebreak.stem} ({high} high) -> {tiebreak_path}")


if __name__ == "__main__":
    main()
