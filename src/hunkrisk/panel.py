"""Build the gold test labels from two independent judges plus human adjudication.

    uv run python -m hunkrisk.panel data/labels/test/claude-sonnet-5.jsonl \\
        data/labels/test/gpt-5.6-terra.jsonl
    uv run python -m hunkrisk.gold --ids data/gold/disagreements.txt
    uv run python -m hunkrisk.panel <val judge a> <val judge b> --out data/labels/val-panel

Where the judges agree on risk, that's the gold label; when both say `high`, the categories are
the union of theirs. Where they disagree, a human decides, blind to both judges, using the gold
labeling tool. Human labels always win. With --out, the same merge builds a reference for
another split (validation, for threshold tuning), where disagreements are simply left out.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from hunkrisk.compare import load_labels
from hunkrisk.schema import ALLOWED_CATEGORIES

PANEL_PATH = Path("data/gold/panel.jsonl")
HUMAN_PATH = Path("data/gold/test.jsonl")
DISAGREEMENTS_PATH = Path("data/gold/disagreements.txt")


def merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    """The agreed label of two judges, or None when they disagree on risk."""
    if a["risk"] != b["risk"]:
        return None
    if a["risk"] == "low":
        return {"risk": "low", "categories": ["none"]}
    union = set(a["categories"]) | set(b["categories"])
    return {"risk": "high", "categories": [c for c in ALLOWED_CATEGORIES if c in union]}


def load_gold(
    panel_path: Path = PANEL_PATH, human_path: Path = HUMAN_PATH
) -> dict[str, dict[str, Any]]:
    """Gold label per hunk id, with its source. Disagreements without a human label are absent."""
    gold: dict[str, dict[str, Any]] = {}
    if panel_path.exists():
        for entry in map(json.loads, panel_path.open()):
            gold[entry["id"]] = entry["label"] | {"source": "panel"}
    if human_path.exists():
        for entry in map(json.loads, human_path.open()):  # later lines win: relabels are appended
            gold[entry["id"]] = entry["label"] | {"source": "human"}
    return gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("judges", type=Path, nargs=2)
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


if __name__ == "__main__":
    main()
