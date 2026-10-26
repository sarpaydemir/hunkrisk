"""Compare label files against a reference, joined on hunk id.

    uv run python -m hunkrisk.compare data/labels/pilot/claude-sonnet-5.jsonl \\
        data/labels/pilot/deepseek-v4.1-flash.jsonl data/labels/pilot/qwen3.8-flash.jsonl

Precision and recall treat the reference as ground truth: the gold set for evaluation, or a
stronger model when choosing a teacher.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from hunkrisk.metrics import category_f1, risk_metrics
from hunkrisk.schema import CATEGORIES


def load_labels(path: Path) -> dict[str, dict[str, Any] | None]:
    """Map hunk id to its label, or None when the output was invalid."""
    return {r["id"]: r["label"] for r in map(json.loads, path.open())}


def compare(
    reference: dict[str, dict[str, Any] | None], candidate: dict[str, dict[str, Any] | None]
) -> dict[str, Any]:
    ids = sorted(i for i in reference.keys() & candidate.keys() if reference[i])
    valid = [i for i in ids if candidate[i]]
    ref = [reference[i] for i in valid]  # type: ignore[misc]
    pred = [candidate[i] for i in valid]  # type: ignore[misc]
    return {
        "invalid": len(ids) - len(valid),
        **risk_metrics([r["risk"] for r in ref], [p["risk"] for p in pred]),
        "categories": category_f1([r["categories"] for r in ref], [p["categories"] for p in pred]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()

    reference = load_labels(args.reference)
    high = sum(1 for label in reference.values() if label and label["risk"] == "high")
    print(f"reference: {args.reference.stem} ({len(reference)} hunks, {high} high)\n")
    header = ["candidate", "n", "invalid", "agree", "kappa", "P(high)", "R(high)", "F1(high)"]
    print("| " + " | ".join(header + list(CATEGORIES)) + " |")
    print("|" + "---|" * (len(header) + len(CATEGORIES)))
    for path in args.candidates:
        m = compare(reference, load_labels(path))
        cats = [f"{v:.2f}" if v is not None else "n/a" for v in m["categories"].values()]
        row = [path.stem, m["n"], m["invalid"], f"{m['agreement']:.2f}", f"{m['kappa']:.2f}"]
        row += [f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}", *cats]
        print("| " + " | ".join(map(str, row)) + " |")


if __name__ == "__main__":
    main()
