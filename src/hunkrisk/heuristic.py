"""Keyword and path heuristic: the deterministic floor that every model has to beat.

    uv run python -m hunkrisk.heuristic --split test

A hunk is `high` when its path or its changed lines match a category's keywords. Test, doc and
fixture files are always `low`, as a careful rule-writer would do.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from hunkrisk.hunks import Hunk, changed_lines
from hunkrisk.sampling import normalize_path, risk_categories
from hunkrisk.schema import CATEGORIES

CONTENT: dict[str, re.Pattern[str]] = {
    "auth": re.compile(
        r"\b(auth\w*|login|logout|password|permission\w*|authori[sz]\w*|roles?|polic(y|ies)"
        r"|guard\w*|csrf|cors|session\w*|jwt|oauth\w*|is_?admin|login_required|middleware)\b",
        re.IGNORECASE,
    ),
    "money": re.compile(
        r"\b(price\w*|amount|total|currency|payment\w*|invoice\w*|refund\w*|discount\w*|tax\w*"
        r"|billing|charge\w*|stripe|subscription\w*|payout\w*|cents)\b",
        re.IGNORECASE,
    ),
    "schema": re.compile(
        r"\b(migration\w*|create_table|add_column|addColumn|dropColumn|drop_table|alter table"
        r"|create table|drop table|Schema::|AddField|RemoveField|AlterField|ForeignKey)\b",
        re.IGNORECASE,
    ),
    "data": re.compile(
        r"(\bdelete from\b|\bupdate \w+ set\b|\bbulk_(update|create|delete)\b|\btruncate\b"
        r"|\bpurge\b|\.delete\(|->delete\(|\.destroy\(|\btenant\w*|\bbackfill\w*|\bgdpr\b"
        r"|\banonymi[sz]\w*)",
        re.IGNORECASE,
    ),
    "secrets": re.compile(
        r"\b(secret\w*|api_?key|private_?key|passwd|credential\w*|access_?key|client_secret"
        r"|verify\s*=\s*False|InsecureSkipVerify|rejectUnauthorized)\b",
        re.IGNORECASE,
    ),
}

SAFE_PATH = re.compile(
    r"(^|/)(tests?|__tests__|spec|fixtures?|docs?|examples?)/|_test\.\w+$|\.(test|spec)\.\w+$"
    r"|(^|/)test_[^/]+$|\.(md|rst|txt)$"
)


def classify(file: str, hunk_text: str) -> dict[str, Any]:
    if SAFE_PATH.search(normalize_path(file)):
        return {"risk": "low", "categories": ["none"], "reason": "test, doc or fixture file"}
    changed = "\n".join(line[1:] for line in changed_lines(Hunk(file, 0, hunk_text)))
    found = set(risk_categories(file)) | {c for c, rx in CONTENT.items() if rx.search(changed)}
    if not found:
        return {"risk": "low", "categories": ["none"], "reason": "no risk keyword"}
    categories = [c for c in CATEGORIES if c in found]
    return {"risk": "high", "categories": categories, "reason": "keyword match"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("data/predictions/heuristic"))
    args = parser.parse_args()

    records = [
        json.loads(line) for line in Path(f"data/hunks/{args.split}.jsonl").read_text().splitlines()
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [{"id": r["id"], "label": classify(r["file"], r["hunk"])} for r in records]
    (args.out / f"{args.split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    high = sum(r["label"]["risk"] == "high" for r in rows)
    print(f"{args.split}: {high}/{len(rows)} high -> {args.out / args.split}.jsonl")


if __name__ == "__main__":
    main()
