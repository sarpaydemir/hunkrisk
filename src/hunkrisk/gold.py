"""Local labeling tool for the gold test set.

    uv run python -m hunkrisk.gold            # then open http://127.0.0.1:8765
    uv run python -m hunkrisk.gold --ids data/gold/disagreements.txt   # adjudication only

Blind by construction: the tool reads only the hunks file and the gold file, and shows the
file path and hunk, which is exactly what the model sees. Labels are appended to
data/gold/test.jsonl as they're saved; the latest entry per hunk wins, so relabeling is safe
and a session can stop and resume at any point.
"""

import argparse
import json
import random
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from hunkrisk.schema import InvalidOutput, validate_decision

PAGE = Path(__file__).with_name("gold.html")
ORDER_SEED = 13  # interleaves repositories so a session doesn't see one repo in a row


class GoldStore:
    def __init__(self, hunks_path: Path, gold_path: Path, ids: set[str] | None = None) -> None:
        records = [json.loads(line) for line in hunks_path.open()]
        if ids is not None:
            records = [r for r in records if r["id"] in ids]
        # Only what the model sees. Repo, commit subject and stratum stay hidden.
        self.items = [{"id": r["id"], "file": r["file"], "hunk": r["hunk"]} for r in records]
        self.items.sort(key=lambda item: item["id"])
        random.Random(ORDER_SEED).shuffle(self.items)
        self._ids = {item["id"] for item in self.items}
        self.path = gold_path
        self.labels: dict[str, dict[str, Any]] = {}
        if gold_path.exists():
            for line in gold_path.open():
                entry = json.loads(line)
                if entry["id"] in self._ids:
                    self.labels[entry["id"]] = entry

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("id") not in self._ids:
            raise InvalidOutput(f"unknown hunk id {payload.get('id')!r}")
        risk, categories = validate_decision(payload.get("risk"), payload.get("categories"))
        entry = {
            "id": payload["id"],
            "label": {"risk": risk, "categories": list(categories)},
            "note": str(payload.get("note") or "").strip(),
            "seconds": round(float(payload.get("seconds") or 0), 1),
            "labeled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.labels[entry["id"]] = entry
        return entry


def make_handler(store: GoldStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/items":
                labels = {i: e["label"] | {"note": e["note"]} for i, e in store.labels.items()}
                self._json(HTTPStatus.OK, {"items": store.items, "labels": labels})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/label":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                entry = store.save(payload)
            except (InvalidOutput, ValueError) as e:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            self._json(HTTPStatus.OK, entry)

        def _json(self, status: HTTPStatus, body: Any) -> None:
            self._send(status, json.dumps(body).encode(), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass  # keep the terminal quiet during a labeling session

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hunks", type=Path, default=Path("data/hunks/test.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/gold/test.jsonl"))
    parser.add_argument("--ids", type=Path, help="only label the hunk ids listed in this file")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    ids = set(args.ids.read_text().split()) if args.ids else None
    store = GoldStore(args.hunks, args.gold, ids)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"{len(store.labels)}/{len(store.items)} labeled. Open http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped; {len(store.labels)}/{len(store.items)} labeled in {args.gold}")


if __name__ == "__main__":
    main()
