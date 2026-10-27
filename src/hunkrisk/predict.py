"""Score each hunk of a diff for review risk.

    git diff main... | uv run python -m hunkrisk.predict \\
        --model sarpaydemir/hunkrisk-qwen3.5-2b-lora

Prints one JSON object per hunk: the file, the hunk header, P(high), the decision at the threshold
and the model's own answer with its reason. Lockfiles, generated and vendored files are skipped,
as they were in the dataset.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from hunkrisk.evaluate import Predictor, load
from hunkrisk.hunks import is_excluded, split_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("diff", nargs="?", type=Path, help="a unified diff; default: stdin")
    parser.add_argument("--model", required=True, help="adapter directory or Hugging Face repo id")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="flag hunks with P(high) at or above this; 0.5 follows the model's own answer",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true", help="for QLoRA-trained adapters")
    args = parser.parse_args()

    diff = args.diff.read_text() if args.diff else sys.stdin.read()
    hunks = [h for h in split_diff(diff) if not is_excluded(h.file)]
    if not hunks:
        parser.exit(message="no reviewable hunks in the diff\n")

    model_dir = args.model if Path(args.model).exists() else snapshot_download(args.model)
    with contextlib.redirect_stdout(sys.stderr):  # Unsloth prints a banner; keep stdout JSON
        model, tokenizer = load(model_dir, args.device, args.load_in_4bit)
    predictor = Predictor(model, tokenizer, args.device, full_guide=False)
    for hunk in hunks:
        p = predictor.predict(hunk.file, hunk.text, max_new_tokens=160)
        result = {
            "file": hunk.file,
            "hunk": hunk.text.splitlines()[0],
            "risk": "high" if p["p_high"] >= args.threshold else "low",
            "p_high": p["p_high"],
            "answer": p["label"] or p["raw"],
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
