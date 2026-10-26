"""Splitting file patches into hunks, and the filters applied before labeling."""

import hashlib
import re
from dataclasses import dataclass

HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")

# Files whose diffs carry no reviewable logic: lockfiles, vendored or generated code,
# build output, translations, snapshots and binary-like assets.
EXCLUDED_PATH = re.compile(
    r"""
      (^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|composer\.lock|poetry\.lock|uv\.lock
           |Pipfile\.lock|go\.sum|Cargo\.lock|Gemfile\.lock|bun\.lockb?)$
    | (^|/)(vendor|node_modules|dist|build|third_party|\.yarn|__snapshots__|generated|mocks?)/
    | (^|/)(locales?|lang|i18n|translations)/
    | \.(min\.js|min\.css|map|snap|po|mo|ipynb|svg|png|jpe?g|gif|ico|webp|woff2?|ttf|eot|pdf|csv)$
    | (_pb2(_grpc)?\.pyi?|\.pb\.go|\.pb\.gw\.go|\.gen\.(go|ts)|\.generated\.\w+)$
    | (^|/)zz_generated[^/]*$
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True)
class Hunk:
    file: str
    index: int  # position within the file's patch
    text: str  # the @@ header line plus the hunk body


def split_patch(file: str, patch: str) -> list[Hunk]:
    """Split one file's unified-diff patch (as returned by the GitHub API) into hunks."""
    groups: list[list[str]] = []
    for line in patch.splitlines():
        if HUNK_HEADER.match(line):
            groups.append([line])
        elif groups:
            groups[-1].append(line)
    return [Hunk(file, i, "\n".join(lines)) for i, lines in enumerate(groups)]


def is_excluded(path: str) -> bool:
    return EXCLUDED_PATH.search(path) is not None


def changed_lines(hunk: Hunk) -> list[str]:
    return [line for line in hunk.text.splitlines()[1:] if line.startswith(("+", "-"))]


def has_meaningful_change(hunk: Hunk) -> bool:
    return any(line[1:].strip() for line in changed_lines(hunk))


def dedupe_key(hunk: Hunk) -> str:
    """Whitespace-insensitive fingerprint of the changed lines, ignoring path and context.

    Catches cherry-picks, backports and the same edit repeated across files.
    """
    normalized = "\n".join(line[0] + " ".join(line[1:].split()) for line in changed_lines(hunk))
    return hashlib.sha1(normalized.encode()).hexdigest()


def render(hunk: Hunk) -> str:
    """The text a model sees for one hunk: the file path, then the hunk."""
    return f"{hunk.file}\n{hunk.text}"
