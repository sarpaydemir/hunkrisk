"""Path heuristics for the targeted sampling stratum.

These only decide which commits to look at. Labels come from the teacher and the human,
never from these patterns.
"""

import re
from collections.abc import Iterable, Iterator
from itertools import chain, zip_longest

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _words(*words: str) -> re.Pattern[str]:
    # Match at a word start only, so "tax" doesn't fire on "syntax".
    return re.compile(r"(?<![a-z])(" + "|".join(words) + ")")


RISK_PATHS: dict[str, re.Pattern[str]] = {
    "auth": _words(
        "auth(?!or)", "oauth", "permission", "polic(?:y|ies)", "guard", "middleware", "session",
        "login", "rbac", "acl", "roles?(?![a-z])", "jwt", "saml", "sso(?![a-z])", "mfa", "totp",
        "two_factor", "csrf", "cors",
    ),
    "money": _words(
        "payment", "billing", "invoice", "checkout", "pric(?:e|ing)", "subscription", "stripe",
        "refund", "cart(?![a-z])", "tax", "discount", "coupon", "wallet", "ledger", "currenc",
        "payout", "order",
    ),
    "schema": re.compile(
        r"(?<![a-z])(migrations?|alembic|prisma)(?![a-z])|(?<![a-z])schema|\.sql$"
    ),
    "secrets": re.compile(
        r"(^|/)\.env|(?<![a-z])(secret|credential|encrypt|crypto|keyring|vault)"
        r"|(^|/)settings\.py$|(^|/)config/"
    ),
    "data": _words(
        "export", "backfill", "cleanup", "purge", "retention", "gdpr", "privacy", "tenant",
        "anonymi",
    ),
}  # fmt: skip


def normalize_path(path: str) -> str:
    """Lowercase, with camelCase split into snake_case so word-start matching works."""
    return _CAMEL_BOUNDARY.sub("_", path).lower()


def risk_categories(path: str) -> list[str]:
    normalized = normalize_path(path)
    return [category for category, pattern in RISK_PATHS.items() if pattern.search(normalized)]


def candidate_paths(files: list[str]) -> dict[str, list[str]]:
    """Group risk-looking paths by category, for commit listings filtered by path.

    When the directory itself matches (a migrations folder, an auth package), the directory is
    used instead of the file, so one listing covers every file in it.
    """
    grouped: dict[str, set[str]] = {category: set() for category in RISK_PATHS}
    for file in files:
        parent = file.rpartition("/")[0]
        for category in risk_categories(file):
            parent_matches = parent and RISK_PATHS[category].search(normalize_path(parent))
            grouped[category].add(parent if parent_matches else file)
    return {category: sorted(paths) for category, paths in grouped.items() if paths}


def interleave(groups: Iterable[list[str]]) -> Iterator[str]:
    """Round-robin across groups, dropping repeats, so a busy category can't crowd out the rest."""
    merged = chain.from_iterable(zip_longest(*groups))
    return iter(dict.fromkeys(item for item in merged if item is not None))
