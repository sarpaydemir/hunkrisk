"""Minimal GitHub REST client with an on-disk response cache.

The cache makes reruns free and keeps them reproducible: a dataset rebuilt from the same
cache is byte-identical.
"""

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

API_URL = "https://api.github.com"
_LAST_PAGE = re.compile(r'<[^>]*[?&]page=(\d+)[^>]*>;\s*rel="last"')


def last_page(link_header: str | None) -> int:
    """Number of the last page in a paginated response, from its Link header."""
    match = _LAST_PAGE.search(link_header or "")
    return int(match.group(1)) if match else 1


def _token() -> str:
    if token := os.environ.get("GITHUB_TOKEN"):
        return token
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


class GitHub:
    def __init__(self, cache_dir: Path, token: str | None = None) -> None:
        self.cache_dir = cache_dir
        self.client = httpx.Client(
            base_url=API_URL,
            timeout=30,
            headers={
                "Authorization": f"Bearer {token or _token()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def get(self, endpoint: str, /, **params: Any) -> tuple[Any, str | None]:
        """GET an API endpoint. Returns the JSON body and the Link header.

        `endpoint` is positional-only because `path` is itself a GitHub query parameter.
        """
        key = hashlib.sha1(json.dumps([endpoint, sorted(params.items())]).encode()).hexdigest()
        cached = self.cache_dir / key[:2] / f"{key}.json"
        if cached.exists():
            entry = json.loads(cached.read_text())
        else:
            response = self._request(endpoint, params)
            entry = {"endpoint": endpoint, "params": params, "body": response.json()}
            entry["link"] = response.headers.get("link")
            cached.parent.mkdir(parents=True, exist_ok=True)
            partial = cached.with_suffix(".partial")
            partial.write_text(json.dumps(entry))
            partial.replace(cached)  # atomic, so an interrupted run never leaves a bad entry
        return entry["body"], entry["link"]

    def _request(self, endpoint: str, params: dict[str, Any], attempts: int = 6) -> httpx.Response:
        for attempt in range(attempts):
            response = self.client.get(endpoint, params=params)
            if response.status_code == 200:
                return response
            if response.status_code in (403, 429) and _is_rate_limited(response):
                time.sleep(_rate_limit_wait(response, attempt))
            elif response.status_code >= 500:
                time.sleep(2**attempt)
            else:
                response.raise_for_status()
        raise RuntimeError(f"GET {endpoint} failed after {attempts} attempts")


def _is_rate_limited(response: httpx.Response) -> bool:
    return (
        response.headers.get("x-ratelimit-remaining") == "0"
        or "retry-after" in response.headers
        or "rate limit" in response.text.lower()
    )


def _rate_limit_wait(response: httpx.Response, attempt: int) -> float:
    if retry_after := response.headers.get("retry-after"):
        return float(retry_after)
    if response.headers.get("x-ratelimit-remaining") == "0":
        return max(0.0, float(response.headers["x-ratelimit-reset"]) - time.time()) + 1
    return 60.0 * (attempt + 1)  # secondary limit without guidance: back off linearly
