import pytest

from hunkrisk.hunks import Hunk, dedupe_key, has_meaningful_change, is_excluded, split_patch

PATCH = """\
@@ -1,3 +1,4 @@
 import os
+import sys

 def main():
@@ -10,2 +11,2 @@ def main():
-    return 1
+    return 0"""


def test_split_patch_yields_one_hunk_per_header():
    hunks = split_patch("app.py", PATCH)
    assert [h.index for h in hunks] == [0, 1]
    assert hunks[0].text.splitlines()[0] == "@@ -1,3 +1,4 @@"
    assert hunks[1].text.splitlines() == [
        "@@ -10,2 +11,2 @@ def main():",
        "-    return 1",
        "+    return 0",
    ]


def test_split_patch_ignores_lines_before_first_header():
    assert split_patch("a.py", "Binary files differ") == []


def test_split_patch_keeps_single_line_hunk_headers():
    hunks = split_patch("a.py", "@@ -1 +1 @@\n-a\n+b")
    assert len(hunks) == 1


@pytest.mark.parametrize(
    "path",
    [
        "package-lock.json",
        "web/yarn.lock",
        "go.sum",
        "vendor/github.com/x/y.go",
        "static/app.min.js",
        "api/v1/user_pb2.py",
        "locale/de/LC_MESSAGES/django.po",
        "resources/lang/en/auth.php",
        "src/__snapshots__/form.test.ts.snap",
        "pkg/apis/v1/zz_generated.deepcopy.go",
    ],
)
def test_excludes_lockfiles_generated_and_vendored(path):
    assert is_excluded(path)


@pytest.mark.parametrize(
    "path",
    [
        "database/migrations/2024_01_01_000000_add_plan_to_accounts.php",
        "app/Http/Middleware/Authenticate.php",
        "saleor/payment/gateway.py",
        "internal/auth/jwt.go",
        ".env.example",
        "src/builder.ts",
    ],
)
def test_keeps_source_files(path):
    assert not is_excluded(path)


def test_whitespace_only_change_is_not_meaningful():
    assert not has_meaningful_change(Hunk("a.py", 0, "@@ -1 +1 @@\n-    \n+"))
    assert has_meaningful_change(Hunk("a.py", 0, "@@ -1 +1 @@\n-x = 1\n+x = 2"))


def test_context_lines_are_not_changes():
    assert not has_meaningful_change(Hunk("a.py", 0, "@@ -1,2 +1,2 @@\n context\n another"))


def test_dedupe_key_ignores_path_context_and_whitespace():
    a = Hunk("a.py", 0, "@@ -1,2 +1,2 @@\n context\n-x = 1\n+x  =  2")
    b = Hunk("lib/b.py", 3, "@@ -9,2 +9,2 @@\n other\n-x = 1\n+x = 2")
    assert dedupe_key(a) == dedupe_key(b)


def test_dedupe_key_distinguishes_additions_from_removals():
    added = Hunk("a.py", 0, "@@ -1 +1 @@\n+x = 1")
    removed = Hunk("a.py", 0, "@@ -1 +1 @@\n-x = 1")
    assert dedupe_key(added) != dedupe_key(removed)
