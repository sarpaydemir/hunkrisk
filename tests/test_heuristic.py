import pytest

from hunkrisk.heuristic import classify


@pytest.mark.parametrize(
    ("file", "hunk", "risk", "categories"),
    [
        ("app/Http/Controllers/InvoiceController.php", "@@ -1 +1 @@\n-        abort(403);", "high",
         ["money"]),
        ("src/views/home.tsx", "@@ -1 +1 @@\n-<h1>Hi</h1>\n+<h1>Hello</h1>", "low", ["none"]),
        ("tests/test_auth.py", "@@ -1 +1 @@\n+    assert login(user)", "low", ["none"]),
        ("app/models.py", "@@ -1 +1 @@\n+    api_key = 'abc'", "high", ["secrets"]),
        ("db/schema.go", "@@ -1 +1 @@\n+// ALTER TABLE users ADD COLUMN plan", "high", ["schema"]),
    ],
)  # fmt: skip
def test_classify(file, hunk, risk, categories):
    label = classify(file, hunk)
    assert (label["risk"], label["categories"]) == (risk, categories)


def test_context_lines_do_not_trigger_keywords():
    assert (
        classify("app/a.py", "@@ -1,2 +1,2 @@\n password = get()\n-x = 1\n+x = 2")["risk"] == "low"
    )
