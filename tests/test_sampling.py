import pytest

from hunkrisk.sampling import candidate_paths, interleave, risk_categories


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("database/migrations/2024_01_01_add_plan.php", ["schema"]),
        ("app/Http/Middleware/VerifyCsrfToken.php", ["auth"]),
        ("app/Http/Controllers/UserRoleController.php", ["auth"]),
        ("saleor/payment/gateways/stripe/webhooks.py", ["money"]),
        ("internal/tenant/cleanup.go", ["data"]),
        (".env.example", ["secrets"]),
        ("docs/syntax.md", []),
        ("AUTHORS.md", []),
    ],
)
def test_risk_categories(path, expected):
    assert risk_categories(path) == expected


def test_candidate_paths_use_the_directory_when_it_matches():
    files = [
        "database/migrations/2024_a.php",
        "database/migrations/2024_b.php",
        "app/Models/Invoice.php",
        "README.md",
    ]
    assert candidate_paths(files) == {
        "money": ["app/Models/Invoice.php"],
        "schema": ["database/migrations"],
    }


def test_interleave_alternates_groups_and_drops_repeats():
    busy = ["a1", "a2", "a3", "shared"]
    quiet = ["shared", "b1"]
    assert list(interleave([busy, quiet])) == ["a1", "shared", "a2", "b1", "a3"]
