# Labeling guide

This file is the source for the teacher prompt and the reference for the human who labels the gold test set. Both must apply the same rules.

## Task

Given one diff hunk (file path plus unified diff), decide whether a reviewer must read it closely before it ships.

The rule: changes to **money, auth, schema, data or secrets** are reviewed first. A hunk is `high` when it **changes behavior** in one of these areas and a mistake could cost money, weaken access control, corrupt or lose data, or expose a secret. Touching a sensitive file isn't enough on its own; the change has to matter.

When torn between `high` and `low`, choose `high`. A missed risky hunk costs more than an extra look.

## Output

```json
{"risk": "high", "categories": ["auth"], "reason": "Removes the ownership check before deleting an invoice."}
```

- `categories`: every category the change affects, or exactly `["none"]` when `risk` is `low`.
- `reason`: one sentence naming the concrete behavior change. For `low`, say why it's safe.

## Categories

| Category | `high` when the hunk changes… | Examples |
|---|---|---|
| `auth` | who can do what: authentication, authorization, sessions, roles, permissions, policies, guards, CSRF/CORS, password or token handling | dropping a `@login_required`, editing a policy check, changing token expiry, exposing a new public route |
| `money` | amounts, prices, currency, rounding, discounts, taxes, invoices, refunds, payouts, subscriptions, payment provider calls | changing rounding in a total, a new discount path, payment webhook handling |
| `schema` | the persisted data structure: database migrations, models, columns, indexes, constraints | adding a non-null column, dropping a table, changing a unique index |
| `data` | what data is written, deleted or exposed: bulk updates or deletes, backfills, data migrations, retention jobs, tenant or user scoping in queries, personal data handling, exports | removing a `where tenant_id = ?`, a cleanup job that deletes rows, logging user emails |
| `secrets` | credentials and keys: API keys, passwords, private keys, `.env` files, secret configuration, encryption keys, secrets reaching logs | a committed key, disabling TLS verification, printing a token |
| `none` | nothing above; used alone and only with `low` | |

A hunk can belong to several categories. A migration that backfills rows is `schema` and `data`.

## Usually `low`, even in sensitive files

- Test-only changes. **Exception:** deleting or weakening a test that covers auth, money or data behavior is `high`.
- Comments, docs, formatting, imports, type hints, pure renames
- Logging and metrics, unless they log secrets or personal data
- UI copy, styling and layout
- Dependency version changes in manifests

## Scope

- **`schema` means persisted data only.** API contract changes (OpenAPI, GraphQL, protobuf) are a compatibility risk, not this one.
- **CI, Docker and infrastructure files** get no category of their own. They're `high` only through an existing one, such as a secret in a workflow (`secrets`) or an auth check disabled in config (`auth`).
- **Judge only what the hunk shows.** Use the file path and the visible lines; don't assume code outside the hunk.

## Examples

`app/Http/Controllers/InvoiceController.php`

```diff
-        if ($invoice->user_id !== $request->user()->id) {
-            abort(403);
-        }
         $invoice->delete();
```

→ `high`, `["auth", "money"]`: removes the ownership check, so any user can delete any invoice.

`migrations/0042_add_plan_to_account.py`

```diff
+        migrations.AddField("account", "plan", models.CharField(max_length=20, default="free")),
+        migrations.RunPython(assign_legacy_plans),
```

→ `high`, `["schema", "data"]`: adds a column and backfills it for existing accounts.

`tests/test_checkout.py`

```diff
+@pytest.fixture
+def eur_cart(cart):
+    cart.currency = "EUR"
+    return cart
```

→ `low`, `["none"]`: test-only change that adds a fixture.

`internal/server/middleware.go`

```diff
-	log.Printf("request %s %s", r.Method, r.URL.Path)
+	slog.Info("request", "method", r.Method, "path", r.URL.Path)
```

→ `low`, `["none"]`: switches to structured logging without changing what is logged.
