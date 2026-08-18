# src/live/ — LIVE-TRADING GOVERNANCE

Safety-critical. Every behavior that looks like a bug here is documented as deliberate in module docstrings — read them BEFORE editing. Changes need targeted regression tests (test_sdk_order_gate.py, test_mandate_enforcement.py, test_killswitch_blocks_orders.py, test_readonly_default.py).

## WHERE TO LOOK
- mandate.py — trading-mandate enforcement (30 importers).
- order_guard.py / sdk_order_gate.py — order gates for API path and direct-SDK connectors.
- audit.py — dual-ledger audit trail.
- halt.py — kill switch.
- classification.py — order classification (19 importers).

## DELIBERATE BEHAVIORS (do NOT "fix")
- All gate checks fail-closed BEFORE the broker call; DENY is the default on any uncertainty.
- Daily order count increments only on confirmed non-error ALLOW.
- `repeatable = False`: a live order must NEVER be silently re-issued.
- Notional enforced as max(explicit, qty×price); DENY when unpriceable.
- audit.py: classic audit.jsonl written+fsynced FIRST, chained copy second; chain-append failure is deliberately swallowed (crashing an order over a logging fault is worse). Two files intentional — never retrofit chain format onto audit.jsonl.
- governance/ledger.py (sibling package): append verifies the ENTIRE chain (O(n) deliberate); raises LedgerCorruptionError rather than extend a broken chain; caller payloads must not set seq/prev_record_hash/record_hash.

## ANTI-PATTERNS
- Never open a live order path on structurally paper-only connectors (shoonya, dhan).
- Broker-write changes must remain mandate-gated, kill-switch-aware, fail-closed, audit-logged — all four.
