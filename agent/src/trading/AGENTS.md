# src/trading/ — BROKER CONNECTORS

Per-broker adapters under connectors/ (60 files, 77 importers of the package). All writes flow through src/live/ gates.

## STRUCTURE
- connectors/<broker>/sdk.py — one package per broker (ibkr, longbridge, robinhood, shoonya, dhan, mt5, ...). Optional deps via pyproject extras (ibkr, longbridge, mt5, krx...).
- profiles, service, tap_forward — connector selection/profile machinery.

## CONVENTIONS
- Every connector write must be: mandate-gated, kill-switch-aware, fail-closed, audit-logged.
- Lazy-import heavy broker SDKs; degrade with an actionable install hint when the extra is absent (mt5 pattern: Windows-only marker, no-op elsewhere).
- Live-broker config (robinhood/ibkr) is gated in src/config/schema.py — wildcard broker names are rejected.

## ANTI-PATTERNS
- shoonya/dhan are STRUCTURALLY paper-only: no runtime paper/live discriminator exists and the live path must never be opened.
- Do not add retry logic to order submission — the no-retry stance is policy (see src/live/order_guard.py).
- mt5 loader/connector is Windows-only by design; don't force it on other platforms.
