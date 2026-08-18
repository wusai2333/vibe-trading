# src/factors/zoo/ — ~460 FACTOR MODULES

One file per alpha: `__alpha_meta__` dict literal + pure `compute(panel)`. gtja191/alpha101/qlib158 are machine-templated (contract frozen for porting agents); academic/ is hand-authored; fundamental/, limit/, session/, vol/ are smaller themed zoos.

## ADDING A FACTOR
1. Create `<zoo>/<short>.py` (token `^[a-z][a-z0-9_]{0,31}$`, file ≤ 200KB). Template: qlib158/beta5.py.
2. `__alpha_meta__` dict literal: `id="<zoo>_<short>"`, `theme` from fixed set (momentum/reversal/volume/volatility/quality/value/liquidity/microstructure/sentiment/growth/leverage), `formula_latex`, `columns_required` (price set {open,high,low,close,volume,vwap,amount} or `fund:*`), optional `extras_required`/`requires_sector`, `universe`, `frequency`, `decay_horizon` 0–512, `min_warmup_bars`, `notes`.
3. `compute(panel: dict[str, DataFrame]) -> DataFrame` — pure, wide output same shape as `panel["close"]`; warmup NaN ok, ±inf rejected, >95% NaN rejected.
4. Golden CSV in tests/factors/fixtures/goldens/ + parametrize entry in the zoo's sample test.
5. Validate: `pytest agent/tests/factors/test_alpha_purity.py agent/tests/factors/test_registry.py agent/tests/factors/test_lookahead.py -q` + zoo sample test.

## PURITY GATE (tests/factors/test_alpha_purity.py — AST-enforced)
- Imports allowlist ONLY: pandas, numpy, scipy.*, src.factors.base, __future__, typing, math, dataclasses.
- Forbidden names anywhere: os, sys, subprocess, socket, urllib, requests, httpx, aiohttp, pathlib, Path, open, eval, exec, compile, `__import__`, breakpoint, input, help, memoryview, globals, locals, vars, dunder ladders (`__class__`/`__base__`/`__mro__`/`__globals__`/`__builtins__`).
- Module level: only imports, function defs, `ALPHA_ID`, `__alpha_meta__`, docstring — no classes, no top-level calls, no `if`.
- Lookahead ban (test_lookahead.py): no negative shifts; `delta(df, d)` requires d ≥ 1.

## ANTI-PATTERNS
- F401 is ruff-ignored here BY DESIGN (templated verbatim import blocks); F841 stays active to catch formula bugs — do not "clean up" imports in generated files.
- Registry never imports modules to read meta; keep `__alpha_meta__` a plain literal (no computed values, no tuples-of-sets).
- SkipAlpha is the sanctioned way to bail on unmet preconditions (sector/columns) — raise it, don't return NaNs silently.
