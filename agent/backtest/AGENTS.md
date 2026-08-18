# backtest/ — BACKTEST ENGINE

Imported as top-level `backtest.*` (package-dir remap). Generated strategy code runs sandboxed via src/core/runner.py.

## STRUCTURE
```
backtest/
├── runner.py       # entry: config boundary validation + execution orchestration
├── engines/        # per-market engines: china_a, crypto, forex, global_equity, india, korea, options_portfolio
│   └── _market_hooks.py  # THE symbol-classification source (audit-2026-05-18 B1+C1+C2)
├── loaders/        # ~30 data-source loaders (yahoo, tushare, akshare, ccxt, pykrx, baostock, tencent, ...)
├── optimizers/     # risk_parity, mean_variance, turnover_aware, ...
├── metrics/        # performance stats
├── validation/     # config/data validation
├── regime/         # regime detection
├── models.py       # shared dataclasses
└── run_card.py     # run specification
```

## CONVENTIONS
- `initial_cash` validated >0 at the config boundary (rejects inf/NaN) — keep validation at the boundary, not deep in engines.
- Loader tests: patch at the import site (see tests/test_yahoo_loader.py), class-grouped, `tmp_path` fixtures.
- Data-source fallback chains live in src/market_data.py (shared with MCP/tools), not in loaders.

## ANTI-PATTERNS
- Do NOT duplicate `_detect_market` / symbol classification — it lives ONLY in engines/_market_hooks.py.
- Do not widen sandbox assumptions: runner subprocess drops to vibe-sandbox UID with RLIMIT_AS; engines must not expect write access outside ephemeral HOME.
