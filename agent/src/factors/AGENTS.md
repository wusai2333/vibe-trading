# src/factors/ — FACTOR INFRASTRUCTURE

Registry + operator library + bench/compare runners around `zoo/` (~460 alphas). The zoo contract is in zoo/AGENTS.md.

## WHERE TO LOOK
| File | Role |
|------|------|
| registry.py | Discovery/validation: AST-parses `__alpha_meta__` (never imports), pydantic AlphaMeta (extra=forbid), lazy compute with output validation. Singleton `get_default_registry()` caches one scan. **Contract frozen — referenced by zoo-porting agents.** |
| base.py | Operator library all factors compose: rank/zscore/scale, ts_* (rank/corr/cov/mean/std/max/min/argmax/argmin), delta, decay_linear, signed_power, safe_div, vwap. 454 importers — most central module in the repo. |
| _backend.py | Lazy bottleneck acceleration (move_argmax/argmin) + numpy sliding_window_view; env kill-switch. |
| factor_analysis_core.py | Shared Spearman IC/IR math. |
| bench_runner.py / bench_runner_strict.py | Zoo-wide IC bench (alive/reversed/dead), ProcessPoolExecutor, deflated-Sharpe multiple-testing. Consumed by CLI, api/alpha_routes.py, tools/alpha_bench_tool.py. |
| compare_runner.py | Head-to-head alpha compare (CLI/API/tool share it). |
| cli_handlers.py | `vibe-trading alpha {list,show,bench,compare,export-manifest}`. |

## CONVENTIONS
- Meta is read STATICALLY: `load_alpha_meta_from_py()` AST-parses + `ast.literal_eval`s the dict; modules are imported only at compute time. Never add an import-side path for metadata.
- `_meta.yaml` / manifest outputs are EXPORT-only, never the load path.
- compute() output contract enforced by registry: DataFrame, shape == panel["close"], no ±inf, NaN ≤ 95%.
- Zoo subdirs auto-discovered on scan; id = `<zoo>_<short>`.

## ANTI-PATTERNS
- Do not mass-edit generated zoos (gtja191/alpha101/qlib158) — the 16-op import boilerplate is intentional template output, lint debt not logic debt.
- base.py ops have subtle NaN/causality semantics (ts_rank, decay_linear, ts_argmax/argmin) — changes require golden tests in tests/factors/fixtures/goldens/.
- AlphaMeta schema changes break ~460 modules; treat as frozen.
