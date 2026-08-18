# PROJECT KNOWLEDGE BASE

**Generated:** 2026-08-18T22:08:47+0800
**Commit:** 8a29f3d
**Branch:** main

## OVERVIEW
Vibe-Trading (`vibe-trading-ai` v0.1.13): natural-language finance research AI agent with backtesting. Python ≥3.11,<3.14 backend (FastAPI + LangChain/LangGraph ReAct loop + ~460-alpha factor zoo + multi-market backtest engines), React 19/Vite SPA, MCP server surface.

## STRUCTURE
```
Vibe-Trading/
├── agent/          # the ENTIRE Python product (not just agent logic) → agent/AGENTS.md
│   ├── api_server.py   # thin FastAPI assembler; infra in src/api/
│   ├── mcp_server.py   # MCP surface (~59 read-only tools)
│   ├── cli/            # two-layer CLI: main.py front door + _legacy.py (5.8k lines, source of truth)
│   ├── src/            # ~28 subpackages (factors, channels, live, trading, swarm, ...)
│   ├── backtest/       # engines/loaders/optimizers; imported as top-level `backtest.*`
│   └── tests/          # ~430 FLAT test_*.py, no package mirroring
├── frontend/       # React 19 + Vite 6 + strict-TS SPA; version locked to pyproject
├── tools/          # MISLEADING NAME: 3 CI policy-gate scripts only, not dev utilities
├── scripts/dev     # single extensionless bash orchestrator (a file, not a directory)
├── wiki/           # hand-written static site (vibetrading.wiki, Cloudflare Pages)
└── reports/        # user's live A-share research workspace; HANDOFF.md = ops runbook
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add/bench factors | agent/src/factors/ | registry is AST-based (never imports); contract frozen |
| Factor modules (~460) | agent/src/factors/zoo/ | strict purity contract → zoo AGENTS.md |
| Backtest engines/loaders | agent/backtest/ | engines: china_a, crypto, forex, global_equity, india, korea, options_portfolio |
| HTTP routes | agent/src/api/*_routes.py | api_server.py only assembles |
| Agent ReAct loop | agent/src/agent/loop.py | PROTECTED: discuss before changing |
| Live-order safety | agent/src/live/ | fail-closed gates; do not "fix" documented behaviors |
| Broker connectors | agent/src/trading/connectors/ | shoonya/dhan structurally paper-only |
| Chat integrations (16) | agent/src/channels/ | bus/ + registry |
| CLI subcommands | agent/cli/_legacy.py | argparse dispatch; cli/main.py = interactive path only |
| CI-forbidden patterns | tools/ci_grep_gates.sh | source of truth; never inline elsewhere |
| Env-var reads | agent/src/config/accessor.py | EnvConfig singleton; raw os.getenv is CI-gated |
| Ops runbook (A-share daily pipeline) | reports/HANDOFF.md | stable-7 model frozen; no auto-push |
| Wiki content | wiki/ | hand-written CF Pages site → wiki/AGENTS.md |

## CODE MAP
LSP unavailable (basedpyright not installed); centrality measured via grep/ast-grep import counts.

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| src.factors.base | module | agent/src/factors/base.py | 454 | operator library for all zoo factors |
| backtest.loaders | package | agent/backtest/loaders/ | 197 | ~30 data-source loaders |
| src.agent.tools | module | agent/src/agent/tools.py | 78 | tool wiring for the agent loop |
| src.trading.connectors | package | agent/src/trading/connectors/ | 77 | broker SDK adapters |
| backtest.engines | package | agent/backtest/engines/ | 73 | per-market backtest engines |
| src.config.accessor | module | agent/src/config/accessor.py | 45 | EnvConfig singleton (env gate) |
| src.channels.bus | package | agent/src/channels/bus/ | 45 | channel event bus |
| src.live.mandate | module | agent/src/live/mandate.py | 30 | trading-mandate enforcement |

## CONVENTIONS
- Flat-packaging trap: pyproject `package-dir = {"" = "agent"}` → imports are top-level `src.*`, `cli`, `backtest`; there is no `vibe_trading` namespace. `agent/` must be on PYTHONPATH (scripts/dev sets it).
- Python: ruff E/F/W (E501 ignored, 120-col target), no import-order rule, no mypy; black declared but unenforced.
- Env vars: read ONLY via src.config accessor (CI AST gate); escape hatch `# noqa: env-gate`; tests/ exempt.
- Tests: flat `agent/tests/test_<module>_<behavior>.py`; root conftest auto-isolates os.environ + resets EnvConfig — never re-implement; patch mocks at the import site.
- Files <400 lines, 800 hard cap (cli/_legacy.py predates the rule).
- Community commits: DCO `Signed-off-by:` trailer; NO AI-assistant attribution trailers.
- Frontend: tsc strict is the only gate (no linter exists); tests run only from `src/**/__tests__/*.test.{ts,tsx}`.

## ANTI-PATTERNS (THIS PROJECT)
- `yaml.load(` — must be safe_load (CI gate).
- Literal "WorldQuant" in shipped artifacts — say "Kakushadze 101 Formulaic Alphas" (CI gate).
- `datetime.utcnow(` / bare `datetime.now(` in 10 pinned files — use `datetime.now(timezone.utc)` (CI gate).
- Raw `os.getenv`/`os.environ[...]` outside agent/src/config/ (CI gate).
- Per-stock codes (`000001.SH`, `AAPL.US`) in wiki/ json/csv or alpha-library html (CI gate).
- Never run live-trading/broker-write/payment/wallet flows in PR validation.
- Never commit .env with real values, token caches, run artifacts, `*.pkl`.
- src/agent/, src/session/, src/providers/ are protected — prior discussion required.

## UNIQUE STYLES
- Zoo factor modules: `__alpha_meta__` dict literal + pure `compute(panel)`; AST-parsed, never imported.
- Deliberate "bugs" in live/ governance: swallowed chain-append failure, O(n) ledger verify, `repeatable=False` — read module docstrings before touching.
- Two unrelated skills trees: agent/src/skills/ (88 built-in, shipped as package data) vs agent/skills/ (distribution).
- Runtime state (sessions.db, runs/, live/, shadow_runs/) may materialize at repo root OR agent/ depending on CWD; both gitignored.

## COMMANDS
```bash
pip install -e ".[dev,openbb,stats]"
scripts/dev up                # backend :8899 + frontend :5899 (also stop/status/logs)
pytest --ignore=agent/tests/e2e_backtest --ignore=agent/tests/test_e2e_harness_v2.py -q
bash tools/ci_grep_gates.sh   # CI safety gates (run before any PR)
cd frontend && npm ci && npm run build && npx vitest run
```

## NOTES
- Ports: api_server argparse default is 8000 but every deployment path uses 8899.
- Python <3.14 is a real wall: llvmlite ships no cp314 wheels.
- reports/ is the user's live research workspace: stable-7 model frozen, never git-push without explicit request, pkl never committed (reports/HANDOFF.md).
- e2e tests are deliberately gitignored/local-only; CI `--ignore` flags are defensive (targets absent from tree).
- Docker: hardened compose (read-only rootfs, cap_drop ALL, sandbox user uid 10001 for generated-code runs).
