# agent/ — THE PYTHON PRODUCT

Everything Python lives here: API server, MCP server, CLI, backtest engine, all source. Not just "agent logic".

## STRUCTURE
```
agent/
├── api_server.py    # thin FastAPI assembler; routes in src/api/*_routes.py
├── mcp_server.py    # FastMCP, ~59 read-only tools; stdio/SSE/HTTP
├── cli/             # main.py (interactive front door) + _legacy.py (5.8k-line subcommand dispatcher)
├── src/             # ~28 subpackages — see WHERE TO LOOK
├── backtest/        # engines/loaders/optimizers → backtest/AGENTS.md
├── skills/          # DISTRIBUTION skills (ashare-mootdx) — NOT src/skills/
├── scripts/         # benchmark + blog-patching utilities (w4a_*)
└── tests/           # ~430 FLAT test_*.py files
```

## WHERE TO LOOK
| Subsystem | Path | Core type |
|-----------|------|-----------|
| ReAct engine | src/agent/ | AgentLoop (loop.py) |
| Tool registry + ~75 tools | src/agent/tools.py + src/tools/ | BaseTool |
| Factor zoo infra | src/factors/ | Registry |
| Live-trading governance | src/live/ | mandate/order_guard/audit |
| Broker connectors | src/trading/connectors/ | per-broker sdk.py |
| IM channels (17) | src/channels/ | BaseChannel |
| Swarm multi-agent | src/swarm/ | SwarmRuntime + YAML presets |
| Sessions | src/session/ | SessionService (1 AgentLoop/session, 4-thread pool, 409 on concurrent send) |
| LLM providers | src/providers/ | build_llm / ChatLLM |
| Config | src/config/ | EnvConfig singleton + paths (~/.vibe-trading) |
| Sandboxed code exec | src/core/runner.py | UID-drop subprocess, RLIMIT_AS |
| Skills (88 built-in) | src/skills/ | SKILL.md packages, progressive disclosure |
| Quant math | src/quantlib/ | ~250 fns, __init__ imports nothing |
| Market data routing | src/market_data.py | symbol→source fallback chains |
| Paper portfolio | src/shadow_account/ | extraction + HTML report codegen |

## CONVENTIONS
- Imports are top-level `src.*`, `cli`, `backtest` (package-dir remap); run with `PYTHONPATH=agent`.
- CLI: non-interactive subcommands dispatch through `_legacy.main()`; `cli/__init__.py` re-exports the legacy namespace with explicit `_LEGACY_SYNCED_GLOBALS` for test monkeypatching — keep that list explicit.
- New tool: drop a BaseTool subclass module in src/tools/ — auto-discovered via pkgutil + `__subclasses__()`.
- New skill: SKILL.md dir in src/skills/ (frontmatter name/description/category); system prompt gets one-liners, `load_skill` pages full docs.
- EnvConfig is a cached singleton: call `reset_env_config()` after mutating os.environ (tests do this automatically).
- Cross-cutting scrubbing goes through src/tools/redaction.py (agent loop, swarm, live audit all share it).

## ANTI-PATTERNS
- No `vibe_trading` package namespace exists — never write `from vibe_trading import ...`.
- Do not confuse src/skills/ (built-in, shipped) with agent/skills/ (distribution).
- Swarm worker deliberately runs its own minimal ChatLLM ReAct loop — do NOT refactor it onto AgentLoop.
- api_server argparse default port is 8000 but all real paths use 8899.

## TESTS
- Flat `tests/test_<module>_<behavior>.py`; root conftest gives env isolation for free.
- Factors: purity + lookahead + golden-CSV gates under tests/factors/.
- Integration tests use real subprocess servers (tests/fixtures/fake_mcp_*.py + mcp_http_test_helpers.py), not mocks; mark `pytestmark = pytest.mark.integration`.
