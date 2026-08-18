# src/agent/ — REACT CORE (PROTECTED AREA)

The agent engine. PR template requires prior discussion before changes here (also src/session/, src/providers/).

## STRUCTURE
- loop.py — AgentLoop (1,934 lines): ReAct loop, 5-layer context compaction, parallel batching of consecutive readonly tools, grounding ledger, trace writer. 93 importers of the package.
- tools.py — BaseTool + ToolRegistry: THE tool contract (~75 tool modules subclass it; highest fan-in in the repo).
- context.py — ContextBuilder + system prompt assembly (skill one-liners + tool count). On the CI datetime-gate list.
- skills.py — SkillsLoader: progressive disclosure of SKILL.md packages (one-liners in prompt; load_skill pages full docs section-by-section — exists because tushare alone is 102,890 chars).
- grounding.py, memory.py, trace.py, progress.py, frontmatter.py — supporting subsystems.

## CONVENTIONS
- One AgentLoop per session, owned by SessionService (src/session/service.py, 4-thread pool; concurrent send → HTTP 409).
- Context compaction is layered — extend a layer, don't bolt preprocessing onto the loop body.
- Readonly tools may be batched in parallel; tools with side effects must stay serial.

## ANTI-PATTERNS
- Do NOT reuse AgentLoop for swarm workers — worker.py intentionally runs its own minimal ChatLLM loop with a per-worker registry from the preset tool whitelist.
- MCP tools are excluded from the parallel readonly batching path.
- Generated backtest code is UNTRUSTED input — it runs only through the src/core/runner.py sandbox, never in-process.
