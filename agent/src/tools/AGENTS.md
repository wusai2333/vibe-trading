# src/tools/ — ~75 AGENT TOOL MODULES

Zero-config auto-discovery: tools/__init__.py imports every module (pkgutil) then collects `BaseTool.__subclasses__()`; `build_registry()` applies policy gates.

## ADDING A TOOL
1. New `*_tool.py` module subclassing BaseTool (contract lives in src/agent/tools.py — 78 importers).
2. Implement `check_available()` for capability gating; it runs at registry build.
3. No registration call needed — discovery is automatic.

## WHERE TO LOOK
- __init__.py — discovery + build_registry policy: shell tools (bash/background) OFF by default on networked transports; PersistentMemory/session_id injection; remote MCP tools appended as MCPRemoteTool.
- mcp.py — MCP client adapter wrapping remote MCP tools as BaseTool. Session-level mcpServers injection is silently stripped unless ALLOW_SESSION_MCP_SERVERS=1; MCP tools never enter the parallel readonly path; excluded from Swarm registries v1; no hot reload.
- redaction.py — THE scrubbing choke point (paths CWE-209, credential/PII keys, text patterns); shared by agent loop, swarm worker/store/runtime, live audit.
- quantlib_tool.py — allowlisted gateway to src/quantlib: module allowlist + `__all__`-only + no writer-prefixed names; deliberately avoids widening the shell gate.
- path_utils.py — shared path helpers (20 importers).
- lockup_expiry_tool.py — on the CI datetime-gate list.

## ANTI-PATTERNS
- Never bypass redaction.py for user-visible output.
- Do not widen shell-tool availability on networked transports to "fix" a missing capability — add a scoped tool instead.
- Caller-provided MCP command/URL/env/allowlist injection is forbidden unless the code path explicitly documents and tests the opt-in.
