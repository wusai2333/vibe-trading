# frontend/ — REACT SPA

React 19.2 + Vite 6 + TypeScript 5.7 (strict) + Tailwind 3.4 + zustand 5 + ECharts 6 + i18next (5 locales). Talks to the backend via REST + SSE only — NO WebSockets anywhere. Version locked to pyproject (0.1.13).

## STRUCTURE
- src/lib/api.ts — THE API client (1,147 lines): `request<T>` fetch wrapper + `api` object + SSE URL builders. ~21 importers, the hub.
- src/lib/apiAuth.ts — Bearer key in localStorage; EventSource can't send headers, so each connect mints a single-use SSE ticket (POST /auth/sse-ticket → ?ticket=).
- src/hooks/useSSE.ts — reconnect/backoff (1s→30s ×2), LRU dedup (500), Last-Event-ID resume, HARDCODED event-type whitelist.
- src/stores/agent.ts — the ONLY zustand store (messages, streaming, toolCalls, swarmRuns, sseStatus).
- src/pages/ — 10 lazy routes; Agent.tsx (~1,800 lines, chat is the product).
- src/components/chat/ — 19 components, the heavyweight dir.
- src/components/charts/ — 9 ECharts wrappers sharing lib/echarts + lib/chart-theme + lib/theme-store.
- Path alias: `@/*` → src/* (tsconfig + vite + vitest all define it).

## CONVENTIONS
- New backend endpoint: add a method to the `api` object in lib/api.ts AND the path to PROXY_PATHS in vite.config.ts — src/__tests__/viteProxy.test.ts enforces this.
- New SSE event type: extend the knownTypes whitelist in useSSE.ts, or use raw EventSource (AlphaZoo/swarm bench do this deliberately — their progress/result events would be dropped by the whitelist; documented in AlphaZoo.tsx header).
- Shared UI state goes in the single store (stores/agent.ts). Theme is a hand-rolled useSyncExternalStore store (lib/theme-store.ts).
- Tests: vitest globals (no imports of describe/it/expect), jsdom, restoreMocks; factories in src/tests/helpers/factories.ts; files MUST live at `src/**/__tests__/**/*.test.{ts,tsx}` or they silently never run. Coverage counts only src/lib/** and src/stores/**.
- Polling where no stream exists: Runtime/Scheduled pages poll every 15s.

## ANTI-PATTERNS
- No linter/formatter exists — `tsc -b` strict (noUnusedLocals/noUnusedParameters) is the only gate; type errors fail CI.
- Do not add a second state store or Redux/React Query; the single-store pattern is deliberate.
- Dev server is pinned to port 5899 with proxy to :8899 — do not assume Vite default 5173.

## COMMANDS
```bash
npm ci && npm run build      # tsc -b && vite build (CI gate)
npm run test:run             # vitest
npm run dev                  # port 5899
```
