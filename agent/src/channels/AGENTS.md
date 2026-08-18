# src/channels/ — 17 IM ADAPTERS

Inbound IM messages become ordinary agent sessions: BaseChannel → MessageBus (bus/) → ChannelRuntime → SessionService; replies flow back outbound.

## STRUCTURE
- base.py — BaseChannel ABC (start/stop/send/login); every adapter implements it.
- registry.py — auto-discovery of built-ins + external plugins; availability flags + install-hint extras per platform.
- manager.py — ChannelManager: starts enabled adapters, outbound retry backoff.
- runtime.py — ChannelRuntime: async bridge into SessionService.
- bus/ — MessageBus + Inbound/OutboundMessage events.
- pairing/ — /pairing authorization, fail-closed.
- One module per platform: telegram, slack, discord, feishu, dingtalk, wecom, weixin, whatsapp, matrix, qq, msteams, napcat, mochat, email, signal, websocket.

## CONVENTIONS
- New platform: implement BaseChannel in a new module — registry discovers it automatically. Heavy deps go behind a pyproject extra (telegram, discord, feishu, ... or `channels` for all).
- Runtime singletons (bus/manager/runtime) are lazy, gated by ENABLE_SESSION_RUNTIME, owned by api/state.py.
- Adapters are started/stopped via api/channels_routes.py.

## ANTI-PATTERNS
- mochat.py is on the CI datetime-gate file list: only `datetime.now(timezone.utc)`.
- Pairing must stay fail-closed; do not auto-authorize unknown senders.
- Availability checks must degrade gracefully when an extra is missing (registry install hints, never hard ImportError at discovery).
