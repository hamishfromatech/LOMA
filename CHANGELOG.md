# Changelog

All notable changes to LOMA. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2026-11-25] — audit and polish pass

One combined pass on the UI/UX audit, the backend P2 hardening, and the
Next.js security upgrade.

### Security

- Bumped Next.js from 16.4.3 to 16.4.6 (exact pin, `eslint-config-next` aligned
  to 16.4.6). The previous version is affected by a published security advisory.
- Response headers in `next.config.ts`: `X-Frame-Options: DENY`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, and a
  Content-Security-Policy — `connect-src` allows the local API default and
  `localhost:5050`; `script-src`/`style-src` are unsafe-inline only for the dev/
  launch stack; `object-src`, `base-uri` and `frame-ancestors` locked down.
- Backend P2 hardening: `_env` coercion, atomic file writes, tolerant NDJSON
  parsing, Ollama negative-cacheing, tool-output prompt-injection coverage,
  Host/Origin guard on the local API.

### Added

- Toast notifications: transient, bottom-right, `aria-live=assertive`,
  auto-dismiss after 4.5s, max 3 stacked — used for voice-toggle revert, model
  refresh and model switch instead of silent failures or persistent banners.
- Scroll-effect indicator: sticky gradient fade on the messages container derived
  from real scroll position.
- `usePrefersReducedMotion` (matchMedia + `useSyncExternalStore`) driving
  `resize="instant"` on the chat container — the CSS reduced-motion block could
  not reach `use-stick-to-bottom`'s JS-driven smooth tween.
- Skeleton loading states (`ui/skeleton.tsx`) for the chat "loading conversation"
  state and the settings loading state.
- Skip-link target `#loma-main` (also during onboarding), keyboard focus trap and
  Escape handling in the search drawer.
- Cmd+Enter send and Cmd+Shift+K search on the composer.
- Composer footer hints now reflect busy state (`↵ send` / `⇧↵ new line`).
- Backend endpoint for tool-result rendering with the event stream preserved.

### Changed

- Markdown external links now render with `target="_blank" rel="noopener noreferrer"`.
- Conversation copy tristates (idle/copying/copied/failed); failed copy surfaces
  as an error instead of a silent no-op.
- Composer busy behavior: Enter starts a new line while a turn is running and
  sends when idle (footer hint matches).
- `lib/api.ts` raises on non-2xx responses instead of treating errors as done.
- Settings page: skeleton rows while loading, local temperature state with a
  debounced PATCH, real heading structure, dirty-check before exit.
- Accessibility pass: heading structure, contrast bumps (`--loma-faint` #9b9188),
  focus return, `aria-live=polite` transcript region, `role="status"` loading
  regions, dialog semantics on the search drawer and mobile drawer.
- Memoized transcript message components (`memo`/`useCallback`/`useMemo`); tool
  status transitions re-render correctly.
- Backend P2: typed load/save, validation on MCP config, tolerant date coercion,
  AI-generated errors surfaced as errors, not silent no-ops.
- README aligned with reality (331 tests, missing modules, largest file ~1070 lines).

### Removed

- Dead code from the UI audit: ~11 Loader variants down to `DotsLoader`, the
  unused Avatar/Message/Reasoning/System-message/Prompt-suggestion/Tool
  components, and `Wordmark`.
- Dead `GET /api/history` endpoint, `MAX_BODY_BYTES` constant (now derived),
  `/api/mcp` re-added cleanly after the hardening pass.

### Fixed

- 331 pytest tests passing, ruff clean, tsc clean, eslint clean,
  `next build` succeeds (`/`, `/settings`, `/_not-found` prerender).
