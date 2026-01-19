---
spec: SPEC-022
title: Settings UI Hardening — Pre-validation, Safe Provider Badge, and .env Persistence
version: 1.0.1
date: 2026-01-09
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-021: Settings UI pre-validation + safe provider badge.
  - NFR-SEC-001: Default egress disabled; only local endpoints allowed unless explicitly configured.
  - NFR-SEC-002: Local data remains on device; logging excludes sensitive content.
  - NFR-SEC-004: Streamlit UI must not execute unsafe HTML/JS.
related_adrs: ["ADR-041", "ADR-024", "ADR-013", "ADR-016"]
---

## Objective

Harden the Streamlit Settings page so that:

- invalid configuration cannot be saved to `.env`
- runtime Apply only succeeds when LlamaIndex runtime is actually bound
- provider status UI uses Streamlit-native components (no `unsafe_allow_html=True`)

## Non-goals

- Creating a new persistence backend for settings (SQLite/JSON)
- Adding new providers beyond existing supported backends
- Exposing “dangerous” low-level tuning knobs in the UI

## User stories

1. As a user, when I input an invalid base URL, I see a clear error and cannot save/apply it.

2. As a user, when I change provider/model and press Apply, the app confirms the runtime is active (or shows an actionable failure message).

3. As a user, I can see a compact provider + model badge in Chat and Settings without any unsafe HTML rendering.

## Technical design

### Provider badge (safe)

Replace `src/ui/components/provider_badge.py` HTML markup with Streamlit-native UI:

- `st.badge(label, icon=..., help=...)` for provider and GraphRAG status
- `st.caption` for model id and resolved base URL (when applicable)

No `unsafe_allow_html=True` usage is permitted.

### Proposed settings validation

Implement a "proposed settings" pattern in `src/pages/04_settings.py`:

1. Collect widget values into an env-mapping using canonical keys (e.g., `DOCMIND_LLM_BACKEND`, `DOCMIND_OPENAI__BASE_URL`, `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS`).
2. Validate the candidate mapping:
   - Build a candidate dict with nested structure:

     ```python
     candidate = {
         "llm_backend": "lmstudio",
         "model": "Hermes-2-Pro-Llama-3-8B",
         "lmstudio_base_url": "http://localhost:1234/v1",
         "openai": {"base_url": "http://localhost:1234/v1"},
         "security": {"allow_remote_endpoints": False},
     }
     ```

     - Call `DocMindSettings.model_validate(candidate)` (or a helper) to ensure:
     - types/ranges are correct
     - OpenAI-compatible base URLs are normalized to a single `/v1` segment (see `src/config/settings_utils.py::ensure_v1`)
     - security policy validation passes (allowlist + localhost-only posture when `allow_remote_endpoints=false`)
3. Only after validation succeeds:
   - update the global `settings` singleton (in-memory)
     - apply updates **in-place** (do not rebind `src.config.settings.settings`) so modules importing `from src.config import settings` keep a consistent instance
   - call `initialize_integrations(*, force_llm: bool = False, force_embed: bool = False)` for Apply (see `src/config/integrations.py`)
   - persist to `.env` for Save.

### .env persistence

Replace custom `.env` writer in `src/pages/04_settings.py` with `python-dotenv==1.2.1` (pinned in `pyproject.toml`):

- Use `dotenv.set_key(".env", key, value, quote_mode="auto")` for values
- Use `dotenv.unset_key(".env", key)` for empty values (removes the key entirely)
- If `.env` does not exist, create it before calling `set_key`
- Keep comments and unrelated keys intact (best-effort; rely on python-dotenv semantics)

> **Edge cases**: Empty string values are treated as unset (removed via `unset_key`). Verify round-trip correctness in unit tests: write → read → validate returns expected values and preserves comments best-effort.

## Observability

- On Apply success/failure, emit a local JSONL event via `log_jsonl`:
  - `settings.apply`: { success, backend, model, reason? }
- Do not log secrets or full URLs beyond host:port (truncate if needed).

## Security

- Enforce offline-first and allowlist rules exactly as `src/config/settings.py` defines them:
  - Use `_norm_lmstudio()` for /v1 normalization and `_validate_endpoints_security()` plus `_validate_lmstudio_url()` for allowlist + localhost checks
  - Env var naming follows `DOCMIND_` prefix with `__` nesting (e.g., `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS`)
  - Cross-reference ADR-031 (Config Discipline) for full policy
- Treat the Settings UI as an untrusted input boundary:
  - validate before persist
  - validate before apply
- Remove unsafe HTML rendering from provider badge UI.

## Testing strategy

### Integration (Streamlit AppTest)

Add/extend `tests/integration/test_settings_page.py` (or create a new file) to assert:

- LM Studio base URL is normalized to include `/v1`; Save persists the normalized value to `.env`.
- Remote URL when `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=false` renders an error and disables actions; also test `=true` case to confirm remote URLs are allowed.
- Valid provider change persists to `.env` via `python-dotenv` and can be read back.
- Use `tests/helpers/apptest_utils.py` (`apptest_timeout_sec()`) for
  `AppTest.from_file(..., default_timeout=...)` and override with
  `TEST_TIMEOUT=<seconds>` when reproducing CI slowness locally.
- Keep UI tests offline and import-light: stub the provider badge health check
  (`adapter_registry.get_default_adapter_health`) unless the adapter itself is
  under test.

### Unit

- Provider badge source does not use `unsafe_allow_html=True` (inspect component code).
- `.env` persistence helper:
  - `set_key` called for normal values
  - `unset_key` called for empty values
  - Round-trip test: write → read → validate returns expected values.

## Rollout / migration

- Backward compatible: existing `.env` keys remain valid.
- UI behavior changes only by preventing invalid persistence and improving feedback.

## Performance expectations

- No network calls during validation.
- No heavy imports at module import time.
- Apply runtime should remain bounded by existing integration init cost.

## RTM updates (docs/specs/traceability.md)

Add a new row (planned → implemented):

- FR-021: “Settings pre-validation + safe badge”
  - Code: `src/pages/04_settings.py`, `src/ui/components/provider_badge.py`
  - Tests: `tests/integration/test_settings_page.py` (+ any new AppTest file)
  - Verification: test
