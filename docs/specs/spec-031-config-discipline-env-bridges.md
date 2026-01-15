---
spec: SPEC-031
title: Configuration Discipline (Settings-only; No `os.getenv` Sprawl)
version: 1.0.0
date: 2026-01-10
owners: ["ai-arch"]
status: Draft
related_requirements:
  - NFR-MAINT-001: Library-first; no bespoke config layers.
  - NFR-MAINT-002: Quality gates (ruff/pyright) must pass (ruff enforces pylint-equivalent rules).
  - NFR-MAINT-003: Single source of truth for config; no ADR-XXX markers.
related_adrs: ["ADR-050", "ADR-024"]
---

## Objective

Eliminate scattered `os.getenv` usage and ensure configuration is centralized in `src/config/settings.py`.

## Non-goals

- Introducing a new persistence layer for configuration
- Adding new environment variable names (must map existing ones)

## Technical design

### Settings additions (minimal)

In `src/config/settings.py`:

1. Add `TelemetryConfig` that maps existing env variables:

   - `DOCMIND_TELEMETRY_DISABLED`
   - `DOCMIND_TELEMETRY_SAMPLE`
   - `DOCMIND_TELEMETRY_ROTATE_BYTES`
   - plus a configurable path defaulting to `./logs/telemetry.jsonl`

   Compatibility note:
   - These env vars are *flat* (not nested `__` keys). Prefer either top-level
     `telemetry_*` fields (e.g., `telemetry_disabled`) or explicit aliases so
     we do not introduce a parallel `DOCMIND_TELEMETRY__*` namespace.

2. Add `ImageEncryptionConfig` mapping existing env variables:

   - `DOCMIND_IMG_AES_KEY_BASE64`
   - `DOCMIND_IMG_KID`
   - `DOCMIND_IMG_DELETE_PLAINTEXT`

3. Add optional `environment: str | None` at top-level settings (maps `DOCMIND_ENVIRONMENT`) for OTEL resource tags.

4. Fix the `ADR-XXX` marker and formalize hashing secret usage:

   - Replace `# Canonical hashing (ADR-XXX)` with a real ADR reference (`ADR-050` or `ADR-047`).
   - Fix the validator error message for `HashingConfig.hmac_secret` to reference the correct env var:
     - `DOCMIND_HASHING__HMAC_SECRET`
   - Use `settings.hashing.hmac_secret` for keyed fingerprints in `src/utils/log_safety.py` (see `SPEC-028`) and as the secret source for `src/utils/canonicalization.py` configuration (so tests and production share the same policy).

### Consumer refactors

Replace `os.getenv` reads with `settings.*` lookups in:

- `src/utils/telemetry.py`
- `src/telemetry/opentelemetry.py`
- `src/utils/security.py` and `src/processing/pdf_pages.py` encryption metadata plumbing

### Backward compatibility

- Existing env var names remain valid.
- Dotenv loading remains handled by Pydantic Settings (do not add `load_dotenv()` elsewhere). In DocMind, `.env` is loaded explicitly at startup via `bootstrap_settings()` (or by passing `_env_file=...`) to avoid import-time filesystem reads and to keep tests hermetic.
- Settings source precedence follows pydantic-settings defaults (init kwargs > env vars > `.env` (when loaded) > secrets > defaults); `.env` never overrides exported env vars.
- Optional local-dev override modes (explicitly gated; not for production):
  - `DOCMIND_CONFIG__DOTENV_PRIORITY=dotenv_first` can be used to make repo `.env` override exported env vars **for DocMind settings only**, with a safety guard that keeps `security.*` env-first.
  - `DOCMIND_CONFIG__ENV_MASK_KEYS` / `DOCMIND_CONFIG__ENV_OVERLAY` provide an allowlist mechanism to prevent accidental usage of global machine env vars by dependencies and to expose compatible env vars sourced from validated settings.

## Testing strategy

- Unit tests for new settings mapping (env → settings):
  - telemetry disabled, sampling, rotation bytes
  - image encryption key parsing behavior
  - environment mapping
- Unit tests for telemetry emitter to ensure it uses settings (no env reads).
- Unit tests for `HashingConfig`:
  - `DOCMIND_HASHING__HMAC_SECRET` validation (min length)
  - error message references the correct env var name (human-facing, but important for UX)

## Security

- Ensure secrets fields are `repr=False` and never included in logs/telemetry.

## RTM updates (docs/specs/traceability.md)

Add a planned row:

- NFR-MAINT-003: “Config discipline (settings-only; remove env sprawl)”
  - Code: settings + telemetry/security modules
  - Tests: new unit tests for mapping
  - Verification: test
  - Status: Planned → Implemented
