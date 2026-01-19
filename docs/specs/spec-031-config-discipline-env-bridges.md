---
spec: SPEC-031
title: Configuration Discipline (Settings-only; No `os.getenv` Sprawl)
version: 1.0.0
date: 2026-01-10
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - NFR-MAINT-001: Library-first; no bespoke config layers.
  - NFR-MAINT-002: Quality gates (ruff/pyright) must pass (ruff enforces pylint-equivalent rules).
  - NFR-MAINT-003: Single source of truth for config; no ADR placeholder markers.
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

   Implementation note:
   - These env vars are *flat* (not nested `__` keys) and are the canonical
     operator contract. Because the settings schema is nested,
     `DOCMIND_TELEMETRY__*` env vars may also exist as a derived/advanced surface
     (e.g., `DOCMIND_TELEMETRY__JSONL_PATH`). When both flat and nested are
     provided, the flat `DOCMIND_TELEMETRY_*` values take precedence to preserve
     compatibility.

2. Add `ImageEncryptionConfig` mapping existing env variables:

   - `DOCMIND_IMG_AES_KEY_BASE64`
   - `DOCMIND_IMG_KID`
   - `DOCMIND_IMG_DELETE_PLAINTEXT`

   Compatibility note:
   - The canonical operator contract is the flat `DOCMIND_IMG_*` env vars. Since
     the schema is nested (`settings.image_encryption.*`), derived nested env
     vars may also exist (e.g., `DOCMIND_IMAGE_ENCRYPTION__KID`). When both are
     provided, `DOCMIND_IMG_*` takes precedence.

3. Add optional `environment: str | None` at top-level settings (maps `DOCMIND_ENVIRONMENT`) for OTEL resource tags.

4. Fix the ADR placeholder marker and formalize hashing secret usage:

   - Replace `# Canonical hashing (ADR-<placeholder>)` with a real ADR reference (`ADR-050` or `ADR-047`).
   - Fix the validator error message for `HashingConfig.hmac_secret` to reference the correct env var:
     - `DOCMIND_HASHING__HMAC_SECRET`
   - Use `settings.hashing.hmac_secret` for keyed fingerprints in `src/utils/log_safety.py` (see `SPEC-028`) and as the secret source for `src/utils/canonicalization.py` configuration (so tests and production share the same policy).

### Consumer refactors

Replace direct env reads with `settings.*` lookups in:

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

Record implementation in RTM:

- `NFR-MAINT-003.2`: “Config discipline (settings-only; remove env sprawl)”
  - Code: `src/config/settings.py`, `src/utils/telemetry.py`, `src/utils/security.py`
  - Tests: `tests/unit/config/test_telemetry_image_mappings.py`
  - Verification: test
  - Status: Implemented
