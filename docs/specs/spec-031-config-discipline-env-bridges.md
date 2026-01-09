---
spec: SPEC-031
title: Configuration Discipline (Settings-only; No `os.getenv` Sprawl)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - NFR-MAINT-001: Library-first; no bespoke config layers.
  - NFR-MAINT-002: Quality gates (ruff/pyright/pylint) must pass.
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

2. Add `ImageEncryptionConfig` mapping existing env variables:

   - `DOCMIND_IMG_AES_KEY_BASE64`
   - `DOCMIND_IMG_KID`
   - `DOCMIND_IMG_DELETE_PLAINTEXT`

3. Add optional `environment: str | None` at top-level settings (maps `DOCMIND_ENVIRONMENT`) for OTEL resource tags.

4. Remove the unused `HashingConfig` block and the `ADR-XXX` marker (unless hashing config is truly used).

### Consumer refactors

Replace `os.getenv` reads with `settings.*` lookups in:

- `src/utils/telemetry.py`
- `src/telemetry/opentelemetry.py`
- `src/utils/security.py` and `src/processing/pdf_pages.py` encryption metadata plumbing

### Backward compatibility

- Existing env var names remain valid.
- `.env` loading remains handled by `DocMindSettings` (do not add `load_dotenv()` elsewhere).

## Testing strategy

- Unit tests for new settings mapping (env → settings):
  - telemetry disabled, sampling, rotation bytes
  - image encryption key parsing behavior
  - environment mapping
- Unit tests for telemetry emitter to ensure it uses settings (no env reads).

## Security

- Ensure secrets fields are `repr=False` and never included in logs/telemetry.

## RTM updates (docs/specs/traceability.md)

Add a planned row:

- NFR-MAINT-003: “Config discipline (settings-only; remove env sprawl)”
  - Code: settings + telemetry/security modules
  - Tests: new unit tests for mapping
  - Verification: test
  - Status: Planned → Implemented
