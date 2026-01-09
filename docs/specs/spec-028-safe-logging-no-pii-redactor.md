---
spec: SPEC-028
title: Safe Logging Policy — No PII Redactor Stub, Log Metadata Only
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - NFR-SEC-002: Logging excludes sensitive content.
  - NFR-MAINT-003: No misleading stubs in security helpers.
related_adrs: ["ADR-047", "ADR-024", "ADR-032"]
---

## Objective

Ensure DocMind logs and telemetry are safe-by-default by:

- removing the no-op `redact_pii()` function
- standardizing on “metadata-only” logging for user prompts, documents, and model outputs

## Non-goals

- Implementing comprehensive PII redaction in-app (high risk of gaps; high maintenance)
- Adding external scrubbing/export dependencies for v1

## Technical design

### Remove stub

- Delete `src/utils/security.py::redact_pii`
- Update `__all__` exports accordingly
- Delete/update the unit tests that assert no-op redaction

### Metadata-only helpers

Add a minimal helper module:

- `src/utils/log_safety.py` (new)

Suggested helpers:

- `fingerprint_text(text: str) -> dict[str, str | int]` returning:
  - `len` (int)
  - `sha256` (hex)
- `safe_url_for_log(url: str) -> str` returning only `scheme://host:port` (no path/query)

### Policy enforcement

Update any logging/telemetry events within scope to:

- avoid raw prompt, document text, or model output in logs
- log metadata only (counts, sizes, hashes)

## Security

- Never log secrets, API keys, or raw user content.
- Treat exceptions carefully: do not log exception messages that embed raw user inputs (wrap with safe metadata when necessary).

## Testing strategy

- Unit tests for `src/utils/log_safety.py`.
- Update existing security tests to remove `redact_pii` expectations.

## Rollout / migration

- No migration required. Removal is internal-only and reduces surface area.

## RTM updates (docs/specs/traceability.md)

Add a planned row:

- NFR-SEC-002: “Safe logging (no raw content; no no-op redactor)”
  - Code: `src/utils/security.py`, `src/utils/log_safety.py`
  - Tests: updated security tests + new unit tests
  - Verification: test
  - Status: Planned → Implemented
