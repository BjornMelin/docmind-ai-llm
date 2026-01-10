---
spec: SPEC-028
title: Safe Logging Policy — No PII Redactor Stub; Metadata-only + Keyed Fingerprints + Backstop
version: 1.0.0
date: 2026-01-10
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
- standardizing on **metadata-only** logging for user prompts, documents, and model outputs
- adding **keyed HMAC fingerprints** for correlation without content
- adding a **deterministic regex backstop** for rare string fields that may reach logs (for example exception messages)

## Non-goals

- Implementing comprehensive PII redaction in-app (high risk of gaps; high maintenance)
- Adding external scrubbing/export dependencies
- Logging raw prompts/documents/model outputs “temporarily for debugging”

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
  - `hmac_sha256_12` (hex prefix, keyed) for correlation
- `safe_url_for_log(url: str) -> str` returning only `scheme://host[:port]` (no path/query)
- `redact_text_backstop(text: str) -> str` (deterministic regex redaction) intended only for:
  - exception messages
  - rare string fields that could accidentally contain content

#### Key management

- Key source: `DocMindSettings.hashing.hmac_secret` (local secret; must never be logged). This secret is also the source-of-truth for HMAC canonicalization in `src/utils/canonicalization.py`.
- Implementation detail: encode `DocMindSettings.hashing.hmac_secret` to UTF-8 bytes when constructing `CanonicalizationConfig` to match the `hmac.new()` bytes requirement and the 32-byte minimum. Ensure `src/utils/log_safety.py::fingerprint_text()` applies the same UTF-8 encoding path for consistency.
- Env var: `DOCMIND_HASHING__HMAC_SECRET` (use a strong random string; at least 32 bytes).
- Rotation: bump `DocMindSettings.hashing.hmac_secret_version` and treat fingerprints as non-stable across rotations.

### Policy enforcement

Update any logging/telemetry events within scope to:

- avoid raw prompt, document text, or model output in logs
- log metadata only (counts, sizes, fingerprints)
- avoid exception messages that embed raw user/document text; use `redact_text_backstop()` for the message if you must log it

## Security

- Never log secrets, API keys, or raw user content.
- Treat exceptions carefully: do not log exception messages that embed raw user inputs (wrap with safe metadata and apply `redact_text_backstop()`).

## Testing strategy

- Unit tests for `src/utils/log_safety.py` (fingerprints, URL sanitization, and backstop redaction).
- Update existing security tests to remove `redact_pii` no-op expectations.
- Add a “canary” test:
  - add in `tests/unit/utils/test_log_safety.py` (or `tests/integration/test_logging_safety.py` if integration scope is preferred)
  - configure standard logging with an in-memory handler (StringIO-backed `StreamHandler`) and/or a JSONL telemetry sink mock
  - emit logs containing a known canary string (e.g., `CANARY_SECRET_12345`) in message, exception, and structured context
  - run the safe-logging pipeline (`redact_text_backstop()` / `fingerprint_text()` helpers or logger wrapper)
  - assert the canary does not appear in captured logs/JSONL output (cover message, exception message, and structured fields)

## Rollout / migration

- No migration required. Removal is internal-only and reduces surface area.

## RTM updates (docs/specs/traceability.md)

Add a planned row:

- NFR-SEC-002: “Safe logging (no raw content; no no-op redactor)”
  - Code: `src/utils/security.py`, `src/utils/log_safety.py`
  - Tests: updated security tests + new unit tests
  - Verification: test
  - Status: Planned → Implemented
