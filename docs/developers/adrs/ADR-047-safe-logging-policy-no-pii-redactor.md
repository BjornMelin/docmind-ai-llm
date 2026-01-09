---
ADR: 047
Title: Safe Logging Policy — Remove No-op PII Redactor Stub
Status: Proposed
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 024, 032
Tags: security, privacy, logging, telemetry
References:
  - https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
  - https://owasp.org/Top10/2021/A09_2021-Security_Logging_and_Monitoring_Failures/
---

## Description

Remove the `src.utils.security.redact_pii()` no-op stub and adopt a clear policy: **never log raw user content or document text**. Prefer logging metadata (counts, sizes, hashes) instead of attempting in-process redaction.

## Context

`src/utils/security.py` currently exposes:

- `redact_pii(text: str) -> str` which is a documented no-op
- unit tests that assert the no-op behavior

This creates a false sense of safety and increases the likelihood that future code will log sensitive text assuming it was “redacted”.

DocMind’s posture is local-first, offline-first, and security-first; logging should be safe by default.

## Alternatives

- A: Implement regex-based PII redaction in core (high false-negative risk; ongoing maintenance)
- B: Keep the no-op stub (unacceptable; false confidence)
- C: Remove the stub; enforce “no raw content in logs” via code patterns + tests (Selected)
- D: Add a dedicated scrubbing dependency (out-of-scope for v1; adds complexity and policy surface)

### Decision Framework (≥9.0)

| Option                                  | Leverage (35%) | Value (25%) | Risk Reduction (25%) | Maint (15%) |    Total | Decision    |
| --------------------------------------- | -------------: | ----------: | -------------------: | ----------: | -------: | ----------- |
| **C: Remove stub + no-content logging** |              9 |           8 |                   10 |           9 | **9.05** | ✅ Selected |
| A: Regex redaction                      |              6 |           7 |                    6 |           4 |     5.95 | Rejected    |
| D: Scrubbing dependency                 |              5 |           6 |                    8 |           3 |     5.75 | Rejected    |
| B: Keep stub                            |              1 |           2 |                    0 |           7 |     1.65 | Rejected    |

## Decision

1. Remove `redact_pii()` and its tests.

2. Add small “log safety” helpers (hash/length only) where needed and require all telemetry/log events to avoid raw content.

3. Update docs/specs to reflect the real policy: **avoid logging sensitive content** rather than pretending to redact.

## Consequences

### Positive Outcomes

- Eliminates false confidence in a no-op redactor.
- Simplifies compliance with NFR-SEC-002 (logging excludes sensitive content).
- Reduces risk of accidental prompt/document leakage in logs.

### Trade-offs

- We do not attempt automatic redaction; operators needing redaction must implement it in their own logging/export pipeline or use a dedicated scrubbing solution in a follow-up.
