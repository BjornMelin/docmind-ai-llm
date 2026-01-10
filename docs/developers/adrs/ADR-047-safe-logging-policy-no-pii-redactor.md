---
ADR: 047
Title: Safe Logging Policy — Metadata-only + Keyed Fingerprints + Redaction Backstop
Status: Proposed
Version: 1.1
Date: 2026-01-10
Supersedes:
Superseded-by:
Related: 024, 032
Tags: security, privacy, logging, telemetry
References:
  - https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
  - https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html
  - https://owasp.org/Top10/2021/A09_2021-Security_Logging_and_Monitoring_Failures/
---

## Description

Adopt a safe-by-default logging posture: **never log raw user/document/model text**, log **metadata only**, and add (1) **keyed HMAC fingerprints** for correlation and (2) a **deterministic regex redaction backstop** for rare cases where strings could reach logs (for example exception messages).

## Context

`src/utils/security.py` currently exposes:

- `redact_pii(text: str) -> str` which is a documented no-op
- unit tests that assert the no-op behavior

This creates a false sense of safety and increases the likelihood that future code will log sensitive text assuming it was “redacted”.

DocMind’s posture is local-first, offline-first, and security-first; logging should be safe by default.

## Alternatives

- A: Policy only — remove the no-op stub and enforce “no raw content in logs” via patterns + tests, but do not provide a backstop
- B: Keep the no-op stub (unacceptable; false confidence)
- C: Add a heavyweight in-app PII detection/redaction stack (high complexity; false positives/negatives; large offline footprint)
- D: Policy + keyed fingerprints + deterministic backstop (Selected)

### Decision Framework (≥9.0, Tier-2)

Weights: Complexity 40% · Perf 30% · Alignment/Security 30% (10 = best)

| Option                                                            | Complexity (40%) | Perf (30%) | Alignment/Security (30%) |   Total | Decision    |
| ----------------------------------------------------------------- | ---------------: | ---------: | -----------------------: | ------: | ----------- |
| **D: Metadata-only + HMAC fingerprints + regex backstop + tests** |              9.0 |        9.0 |                     10.0 | **9.3** | ✅ Selected |
| A: Policy only + tests (no backstop)                              |              9.5 |        9.5 |                      7.0 |     8.8 | Rejected    |
| C: Heavy PII detection stack                                      |              4.5 |        6.0 |                      6.0 |     5.3 | Rejected    |
| B: Keep no-op stub                                                |              8.0 |       10.0 |                      1.0 |     6.7 | Rejected    |

## Decision

1. Remove the no-op `redact_pii()` stub and its tests.

2. Add a small, typed `src/utils/log_safety.py` module to support safe correlation without raw content:

   - **Keyed fingerprinting** (`HMAC-SHA256`) using a local secret from settings (shared source-of-truth with canonicalization; do not log the secret; support rotation).
   - **Deterministic redaction backstop** for rare strings that may reach logs (for example exception messages). This is a _last-ditch_ defense, not the primary safety control.
   - **URL sanitization** that logs only `scheme://host[:port]`.

3. Enforce “no raw content in logs” via tests:

   - add a canary-string test that asserts no canary content appears in captured logs/JSONL telemetry
   - avoid enabling verbose third-party logging that could dump content (LangGraph/LlamaIndex)

## Security & Privacy

- Treat all user prompts, document text, and model outputs as sensitive content.
- Never log secrets or tokens (API keys, bearer tokens, cookies).
- Prefer strict structured logging with allowlisted fields. The regex backstop is not a replacement for good log discipline.

## Consequences

### Positive Outcomes

- Eliminates false confidence in a no-op redactor while adding a real last-ditch defense.
- Preserves operational debugging value via correlation-friendly fingerprints (without logging content).
- Reduces risk of accidental prompt/document leakage in logs and telemetry.

### Trade-offs

- Regex backstops are imperfect; they reduce harm but do not guarantee full redaction of arbitrary sensitive text.
