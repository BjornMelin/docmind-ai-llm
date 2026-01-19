---
ADR: 050
Title: Configuration Discipline — Eliminate `os.getenv` Sprawl and Formalize Local Secrets
Status: Implemented
Version: 1.1
Date: 2026-01-10
Supersedes:
Superseded-by:
Related: 024, 032, 047
Tags: configuration, settings, maintainability, security
References:
  - https://docs.pydantic.dev/latest/concepts/pydantic_settings/
---

## Description

Centralize runtime configuration in `src/config/settings.py` by:

- removing direct `os.getenv` usage from core modules where settings already exist or should exist
- adding missing settings groups for JSONL telemetry and image encryption toggles
- fixing the ADR placeholder marker and formalizing the existing hashing secret so it can be used for keyed fingerprints (ADR-047)

## Context

DocMind’s configuration contract is:

- `DocMindSettings` is source of truth (Pydantic Settings v2)
- env prefix `DOCMIND_`, nested fields via `__`
- core code should not scatter `os.getenv` calls

Current drift points:

- hashing config still contains an ADR placeholder marker and needs a real ADR reference
- `HashingConfig`’s validator error message references the wrong env var name (it should point to `DOCMIND_HASHING__HMAC_SECRET`)
- `src/utils/canonicalization.py` implements HMAC-based canonical hashes and is test-covered, but is not yet wired to `DocMindSettings.hashing` (and the secret is required for keyed fingerprints in safe logging; ADR-047)

## Alternatives

- A: Keep `os.getenv` reads in place (status quo)
- B: Introduce a thin “env bridge” module with centralized `os.getenv` wrappers
- C: Move all env-driven config into `DocMindSettings` and refactor consumers (Selected)

### Decision Framework (≥9.0)

| Option                               | Leverage (35%) | Value (25%) | Risk Reduction (25%) | Maint (15%) |    Total | Decision                               |
| ------------------------------------ | -------------: | ----------: | -------------------: | ----------: | -------: | -------------------------------------- |
| **C: Move into settings + refactor** |              9 |           8 |                    9 |           9 | **8.75** | ✅ Selected (with simplification loop) |
| B: Env bridge wrappers               |              6 |           6 |                    6 |           6 |      6.0 | Rejected                               |
| A: Status quo                        |              2 |           4 |                    2 |           7 |     3.15 | Rejected                               |

#### Simplification loop to reach ≥9.0

To hit ≥9.0, we keep the change minimal:

- add only the missing settings groups that correspond to existing env variables
- do not introduce new config surfaces or new env var names
- replace the ADR placeholder marker with a real ADR reference and wire the existing hashing secret for safe fingerprints (ADR-047)

Re-scored:

| Option                                                              | Complexity (40%) | Perf (30%) | Alignment (30%) |   Total |
| ------------------------------------------------------------------- | ---------------: | ---------: | --------------: | ------: |
| **C (minimal): Settings groups + refactor + wire hashing secret**   |              9.2 |       10.0 |             9.2 | **9.4** |

## Decision

1. Add missing settings groups:

- `TelemetryConfig` (JSONL local telemetry): disabled flag, sample rate, rotate bytes, path
- `ImageEncryptionConfig` (AES-GCM): key base64, kid, delete_plaintext
- optionally `environment` string (for OTEL resource)

1. Refactor modules to read from `settings` instead of `os.getenv`.

2. Fix the ADR placeholder marker and validator error message for `HashingConfig.hmac_secret` and explicitly use it for keyed fingerprints (ADR-047).

### Env contract (canonical vs derived)

To keep configuration ergonomic without adding bespoke wrappers:

- Canonical operator env vars (stable contract):
  - `DOCMIND_TELEMETRY_DISABLED|SAMPLE|ROTATE_BYTES`
  - `DOCMIND_IMG_AES_KEY_BASE64|IMG_KID|IMG_DELETE_PLAINTEXT`
- Derived/advanced env vars may also exist due to nested settings schema and
  `env_nested_delimiter="__"` (e.g., `DOCMIND_TELEMETRY__JSONL_PATH`).
- Precedence rule when both are present: the canonical flat env vars win (e.g.,
  `DOCMIND_TELEMETRY_DISABLED` overrides `DOCMIND_TELEMETRY__DISABLED`).

## Security & Privacy

- Centralization reduces the risk of “hidden” config toggles.
- Ensure secrets in settings use `repr=False` and are never logged.

## Consequences

### Positive Outcomes

- One configuration source of truth.
- Removes undocumented env behavior and ADR placeholder drift.
- Establishes a safe, local secret suitable for keyed fingerprints (correlation without content).

### Trade-offs

- Touches a few modules; requires careful typing and tests.
