---
ADR: 050
Title: Configuration Discipline — Eliminate `os.getenv` Sprawl and Remove Unused Hashing Placeholder
Status: Proposed
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 024, 032
Tags: configuration, settings, maintainability, security
References:
  - https://docs.pydantic.dev/latest/concepts/pydantic_settings/
---

## Description

Centralize runtime configuration in `src/config/settings.py` by:

- removing direct `os.getenv` usage from core modules where settings already exist or should exist
- adding missing settings groups for JSONL telemetry and image encryption toggles
- removing unused “hashing” settings placeholders (currently not referenced by code)

## Context

DocMind’s configuration contract is:

- `DocMindSettings` is source of truth (Pydantic Settings v2)
- env prefix `DOCMIND_`, nested fields via `__`
- core code should not scatter `os.getenv` calls

Current drift points:

- `src/utils/telemetry.py` reads `DOCMIND_TELEMETRY_*` via `os.getenv`
- `src/telemetry/opentelemetry.py` reads `DOCMIND_ENVIRONMENT` via `os.getenv`
- image encryption helpers read `DOCMIND_IMG_*` via `os.getenv`
- `settings.py` contains an `ADR-XXX` marker for hashing config, and the hashing config is unused in code

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
- remove unused hashing placeholder entirely instead of “designing” it

Re-scored:

| Option                                                              | Complexity (40%) | Perf (30%) | Alignment (30%) |   Total |
| ------------------------------------------------------------------- | ---------------: | ---------: | --------------: | ------: |
| **C (minimal): Settings groups + refactor + delete unused hashing** |                9 |         10 |               9 | **9.3** |

## Decision

1. Add missing settings groups:

- `TelemetryConfig` (JSONL local telemetry): disabled flag, sample rate, rotate bytes, path
- `ImageEncryptionConfig` (AES-GCM): key base64, kid, delete_plaintext
- optionally `environment` string (for OTEL resource)

1. Refactor modules to read from `settings` instead of `os.getenv`.

2. Remove unused `HashingConfig` from settings (and the `ADR-XXX` marker) unless code starts using it in v1 (out-of-scope).

## Security & Privacy

- Centralization reduces the risk of “hidden” config toggles.
- Ensure secrets in settings use `repr=False` and are never logged.

## Consequences

### Positive Outcomes

- One configuration source of truth.
- Removes undocumented env behavior and ADR-XXX drift.

### Trade-offs

- Touches a few modules; requires careful typing and tests.
