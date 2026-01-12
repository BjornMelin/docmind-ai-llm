---
ADR: 053
Title: Analytics Page Hardening (DuckDB Lifecycle + Telemetry Parsing)
Status: Implemented
Version: 1.1
Date: 2026-01-12
Supersedes:
Superseded-by:
Related: ADR-032 (local analytics/metrics), ADR-013 (UI architecture), ADR-016 (Streamlit state)
Tags: streamlit, analytics, duckdb, telemetry
References:
  - https://duckdb.org/docs/
---

## Description

Refactor the Streamlit Analytics page to use safe DuckDB connection lifecycle, remove dynamic imports, and parse local telemetry JSONL efficiently using canonical paths.

## Context

Prior to this change, `src/pages/03_analytics.py`:

- opens a DuckDB connection without closing it
- uses `__import__("pandas")` dynamically
- reads `./logs/telemetry.jsonl` via hardcoded path and reads the file twice
- loads the entire telemetry file into memory

This risks file handle leaks, unnecessary memory use, and drift from the telemetry emitter path.

## Decision Drivers

- Keep the Analytics page safe and deterministic (offline-first)
- Avoid resource leaks (DuckDB connection)
- Prefer simple, explicit imports and bounded parsing
- Reuse canonical settings/telemetry paths

## Alternatives

- A: Keep current behavior (Rejected)
- B: Refactor page with helpers + safe resource handling (Selected)
- C: Remove Analytics page entirely (too disruptive; conflicts with FR-010 navigation)

### Decision Framework (≥9.0)

Weights: Complexity 40% · Perf 30% · Alignment 30% (10 = best)

| Option                                  | Complexity (40%) | Perf (30%) | Alignment (30%) |    Total |
| --------------------------------------- | ---------------: | ---------: | --------------: | -------: |
| **B: Safe lifecycle + bounded parsing** |              9.5 |        9.0 |             9.5 | **9.35** |
| A: status quo                           |             10.0 |        3.0 |             4.0 |      6.1 |
| C: remove page                          |              6.0 |       10.0 |             5.0 |      6.9 |

## Decision

We will:

1. Ensure DuckDB connections are closed deterministically (context manager or `try/finally`).
2. Replace dynamic `__import__` with explicit `import pandas as pd` (Analytics already depends on pandas/plotly).
3. Add a small telemetry parsing helper that:
   - streams JSONL lines (no full-file `.read_text().splitlines()` for large files)
   - applies both `max_bytes` and `max_lines` caps (stop parsing and log a warning on cap)
4. Use a canonical telemetry path shared with the emitter by exporting a public constant/getter from `src/utils/telemetry.py` (e.g., `TELEMETRY_JSONL_PATH`).
5. Use a canonical analytics DuckDB path constant/getter from `src/utils/telemetry.py` (default `data/analytics/analytics.duckdb`).

## Security & Privacy

- Analytics stays local-only.
- Telemetry parsing must not surface secrets; only aggregate counts.
- See ADR-047 for safe logging patterns; telemetry must follow metadata-only + keyed HMAC approach.

## Testing

- Unit tests for telemetry parsing helper (valid JSON, invalid JSON, caps) and ensure DuckDB connections close on success and on exceptions.
- Cap enforcement tests verify parsing stops at `max_lines`/`max_bytes` and emits a warning (best-effort).
- Smoke import test for Analytics page remains green and gracefully handles missing `analytics.duckdb`.

## Changelog

- 1.0 (2026-01-09): Proposed for v1 correctness and resource safety.
- 1.1 (2026-01-12): Implemented bounded parsing + deterministic DuckDB close + canonical paths.
