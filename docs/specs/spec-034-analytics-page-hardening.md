---
spec: SPEC-034
title: Analytics Page Hardening (DuckDB + Telemetry Parsing)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - FR-010: Streamlit multipage UI includes Analytics
  - NFR-OBS-001: Local observability is available without network
related_adrs: ["ADR-053", "ADR-032"]
---

## Goals

1. Close DuckDB connections deterministically in the Analytics page.
2. Parse telemetry JSONL in a streaming, bounded way.
3. Use canonical paths from settings/telemetry utilities.
4. Keep the page importable without side effects.

## Non-goals

- Adding new analytics schemas or metrics tables.
- Adding network exporters or remote analytics.
- Building a full analytics dashboard UI redesign.

## Technical Design

### DuckDB lifecycle

Update `src/pages/03_analytics.py` to use a context manager or `try/finally`:

- open connection
- query to DataFrames
- close connection

### Telemetry JSONL parsing helper

Add helper (module-local or reusable) that:

- iterates lines from the telemetry JSONL file
- ignores invalid JSON lines
- aggregates counts for:
  - router_selected route counts
  - snapshot_stale_detected count
  - export_performed count
- enforces caps:
  - `max_lines` (default e.g. 50_000)
  - `max_bytes` (default e.g. 25MB)

### Canonical telemetry path

Use the same path as the telemetry emitter (expose a public getter/constant from `src/utils/telemetry.py`), rather than hardcoding `./logs/telemetry.jsonl`.

## Observability

No new telemetry; this is a consumer-only hardening.

## Security

- Only aggregated counts displayed.
- Do not display raw telemetry event payloads by default.

## Testing Strategy

- Unit tests for the telemetry parsing helper.
- Keep `tests/integration/test_pages_smoke.py` passing.

## RTM Updates

Update `docs/specs/traceability.md`:

- add `src/pages/03_analytics.py` hardening under the relevant observability/UX rows (or create a new planned row under NFR-OBS-001 if needed).
