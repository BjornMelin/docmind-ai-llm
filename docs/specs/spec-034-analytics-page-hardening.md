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
  - `max_lines` (default 50_000)
  - `max_bytes` (default 25 MB)

Rationale: defaults target a bounded ~25 MB parse window based on typical daily telemetry volume; adjust only after measuring log growth (estimate `max_lines ≈ target_bytes / avg_line_bytes` from sample logs).

Scope: caps apply per parse invocation (each call enforces its own `max_lines` and `max_bytes` limits independently), keeping behavior deterministic without session-wide state.

Recourse: when a cap is hit, stop reading additional lines/bytes, log a warning, and return aggregated counts from the bounded window (no exceptions raised for cap hits).

### Canonical telemetry path

Use the same path as the telemetry emitter by referencing a shared public constant or getter (e.g., `TELEMETRY_JSONL_PATH` or `get_telemetry_jsonl_path()` in `src/utils/telemetry.py`) from the Analytics page. Add a test that asserts the analytics helper reads from the same path.

## Observability

No new telemetry; this is a consumer-only hardening.

## Security

- Only aggregated counts displayed.
- Do not display raw telemetry event payloads.

## Testing Strategy

- Unit tests for telemetry parsing helper (`tests/unit/utils/test_telemetry_parsing.py`):
  - parse valid JSONL lines and aggregate counts
  - skip invalid JSON lines without raising
  - enforce `max_lines` and `max_bytes` caps
- Integration tests for Analytics page (`tests/integration/test_pages_analytics.py`):
  - DuckDB connection closes normally and on query exception
  - Analytics page import has no side effects
  - Telemetry helper uses the canonical telemetry path
- Keep `tests/integration/test_pages_smoke.py` passing.

## RTM Updates

Update `docs/specs/traceability.md`:

- add `src/pages/03_analytics.py` hardening under the relevant observability/UX rows (or create a new planned row under NFR-OBS-001 if needed).
