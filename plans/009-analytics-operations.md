# Plan 009: Turn Analytics into a bounded local operations view

> **Executor instructions**: Use only existing local telemetry and preserve the
> analytics feature gate.
>
> **Drift check**:
> `git diff --stat 4fea380..HEAD -- src/pages/03_analytics.py src/ui/analytics_view.py tests/unit/ui/test_analytics_view.py tests/integration/ui/test_app_smoke_flows.py tests/browser/app.spec.ts docs/developers/observability-metrics.md README.md`
> Plan 005 may add shared shell rendering; reconcile that only.

## Status

- **Priority**: P3
- **Effort**: S–M
- **Risk**: LOW
- **Depends on**: Plans 004 and 005
- **Category**: direction
- **Planned at**: commit `4fea380`, 2026-07-16

## Why this matters

Analytics already reads bounded DuckDB/JSONL telemetry and renders aggregate
charts. Small local filters and operational summaries can make failures and
latency understandable without new telemetry or a hosted dashboard.

## Exact mapping

Keep `_load_query_metrics` as the canonical bounded reader and preserve the
disabled state. Recompose `main` into: Overview metrics, latency/query charts,
router selection, stale snapshot/export events, and a bounded local time-window
filter derived from timestamps already present. If a source lacks timestamps,
render the current aggregate with an explicit “all retained events” label; do
not invent time buckets.

## Steps

1. Add pure filter/aggregation helpers in `src/ui/analytics_view.py`; keep file
   reading in the existing page owner. Unit-test empty, bounded, truncated, and
   timestamp-absent inputs.
2. Add native metrics and filters using existing bounded rows/counts. Preserve
   Plotly and feature gate; no refresh loop or unbounded scan.
3. Add AppTest/browser cases for disabled, empty, populated, truncated, desktop,
   mobile, chart overflow, and keyboard filter interaction. Align observability
   docs/README.

## Scope

Only drift-check files. No new telemetry events, schema, external service,
dependency, or network call.

## Git workflow

Use `feat/ui-foundation`; commit `feat(ui): improve local analytics view`. Do
not push/open a PR before parent review.

## Verification

```bash
uv run pytest tests/unit/pages/test_analytics_telemetry_parsing.py tests/unit/ui/test_analytics_view.py tests/integration/ui/test_app_smoke_flows.py -q --no-cov
bun run test:browser
uv run ruff format --check .
uv run ruff check .
uv run pyright --threads 4
uv run pytest tests/unit tests/integration -q --no-cov
uv run python scripts/check_links.py
npx --yes markdownlint-cli@0.47.0 --disable MD013 MD033 MD041 -- README.md docs/developers/observability-metrics.md plans/009-analytics-operations.md
```

Expected: every command exits 0; both browser projects pass.

## Done criteria

- [ ] Feature gate, bounded readers, and existing charts remain.
- [ ] Filters use only evidenced timestamps and never fabricate buckets.
- [ ] No new telemetry, network, dependency, or unbounded scan exists.
- [ ] Desktop/mobile and Python/docs gates pass.

## STOP conditions

Stop if useful filtering requires a telemetry schema migration, network access,
or scanning beyond current caps. Report the missing timestamp/field instead.

## Maintenance notes

New operations panels must reuse bounded telemetry owners; Analytics is a local
view, not a second observability backend.
