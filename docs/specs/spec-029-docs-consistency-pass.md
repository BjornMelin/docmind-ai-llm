---
spec: SPEC-029
title: Documentation Consistency Pass (Specs, Handbook, RTM)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - NFR-MAINT-003: Docs must match shipped code/APIs.
related_adrs: ["ADR-048"]
---

## Objective

Restore documentation correctness for v1 by aligning:

- `docs/specs/*` with current modules and entrypoints
- developer docs with canonical APIs (especially ingestion)
- changelog + ADR numbering consistency
- RTM links and references

## Non-goals

- Full rewrite of every developer document
- Writing new tutorials beyond fixing drift

## Scope (targeted drift fixes)

1. **Ingestion spec drift**

   - Update `docs/specs/spec-002-ingestion-pipeline.md`:
     - replace references to removed `src/processing/document_processor.py` with actual `src/processing/ingestion_pipeline.py`
     - ensure examples align with the canonical ingestion API introduced in SPEC-026

2. **Developer handbook drift**

   - Update `docs/developers/developer-handbook.md`:
     - remove/replace references to placeholder `src.utils.document.process_document` and similar removed APIs
     - prefer canonical ingestion API under `src/processing/`

3. **Architecture maps**

   - Update `docs/developers/system-architecture.md` to reflect actual modules (remove references to deleted/dead files).

4. **Changelog / ADR backfill**

   - Ensure CHANGELOG references correspond to actual ADR files (ADR-039/040 already added; verify no other missing numbers).

5. **RTM link consistency**

   - Ensure `docs/specs/traceability.md` references correct ADR/SPEC IDs and real file paths.

6. **Observability spec drift**

   - Update `docs/specs/spec-012-observability.md` to match current `ObservabilityConfig` schema and `src/telemetry/opentelemetry.py` behavior.
   - Ensure the SRS includes `NFR-OBS-*` requirements referenced by SPEC-012 (add if missing).

## Automated drift check (lightweight)

Add a small script (or extend an existing health script) to detect doc drift:

- scan non-archived docs:
  - include: `docs/specs/`, `docs/developers/`
  - exclude: `docs/**/archived/`, `docs/specs/.archived/`
- detect referenced paths matching `src/<path>.py`
- assert each referenced file exists

Wire it into `scripts/run_quality_gates.py` (or `scripts/test_health.py --check-patterns`) so CI catches regressions.

## Testing strategy

- Unit test for the drift checker with a small fixture corpus (optional).
- Verification: quality gates run the drift checker and fail on missing references.

## Rollout / migration

- Docs-only changes; no runtime impact.

## RTM updates (docs/specs/traceability.md)

Add a planned row:

- NFR-MAINT-003: “Docs drift fixed + drift checker added”
  - Code: `docs/specs/*`, `docs/developers/*`, `scripts/*`
  - Tests: optional unit test for checker
  - Verification: quality gates
  - Status: Planned → Implemented
