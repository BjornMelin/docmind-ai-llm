---
spec: SPEC-029
title: Documentation Consistency Pass (Specs, Handbook, RTM)
version: 1.1.0
date: 2026-07-11
owners: ["ai-arch"]
status: Implemented
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
     - remove/replace references to placeholder ingestion APIs and align with `src/processing/ingestion_api.py`
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

## Implemented documentation gates

The repository uses focused checks rather than a general inline source-path
scanner:

- `scripts/check_links.py` validates internal Markdown links.
- `scripts/verify_structural_parity.py` compares the documented top-level
  package manifest with `src/`.
- `scripts/validate_schemas.py` validates repository schemas.
- Markdownlint validates active Markdown files.

Inline `src/...` references remain review-owned because design records may
intentionally describe deleted files or conceptual examples. No inline-path
allowlist is part of the implemented contract.

## Testing strategy

- Unit tests cover schema validation.
- Documentation CI runs link, structural-parity, and Markdown checks.
- Reviewers verify changed inline code references against the live tree.

## Rollout / migration

- Docs-only changes; no runtime impact.

## RTM updates (docs/specs/traceability.md)

NFR-MAINT-003 records the implemented link, schema, structural-parity, and
Markdown quality gates. It does not claim general inline source-path scanning.
