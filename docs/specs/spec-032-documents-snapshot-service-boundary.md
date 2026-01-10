---
spec: SPEC-032
title: Documents Snapshot Service Boundary (Snapshot Rebuild/Export Service)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - NFR-MAINT-001: Library-first layering and reuse
  - NFR-MAINT-003: No drift between specs and shipped code
related_adrs: ["ADR-051", "ADR-031", "ADR-038", "ADR-013", "ADR-016"]
---

## Goals

1. Move snapshot rebuild/export packaging out of `src/pages/02_documents.py` into a persistence-layer service.
2. Keep Streamlit page code as wiring-only (UI inputs → service call → render output).
3. Make snapshot rebuild behavior unit-testable without importing Streamlit pages.

## Non-goals

- Introducing background execution/progress/cancellation (handled separately).
- Adding new export formats or changing existing snapshot schema.
- Changing snapshot locking/atomicity semantics.

## Technical Design

### New module: `src/persistence/snapshot_service.py`

Add a small, typed orchestration service that:

- begins a snapshot workspace (`SnapshotManager.begin_snapshot`)
- persists vector index (`persist_vector_index`)
- persists graph store when present (`persist_graph_store`)
- optionally packages graph exports into the workspace under `graph/`:
  - JSONL required
  - Parquet optional (PyArrow-gated)
- computes and records export metadata:
  - relative path (within workspace)
  - format
  - `seed_count`, `size_bytes`, `duration_ms`
  - `sha256`
- computes corpus/config hashes and writes manifest metadata (`SnapshotManager.write_manifest`)
- finalizes the snapshot atomically (`SnapshotManager.finalize_snapshot`)
- cleans up workspace on failure (`SnapshotManager.cleanup_tmp`)

### Service API (minimal)

- `rebuild_snapshot(*, vector_index: Any, pg_index: Any | None, settings_obj: Any) -> SnapshotRebuildResult`

Where `SnapshotRebuildResult` includes:

- `snapshot_dir: Path`
- `manifest_meta: dict[str, Any]` (or `None` if caller prefers to reload from disk)
- `graph_exports: list[dict[str, Any]]`

### Snapshot manifest schema (manifest.meta.json)

The service must write manifest metadata with the following fields (matching `SnapshotManager.write_manifest`):

- Required: `index_id`, `graph_store_type`, `vector_store_type`, `corpus_hash`, `config_hash`, `versions`
- Optional: `graph_exports`

`corpus_hash` and `config_hash` are computed via `compute_corpus_hash` and `compute_config_hash`
(`src/persistence/hashing.py`) using the uploaded corpus paths and the current settings config.

When present, `graph_exports` entries include export metadata such as:

- `path` (workspace-relative), `format`, `seed_count`, `size_bytes`, `duration_ms`, `sha256`
- `created_at` (ISO timestamp) when available

### Documents page responsibilities

`src/pages/02_documents.py` becomes:

- ingestion UI wiring (`ingest_files(...)`)
- storing indices/router in `st.session_state`
- calling `snapshot_service.rebuild_snapshot(...)` and rendering results

### Import-time performance

The service and page must avoid heavy imports at module import time:

- move LlamaIndex/Qdrant imports inside functions where possible
- keep Streamlit page importable for `tests/integration/test_pages_smoke.py`

## Observability

- Continue emitting local telemetry for export events (JSONL) via `src/utils/telemetry.log_jsonl`.
- Continue recording graph export metrics via `src/telemetry/opentelemetry.record_graph_export_metric` when meters are configured.

## Security

- No new filesystem write locations beyond snapshot workspace.
- Export paths are workspace-relative; no user-provided paths are used.
- Never log secrets or raw document contents.

## Testing Strategy

### Unit

- Move snapshot rebuild tests from `tests/unit/ui/test_documents_snapshot_utils.py` to a new persistence-focused test module (or update the existing tests to import the service directly).
- Use lightweight stub indices that expose `storage_context.persist`.

### Integration (AppTest)

- Update `tests/integration/ui/test_documents_snapshot_button.py` to stub the service rather than patching per-function exports inside the page.

## Rollout / Migration

- Pure refactor; no user data migration.
- Rollback by reverting the new service module and restoring page-local snapshot rebuild.

## RTM Updates

Update `docs/specs/traceability.md` row `FR-009` (GraphRAG persistence/snapshots):

- add `src/persistence/snapshot_service.py` to Code file(s)
- move unit tests to service module
