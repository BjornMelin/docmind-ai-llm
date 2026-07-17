---
spec: SPEC-032
title: Documents Snapshot Service Boundary (Snapshot Rebuild/Export Service)
version: 2.3.0
date: 2026-07-16
owners: ["ai-arch"]
status: Implemented
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
- Adding new export formats.

## Technical Design

### New module: `src/persistence/snapshot_service.py`

Add a small, typed orchestration service that:

- receives a caller-owned `SnapshotManager` and pre-opened workspace
- records the immutable physical text/image Qdrant collection identities
- persists the complete native property-graph `StorageContext` when present
  (`persist_graph_storage_context`), including the graph-local vector store
  required for semantic graph retrieval after restart
- optionally packages graph exports into the workspace under `graph/`:
  - JSONL required
  - Parquet optional (PyArrow-gated)
- computes and records export metadata:
  - export filename basename (`graph/` is implicit)
  - format
  - `seed_count`, `size_bytes`, `duration_ms`
  - `sha256`
- computes corpus/config hashes and writes manifest metadata (`SnapshotManager.write_manifest`)
- finalizes the snapshot atomically (`SnapshotManager.finalize_snapshot`)

The caller acquires the snapshot lock and opens the workspace before building
physical collections. It retains cleanup ownership until finalization commits.
The service never begins or cleans up a workspace.

### Service API (minimal)

- `rebuild_snapshot(vector_index, pg_index, settings_obj, activation,
  commit_source_changes=..., ...) -> FinalizedSnapshot`
- `SnapshotActivation(manager, workspace, text_collection, image_collection,
  expected_corpus_hash, expected_config_hash, activation_config,
  activation_config_hash, collection_metadata, graph_requested)`

The return value carries the committed snapshot directory and the manifest
verified during finalization as `FinalizedSnapshot(path, manifest)`. Callers build
result presentation from that manifest without reopening the result snapshot.
`text_collection` and `image_collection` are required physical identities;
mutable settings names are not used as activation fallbacks. `manager` must hold
the transaction lock and `workspace` must be the workspace opened before
collection construction. The required `expected_corpus_hash` is the adapter's
content-derived identity for the entire selected upload corpus, including the
empty corpus. The service prepares graph payloads and the incomplete manifest
before calling the durable late-source commit callback. It then recomputes the
canonical uploads-relative identity and rejects any mismatch before finalization.
Promotion, quarantine, and snapshot activation journals recover crashes without
inferring a replacement `CURRENT`.

`expected_config_hash` identifies current global index-affecting settings for
staleness. `activation_config` and `activation_config_hash` preserve exact build
provenance, including one-off parser, encryption, and GraphRAG choices, without
making a deliberate one-off choice immediately stale.

### Snapshot manifest schema (manifest.meta.json)

The service must write manifest metadata with the following fields (matching `SnapshotManager.write_manifest`):

- Required: `index_id`, `graph_store_type`, `vector_store_type`, `collections`,
  `corpus_hash`, `config_hash`, `versions`, `collection_metadata`,
  `graph_exports`, `activation_config`, `activation_config_hash`

`corpus_hash` is the canonical relative-path plus full-content identity.
`config_hash` identifies current global index-affecting settings.
`activation_config_hash` covers the exact effective build provenance. All hashes
use `src/persistence/hashing.py` and are lowercase 64-character SHA-256 values.
`versions` uses string keys and scalar-or-null values. The canonical persistence
validator rejects every other shape before snapshot publication.

When present, `graph_exports` entries include export metadata such as:

- `filename`, `format`, `seed_count`, `size_bytes`, `duration_ms`, `sha256`
- `created_at` (ISO timestamp) when available

Canonical export identity requires a unique basename `filename` of at most 200
characters, a nonempty `format` of at most 32 characters, an exact nonnegative
integer `size_bytes`, and a lowercase SHA-256. `graph_store_type=none` requires an
empty export list. Property-graph exports must also match their payload entries.

### Documents page responsibilities

`src/pages/02_documents.py` becomes:

- transaction-scoped ingestion UI wiring (`ingest_inputs(…)`)
- caller-side snapshot transaction acquisition and failure cleanup
- storing indices/router in `st.session_state`
- calling `snapshot_service.rebuild_snapshot(…)` and rendering results
- preparing runtime ownership with `_prepare_ingest_runtime` and rendering only
  its object-free DTO with `_render_ingest_presentation`; no combined duplicate
  helper is maintained

Physical collection deletion is not a page or activation responsibility. The
offline cleanup CLI requires a quiesced application, a dry-run review, verified
retained manifests, and the exact local deployment identity.

### Import-time performance

The service and page must avoid heavy imports at module import time:

- move LlamaIndex/Qdrant imports inside functions where possible
- keep Streamlit wiring testable through focused page helpers and AppTest

## Observability

- Continue emitting local telemetry for export events (JSONL) via `src/utils/telemetry.log_jsonl`.
- Continue recording graph export metrics via `src/telemetry/opentelemetry.record_graph_export_metric` when meters are configured.
- Telemetry failures (JSONL write or OTel metric recording) must be logged and skipped so snapshot rebuilds proceed; no retries beyond the current attempt.

## Security

- No new filesystem write locations beyond snapshot workspace.
- Export paths are workspace-relative; no user-provided paths are used.
- Never log secrets or raw document contents.

## Test evidence

### Unit

- `tests/unit/persistence/test_snapshot_service.py` proves manifest fields,
  caller-owned workspace lifetime, graph persistence/exports, source commit order,
  and corpus/config identity checks.
- `tests/unit/pages/test_documents_upload_transactions.py` proves pending upload
  promotion, quarantine, rollback, and activation journal wiring.
- `tests/unit/pages/test_documents_page_helpers.py` covers the remaining thin page
  helpers without importing a second persistence implementation.

### Integration (AppTest)

- `tests/integration/ui/test_documents_ingestion_job.py` proves the Documents
  page schedules and completes the background ingestion transaction.
- `tests/integration/test_graphrag_exports.py` proves native property-graph
  export behavior and snapshot packaging.

## Rollout / Migration

- This is a v2 hard cut. Rebuild v1 snapshots into physical collections; v1
  manifests do not satisfy the activation contract.

## Traceability

`docs/specs/traceability.md` maps this boundary to
`src/persistence/snapshot_service.py`, the current transaction tests, and the
Documents ingestion AppTest evidence listed above.
