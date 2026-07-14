---
spec: SPEC-014
title: Collection Activation Snapshots (SnapshotManager)
version: 2.2.0
date: 2026-07-14
owners: ["ai-arch"]
status: Accepted
related_requirements:
  - FR-009.2: SnapshotManager persistence + manifest + lock
  - FR-009.5: Staleness badge integration
related_adrs: ["ADR-038","ADR-031","ADR-034","ADR-019"]
---

## Objective

Define atomic, versioned activation manifests for immutable physical Qdrant
collections and optional property-graph artifacts. Qdrant backups own
point-in-time vector data. The app snapshot provides collection identity,
corpus/config hashes, GraphRAG export metadata, and single-writer activation.

## Filesystem layout

```text
storage/
  errors.jsonl         # Redacted persistence failures outside finalized snapshots
  .activation-transaction.json # Durable promotion intent while activation is pending
  _tmp-<uuid>/
    graph/
      docstore.json                 # Native LlamaIndex graph context
      index_store.json
      graph_store.json
      property_graph_store.json
      default__vector_store.json    # Property-graph node embeddings
      image__vector_store.json
      graph_export-*                # Optional JSONL/Parquet exports
    errors.jsonl       # Possible aborted-workspace graph persistence record
    manifest.jsonl     # Line-delimited entries for payload files (path + hash + metadata)
    manifest.meta.json # Manifest metadata (schema below, complete=false while staging)
    manifest.checksum  # Aggregate digest covering entries + metadata
  <timestamp>/         # Finalized snapshot (atomic rename from _tmp)
  CURRENT              # Pointer file referencing the active snapshot directory (authoritative)
```

## Manifest schema

```json
{
  "index_id": "string",
  "graph_store_type": "property_graph",
  "vector_store_type": "qdrant",
  "collections": {
    "text": "docmind_docs_build-<id>",
    "image": "docmind_images_build-<id>"
  },
  "collection_metadata": {
    "text": {"docmind_schema_version": "2"},
    "image": {"docmind_schema_version": "1"}
  },
  "corpus_hash": "<64 lowercase hexadecimal characters>",
  "config_hash": "<64 lowercase hexadecimal characters>",
  "created_at": "YYYY-MM-DDTHH:MM:SSZ",
  "schema_version": "2.0",
  "persist_format_version": "1.0",
  "complete": true,
  "versions": {
    "app": "x.y.z",
    "llama_index": "x.y.z",
    "qdrant_client": "x.y.z",
    "vector_client": "qdrant",
    "embed_model": "<model-id>"
  },
  "activation_config": {},
  "activation_config_hash": "<64 lowercase hexadecimal characters>",
  "graph_exports": [
    {
      "filename": "graph_export-snapshot-YYYYMMDDTHHMMSSZ.jsonl",
      "format": "jsonl",
      "seed_count": 32,
      "size_bytes": 1234,
      "duration_ms": 12.5,
      "sha256": "<64 lowercase hexadecimal characters>",
      "created_at": "YYYY-MM-DDTHH:MM:SSZ"
    }
  ]
}
```

- `manifest.jsonl` contains one JSON object per payload file with `path` (full workspace-relative path, e.g., `graph/export.jsonl`), `sha256`, `size_bytes`, and `content_type` fields.
- `manifest.meta.json` is written alongside the JSONL entries with `complete=false` while the workspace is still staged and flipped to `true` immediately after the atomic rename succeeds.
- `manifest.checksum` stores `{schema_version, created_at, manifest_sha256}`.
  The digest covers canonical JSON for every complete manifest entry in emitted
  order, including a newline delimiter after each entry, followed by the full
  canonical metadata object.
- `graph_exports` enumerates GraphRAG export artifacts packaged with the snapshot, capturing telemetry metadata (`seed_count`, `duration_ms`, `size_bytes`, `sha256`).
- `collections.text` and `collections.image` are required physical collection
  identities. They are never resolved from mutable runtime settings during load.
- `collection_metadata` records any immutable semantic identity copied from the
  activated collections.
- `graph_store_type=property_graph` requires the exact six-file native
  `StorageContext` shown above plus only exports declared by `graph_exports`.
  Export metadata size and SHA-256 must match its manifest payload row.
  `graph_store_type=none` forbids graph payloads and export metadata.
  `vector_store_type` is exactly `qdrant`. The graph-local SimpleVectorStore
  preserves property-graph node embeddings; it is not a copy of the live Qdrant
  text or image collections.
- `corpus_hash`, `config_hash`, and `activation_config_hash` are bare lowercase
  64-character SHA-256 values. `activation_config_hash` must equal the canonical
  hash of the complete `activation_config` object.

> **Note**: The `filename` field in `graph_exports` entries contains the export file basename only (for example, `graph_export-snapshot-20260109T120000Z.jsonl`). The `graph/` subdirectory location is implicit. This differs from `manifest.jsonl` payload entries, which use a `path` field containing the full workspace-relative path.

## Hashing rules

- corpus_hash: stable SHA‑256 over each canonical POSIX relative path and full
  file-content SHA‑256 for the activated corpus
- config_hash: stable SHA‑256 over canonical JSON of retrieval/chunking/embedding settings and toggles

## Atomicity and locking

- Write all artifacts under `storage/_tmp-<uuid>` and `fsync` + atomic rename to `storage/<timestamp>`.
- Acquire a single-writer operating-system lock on the permanent
  `storage/.lock` sentinel with required `portalocker`. Persist diagnostic
  metadata and refresh its heartbeat during long-running work. Contenders MUST
  ignore heartbeat age; missing operating-system lock support fails closed.
- Update `CURRENT.tmp` → `CURRENT` via atomic rename as the final commit point.
  Readers MUST return no active snapshot when `CURRENT` is absent, malformed, or
  targets an incomplete or invalid manifest. Directory ordering is never an
  activation source.
- Persistence failures append redacted structured records to
  `storage/errors.jsonl`. A graph-persistence failure may first write an
  `errors.jsonl` record inside the aborted workspace; caller cleanup removes that
  uncommitted workspace.

## UI Integration

- Documents page: show snapshot path after ingest; provide rebuild action and export summary from `graph_exports`.
- Chat page: read latest manifest metadata, compute hashes, show a staleness badge
  when mismatched, and direct the operator to rebuild in Documents

## Acceptance Criteria

```gherkin
Feature: Atomic snapshots with manifest
  Scenario: Persist a new snapshot
    Given immutable physical text and image collections and an optional property graph
    When I create a snapshot
    Then storage/<timestamp> SHALL contain the manifest triad and optional graph artifacts
    And manifest.meta.json SHALL identify both physical collections
    And the write SHALL occur under a lock and end with an atomic rename

  Scenario: Reload GraphRAG after restart
    Given a graph-enabled snapshot with its native StorageContext
    When Chat reloads the active snapshot after process restart
    Then property-graph relations and graph vector retrieval SHALL remain available

  Scenario: Staleness detection
    Given a snapshot manifest with corpus_hash and config_hash
    And current corpus/config that differ
    When I open Chat
    Then a staleness badge SHALL be visible
```

## References

- ADR‑038 GraphRAG Persistence and Router Integration

## Manifest fields

- Require `schema_version`, `persist_format_version`, `complete`, `created_at`,
  `index_id`, `graph_store_type`, `vector_store_type`, `collections.text`,
  `collections.image`, `collection_metadata`, `corpus_hash`, `config_hash`,
  `versions`, `graph_exports`, `activation_config`, and
  `activation_config_hash`.
- Write `manifest.jsonl`, `manifest.meta.json`, and `manifest.checksum` together; `manifest.json` SHALL NOT be emitted going forward.
- Record GraphRAG export metadata under `graph_exports` (filename, format,
  seed_count, size_bytes, duration_ms, sha256, created_at). Snapshot exports use
  `graph_export-snapshot-YYYYMMDDTHHMMSSZ.<ext>` and are written before manifest
  hashing.
- Set `complete=true` only after the workspace renames into its immutable directory; prior to rename the field SHALL remain `false` and consumers MUST ignore incomplete manifests.
- Persist redacted structured failure records with stage, snapshot ID, error
  type, error code, and message at `storage/errors.jsonl`. An aborted workspace
  record is not part of a finalized snapshot.

## Relative-path hashing

- Compute `corpus_hash` using POSIX relpaths relative to `uploads/` for OS-agnostic stability.
- Compute `config_hash` from a canonical JSON structure with sorted keys, normalized booleans, and sanitized paths. The same helper MUST be reused by SnapshotManager, UI staleness checks, and tests to guarantee determinism.
- ADR‑031 Local‑first Persistence Architecture
- ADR‑034 Idempotent Indexing and Embedding Reuse

## Lock metadata and heartbeats

- Single-writer locks SHALL require `portalocker`. The operating-system lock on
  the permanent `.lock` sentinel is the sole ownership authority. Missing OS-lock
  support MUST fail closed; lease-file or `os.O_EXCL` fallbacks are forbidden.
- Each lock acquisition writes a JSON sidecar `<lock>.meta.json` containing:
  - `owner_id`: `<pid>@<hostname>` identifying the holder.
  - `created_at` / `last_heartbeat`: UTC timestamps (ISO8601) updated on every refresh.
  - `ttl_seconds`: client-configured lease duration.
  - `takeover_count`: schema-compatible diagnostic value (`0`).
  - `schema_version`: metadata schema revision (currently `1`).
- Holders refresh `last_heartbeat` at least every `ttl_seconds / 2` for operator
  diagnostics. Contenders MUST ignore heartbeat age and defer exclusively to the
  OS lock. Process exit releases ownership without replacing or unlinking the sentinel.
- All metadata writes use write-to-temp + `os.replace` + `fsync` to guarantee durability on POSIX systems. Parent directories SHOULD be fsynced where supported; on Windows best-effort semantics apply.
- A failure after OS-lock acquisition but before metadata/heartbeat initialization
  MUST release the OS lock before propagating the error.

## Promotion and `CURRENT` pointer discipline

- `_tmp-` workspaces and final snapshot directories MUST reside on the same filesystem (validated via `stat().st_dev`) to preserve `os.replace` atomicity.
- Promotion sequence:
  1. Write the payload and `complete=false` manifest triad, then `fsync` them.
  2. Write and `fsync` `.activation-transaction.json` for the exact destination.
  3. Atomically rename `_tmp-<uuid>` to `<timestamp>`.
  4. Set `complete=true`, refresh the checksum, `fsync`, and verify the complete
     closed-world payload and manifest.
  5. Write and `fsync` `CURRENT.tmp`, then atomically replace `CURRENT`. This is
     the commit point.
  6. Clear the activation journal and run best-effort manifest retention.
- Readers MUST resolve only `CURRENT`. Recovery may discard an invalid pointer
  and crash debris but MUST NOT infer or promote an unreferenced directory.
- Recovery deletes the journaled destination when `CURRENT` did not commit. If
  `CURRENT` names the same verified destination, recovery keeps it and retires
  the journal.

## Retention and cleanup

- Retain the latest `settings.snapshots.retention_count` finalized directories,
  never deleting the directory referenced by `CURRENT`.
- `_tmp-*` workspaces and stale lock artifacts are removed after the writer lock
  is acquired. Manager construction performs no unlocked recovery writes.
- SnapshotManager MUST run retention under the same lock to avoid concurrent deletions during promotion.
- Online activation MUST NOT delete physical Qdrant collections because readers
  do not hold collection leases. Cleanup is an explicit, dry-run-first offline
  command after every DocMind reader and writer is stopped. It MUST fail closed
  on any unverified retained snapshot or invalid `CURRENT`, preserve every retained
  manifest reference, and delete only collections whose metadata exactly matches
  `data/.deployment-id` and a supported DocMind owner.
- `data/.deployment-id` is created atomically only when no durable snapshot state
  exists. Missing, invalid, symlinked, or noncanonical identity state fails closed
  once `CURRENT` or a canonical snapshot exists.
- Upload promotion and quarantine journals bind every source move to its SHA-256
  and exact physical text/image collection generation. Recovery runs under the
  snapshot writer lock. It commits only the generation named by `CURRENT`, rolls
  back an uncommitted generation, and preserves unexplained or changed state for
  operator review.

## Telemetry and observability

- Snapshot operations emit OpenTelemetry spans `snapshot.write_manifest`,
  `snapshot.persist_graph`, `snapshot.finalize`, and `snapshot.retention`. Only the
  retention span currently sets an attribute:
  `snapshot.retention.deleted_count`.
- Graph export operations (manual and snapshot-triggered) emit local
  `export_performed` events. The events carry the manifest identity measurements
  with telemetry-specific fields such as `dest_basename`, `export_type`, and
  `context`.
- Lock acquisition timeout and metadata initialization failures emit structured,
  redacted diagnostics without weakening OS-lock ownership.

## Changelog

- 2.1.0 (2026-07-14): Required OS-fenced portalocker ownership, closed-world
  manifests, activation/source crash journals, native property-graph persistence,
  and deployment-scoped offline collection cleanup.
- 2.0.0 (2026-07-14): Made physical Qdrant collection identity explicit,
  removed serialized vectors from app snapshots, and made `CURRENT` the sole
  activation boundary and final commit point.
- 1.1.0 (2025-09-16): Documented portalocker-based locking, graph export metadata, `errors.jsonl`, and removal of legacy `manifest.json`.
- 1.0.0 (2025-09-09): Initial draft spec

## UI staleness badge

- The Chat page MUST show a visible staleness badge when the current manifest differs from the latest content/config hashes.
- Tooltip copy MUST be exactly:
  - “Snapshot is stale (content/config changed). Rebuild in Documents → Rebuild search index.”
- The badge MUST be computed from local state (manifest vs current hashes) and MUST NOT trigger any network calls.

## Lock semantics (single writer)

- SnapshotManager MUST implement a single-writer OS lock backed by required
  `portalocker`; metadata heartbeats are diagnostic and never authorize takeover.
  Timeout errors MUST surface clear user messaging.
- While rebuilding, Documents ingestion controls MUST reflect a locked state and
  prevent concurrent rebuilds.

## Atomic rename

- Writes MUST occur to a temporary directory name with `_tmp-<uuid>` suffix and be atomically renamed to a timestamped final directory on success.
- On failure, incomplete `_tmp-*` directories MUST be cleaned up on next run.

## Acceptance and UX mapping

- Documents page MUST provide:
  - Rebuild button (lock‑aware) and current snapshot path display.
  - Exports section (JSONL baseline; Parquet optional with `pyarrow`) and seed cap display. Snapshot export artifacts MUST use `graph_export-snapshot-YYYYMMDDTHHMMSSZ.*`, remain under `graph/`, and emit telemetry events recording destination basenames, seed counts, file sizes, and durations.
- Chat page MUST show the staleness badge and tooltip per UI Staleness Badge above.
