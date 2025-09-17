---
spec: SPEC-014
title: Index Persistence Snapshots (SnapshotManager)
version: 1.2.0
date: 2025-09-16
owners: ["ai-arch"]
status: Accepted
related_requirements:
  - FR-009.2: SnapshotManager persistence + manifest + lock
  - FR-009.5: Staleness badge integration
related_adrs: ["ADR-038","ADR-031","ADR-034","ADR-019"]
---

## Objective

Define atomic, versioned snapshot persistence for indices and the property graph store. Provide a manifest with corpus/config hashes for staleness detection, GraphRAG export telemetry, and a lockfile with TTL metadata to guarantee single‑writer safety across platforms.

## Filesystem Layout

```text
storage/
  _tmp-<uuid>/
    vector/            # StorageContext.persist output for vector index(es)
    graph/             # SimplePropertyGraphStore persist output + graph_export-* artifacts
    errors.jsonl       # Structured error records when persistence/export fails (optional)
    manifest.jsonl     # Line-delimited entries for payload files (path + hash + metadata)
    manifest.meta.json # Manifest metadata (schema below, complete=false while staging)
    manifest.checksum  # Aggregate digest covering entries + metadata
  <timestamp>/         # Finalized snapshot (atomic rename from _tmp)
  CURRENT              # Pointer file referencing the active snapshot directory (authoritative)
```

## Manifest Schema

```json
{
  "index_id": "string",
  "graph_store_type": "string",
  "vector_store_type": "string",
  "corpus_hash": "sha256:...",
  "config_hash": "sha256:...",
  "created_at": "YYYY-MM-DDTHH:MM:SSZ",
  "schema_version": "1.0",
  "persist_format_version": "1.0",
  "complete": true,
  "versions": {
    "app": "x.y.z",
    "llama_index": "x.y.z",
    "qdrant_client": "x.y.z",
    "vector_client": "qdrant" | "lancedb" | null,
    "embed_model": "<model-id>"
  },
  "graph_exports": [
    {
      "path": "graph/graph_export-YYYYMMDDTHHMMSSZ.jsonl",
      "format": "jsonl" | "parquet",
      "seed_count": 32,
      "size_bytes": 1234,
      "duration_ms": 12.5,
      "sha256": "...",
      "created_at": "YYYY-MM-DDTHH:MM:SSZ"
    }
  ]
}
```

- `manifest.jsonl` contains one JSON object per payload file with `path`, `sha256`, `size_bytes`, and `content_type` fields.
- `manifest.meta.json` is written alongside the JSONL entries with `complete=false` while the workspace is still staged and flipped to `true` immediately after the atomic rename succeeds.
- `manifest.checksum` stores `{schema_version, created_at, manifest_sha256}` where the digest covers the sorted line hashes plus the full metadata payload.
- `graph_exports` enumerates GraphRAG export artifacts packaged with the snapshot, capturing telemetry metadata (`seed_count`, `duration_ms`, `size_bytes`, `sha256`).

## Hashing Rules

- corpus_hash: stable SHA‑256 over sorted list of `(relative_path, size, mtime_ns)` for ingested files
- config_hash: stable SHA‑256 over canonical JSON of retrieval/chunking/embedding settings and toggles

## Atomicity & Locking

- Write all artifacts under `storage/_tmp-<uuid>` and `fsync` + atomic rename to `storage/<timestamp>`.
- Acquire a single-writer lock via `storage/.lock` using `portalocker.Lock` (cross-platform). Persist sidecar metadata `{owner_id, created_at, last_heartbeat, ttl_seconds, takeover_count, schema_version}` and refresh heartbeats during long-running work. When the heartbeat exceeds `ttl_seconds + grace_seconds`, contenders SHALL rotate stale lock/meta files to `.stale-*` suffixes before acquiring the lock. If `portalocker` is unavailable, fall back to an `os.O_EXCL` sentinel with identical metadata semantics.
- Update `CURRENT.tmp` → `CURRENT` via atomic rename. Readers MUST resolve the pointer first and only fall back to lexicographic ordering when `CURRENT` is missing.
- On promotion failure, append a structured record to `errors.jsonl` (stage, snapshot_id, error_code, error message) to aid diagnostics.

## UI Integration

- Documents page: show snapshot path after ingest; provide rebuild action and export summary from `graph_exports`.
- Chat page: read latest manifest metadata, compute hashes, show staleness badge when mismatched; offer rebuild

## Acceptance Criteria

```gherkin
Feature: Atomic snapshots with manifest
  Scenario: Persist a new snapshot
    Given a vector index and a property graph store
    When I create a snapshot
    Then storage/<timestamp> SHALL exist with vector/, graph/, manifest.jsonl, manifest.meta.json, and manifest.checksum
    And the write SHALL occur under a lock and end with an atomic rename

  Scenario: Staleness detection
    Given a snapshot manifest with corpus_hash and config_hash
    And current corpus/config that differ
    When I open Chat
    Then a staleness badge SHALL be visible
```

## References

- ADR‑038 GraphRAG Persistence and Router Integration

Manifest Enrichment

- Include fields: `schema_version`, `persist_format_version`, and `versions` (keys: `app`, `llama_index`, `qdrant_client`, `vector_client`, `embed_model`).
- Write `manifest.jsonl`, `manifest.meta.json`, and `manifest.checksum` together; `manifest.json` SHALL NOT be emitted going forward.
- Record GraphRAG export metadata under `graph_exports` (path, format, seed_count, size_bytes, duration_ms, sha256, created_at). Timestamped export files SHALL follow `graph_export-YYYYMMDDTHHMMSSZ.<ext>` naming and be written *before* manifest hashing.
- Set `complete=true` only after the workspace renames into its immutable directory; prior to rename the field SHALL remain `false` and consumers MUST ignore incomplete manifests.
- Persist an optional `errors.jsonl` capturing structured failure events (stage, snapshot_id, error_code, message) for telemetry and troubleshooting.

Relpath Hashing

- Compute `corpus_hash` using POSIX relpaths relative to `uploads/` for OS-agnostic stability.
- Compute `config_hash` from a canonical JSON structure with sorted keys, normalized booleans, and sanitized paths. The same helper MUST be reused by SnapshotManager, UI staleness checks, and tests to guarantee determinism.
- ADR‑031 Local‑first Persistence Architecture
- ADR‑034 Idempotent Indexing and Embedding Reuse

Lock Metadata & Heartbeats

- Single-writer locks SHALL be implemented via a `SnapshotLock` backed by `portalocker` when available, with a fallback to `os.O_EXCL` semantics.
- Each lock acquisition writes a JSON sidecar `<lock>.meta.json` containing:
  - `owner_id`: `<pid>@<hostname>` identifying the holder.
  - `created_at` / `last_heartbeat`: UTC timestamps (ISO8601) updated on every refresh.
  - `ttl_seconds`: client-configured lease duration.
  - `takeover_count`: cumulative number of stale lock evictions.
  - `schema_version`: metadata schema revision (currently `1`).
- Holders MUST refresh (`last_heartbeat`) at least every `ttl_seconds / 2`. Contenders SHALL treat a lock as stale when `now - last_heartbeat > ttl_seconds + grace_seconds` and rotate both sentinel and metadata files to `.stale-<timestamp>-<owner>-<counter>` before attempting takeover.
- All metadata writes use write-to-temp + `os.replace` + `fsync` to guarantee durability on POSIX systems. Parent directories SHOULD be fsynced where supported; on Windows best-effort semantics apply.
- Stale lock rotation MUST occur prior to every acquisition attempt to guarantee single-writer safety after crashes.

Promotion & CURRENT Pointer Discipline

- `_tmp-` workspaces and final snapshot directories MUST reside on the same filesystem (validated via `stat().st_dev`) to preserve `os.replace` atomicity.
- Promotion sequence:
  1. Write manifest artifacts with `complete=false` and `fsync` each file.
  2. `os.replace` `_tmp-<uuid>` with `<timestamp>`; `fsync` the parent directory.
  3. Write `CURRENT.tmp` containing the promoted directory name; `fsync` file then parent.
  4. `os.replace` `CURRENT.tmp` -> `CURRENT`.
- Readers MUST resolve `CURRENT` and only fall back to lexicographic snapshot lookup when the pointer is absent or corrupted.

Retention & Cleanup

- Retain the latest `settings.snapshot.retention_count` finalized directories, never deleting the directory referenced by `CURRENT`.
- `_tmp-*` workspaces older than `settings.snapshot.gc_grace_seconds` SHALL be removed during initialization. Stale lock artifacts (`*.stale-*`) MAY be pruned opportunistically.
- SnapshotManager MUST run retention under the same lock to avoid concurrent deletions during promotion.

Telemetry & Observability

- Snapshot operations emit OpenTelemetry spans (`snapshot.begin`, `snapshot.persist`, `snapshot.promote`, `snapshot.retention`) annotated with `snapshot_id`, `corpus_hash`, `config_hash`, and export durations.
- Graph export operations (manual and snapshot-triggered) emit `export_performed` events with the telemetry payload recorded in `manifest.graph_exports`.
- Lock evictions emit structured log events at `warning` level including previous owner identifier and elapsed TTL.

## Changelog

- 1.1.0 (2025-09-16): Documented portalocker-based locking, graph export metadata, `errors.jsonl`, and removal of legacy `manifest.json`.
- 1.0.0 (2025-09-09): Initial draft spec

## UI Staleness Badge

- The Chat page MUST show a visible staleness badge when the current manifest differs from the latest content/config hashes.
- Tooltip copy MUST be exactly:
  - “Snapshot is stale (content/config changed). Rebuild in Documents → Rebuild GraphRAG Snapshot.”
- The badge MUST be computed from local state (manifest vs current hashes) and MUST NOT trigger any network calls.

## Lock Semantics (Single‑Writer)

- SnapshotManager MUST implement a single‑writer lock backed by `portalocker` with TTL metadata, heartbeat refresh, and stale eviction (fallback to an `os.O_EXCL` sentinel when `portalocker` is unavailable). Timeout errors MUST surface clear user messaging.
- While rebuilding, UI controls in Documents → Snapshot MUST reflect a locked state and prevent concurrent rebuilds.

## Manifest Fields

- Manifest MUST include at minimum: `corpus_hash`, `config_hash`, `created_at`, `version` (semantic), and optional `notes`.
- Hashes MUST be deterministic and computed from stable inputs (e.g., file list + normalized config serialization).

## Atomic Rename

- Writes MUST occur to a temporary directory name with `_tmp-<uuid>` suffix and be atomically renamed to a timestamped final directory on success.
- On failure, incomplete `_tmp-*` directories MUST be cleaned up on next run.

## Acceptance & UX Mapping

- Documents page MUST provide:
  - Rebuild button (lock‑aware) and current snapshot path display.
  - Exports section (JSONL baseline; Parquet optional with `pyarrow`) and seed cap display. Export artifacts MUST be timestamped using `graph_export-YYYYMMDDTHHMMSSZ.*` naming, stored inside each snapshot under `graph/`, and emit telemetry events recording destination paths, seed counts, file sizes, and durations.
- Chat page MUST show the staleness badge and tooltip per UI Staleness Badge above.
