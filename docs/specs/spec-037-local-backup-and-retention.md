---
spec: SPEC-037
title: Local Backup & Retention (Snapshots, Cache, Analytics, Qdrant)
version: 1.1.0
date: 2026-07-14
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-027: Users can create local backups with retention/restore steps.
  - NFR-SEC-001: Offline-first; no remote endpoints by default.
  - NFR-MAINT-003: No placeholder APIs; docs/specs/RTM match code.
related_adrs: ["ADR-033", "ADR-031", "ADR-030", "ADR-024", "ADR-038"]
---

## Goals

1. Provide a manual, local-only backup mechanism with rotation.
2. Back up the artifacts required to restore a working DocMind environment:
   - ingestion cache DB (DuckDB)
   - snapshots and manifests (`data/storage/`)
   - existing content-addressed page-image and thumbnail artifacts
   - configured chat/checkpoint database (`settings.chat.sqlite_path`)
   - authoritative uploads and stable deployment identity
   - analytics DB (optional)
   - exact active Qdrant collection snapshots
3. Provide safe, documented restore steps.

## Non-goals

- Background schedulers/daemons for backups.
- Remote backup destinations (S3, GDrive) in the default posture.
- Distributed Qdrant capture; its supported backup procedure requires one
  snapshot per node.

## Design

### Primary interface

- CLI script: `scripts/backup.py`
  - `--dest <path>` (optional; default under `data/backups/`)
  - `--include-analytics`
  - `--include-logs`
  - `--include-env`
  - `--keep-last <n>` (default `settings.backup_keep_last`)
  - `--diagnostic-no-uploads` (creates an incomplete diagnostic capture)
  - `--no-qdrant-snapshot` (creates an incomplete diagnostic capture)
  - `--json` (prints the full result summary)

### Artifact set

The default command creates a complete recovery point. It requires:

- `settings.cache.ingestion_db_path` (DuckDB cache)
- the one verified snapshot referenced by `settings.data_dir/storage/CURRENT`,
  copied with a matching `CURRENT` pointer
- `settings.chat.sqlite_path` (SQLite online backup, including committed WAL state)
- `settings.data_dir/uploads/`, with a content inventory that matches the active
  snapshot's corpus hash
- `settings.data_dir/.deployment-id`, matching the active collection ownership
- exact Qdrant snapshots for both collections in the active manifest
- `settings.artifacts.dir` or `settings.data_dir/artifacts/` when the artifact
  store exists

The following artifacts are optional:

- `settings.data_dir/analytics/analytics.duckdb` (if enabled)
- `logs/`
- `.env` only with explicit `--include-env` acknowledgement because it contains secrets

The backup service uses each database engine's native copy boundary. SQLite's
online backup API captures the configured chat database, and DuckDB's
`COPY FROM DATABASE` captures the ingestion cache and optional analytics
database. Both include committed WAL state and produce transactionally
consistent destinations. DuckDB permits online copying only inside its owning
process; stop a separately running DocMind process before invoking the backup
CLI. Copying only a main database file is not a supported backup path.

The backup service acquires the snapshot writer lock while it resolves and
copies `CURRENT`. It records the selected snapshot ID and exact physical text
and image collection names in `manifest.json.activation`. Qdrant capture targets
only those two immutable collections, even if a newer generation activates after
the filesystem copy finishes.

### Manifest contract

`manifest.json` is the recovery contract:

| Field | Contract |
| --- | --- |
| `created_at`, `app_version` | Source time and application version |
| `complete` | `true` only when `warnings` is empty and all required recovery data verifies |
| `included`, `bytes_written` | Captured artifact labels and total bytes |
| `activation` | Active snapshot name, exact text/image collection names, and deployment ID |
| `uploads.files` | Upload-relative path, byte count, and SHA-256 for every copied source file |
| `artifacts.files` | Artifact-relative path, byte count, and SHA-256 for every copied artifact |
| `databases` | Exact backup-relative path, byte count, and SHA-256 for the required cache and chat databases |
| `qdrant` | Safe source URL, server version, and verified collection snapshot records |
| `warnings` | Recoverability failures that force an incomplete diagnostic backup |
| `maintenance_warnings` | Cleanup debt that does not invalidate captured recovery data |

Each `qdrant.collections` record contains `collection`, `snapshot_name`,
`filename`, `size_bytes`, `checksum`, and exact `point_count`. The captured
collection set must equal `activation.collections`. The copied
`data/.deployment-id` must equal `activation.deployment_id`.

Retention revalidates the manifest, required database files, copied tree
inventories, active snapshot, upload corpus hash, deployment identity, and
Qdrant files. A self-consistent manifest cannot make altered recovery data
eligible for pruning.

### Capture Qdrant snapshots

Use Qdrant’s snapshots API via `qdrant-client`:

1. Call `create_snapshot(collection_name=physical_collection_name)`.
2. Download the snapshot file via the Qdrant REST endpoint:
   - `GET /collections/{collection}/snapshots/{snapshot_name}`

Before capture, require a single-node topology and read an exact collection
point count. Store the snapshot file under the backup directory only after its
byte count and optional server-advertised SHA-256 validate. The manifest records:

- qdrant URL
- server version
- collection name
- snapshot file name
- byte count and SHA-256
- exact source point count

### Rotation

- Complete backups use `data/backups/backup_<YYYYMMDD_HHMMSS>_<id>/`.
- Incomplete diagnostic runs use `incomplete-backup_<timestamp>_<id>` and are excluded
  from every later/manual prune.
- Prune older backups beyond `keep_last` after a successful backup completes.
- Set `manifest.json.complete=false` and never prune an older backup if any
  required or explicitly requested artifact produced a warning.
- Keep `complete=true` when only `maintenance_warnings` exist. These warnings
  describe cleanup debt after the recovery data was verified.
- Serialize create/prune with an OS file lock and remove every unpromoted
  workspace after an exception.

### Observability

Emit:

- JSONL event `backup_created` with completion, retention, and maintenance
  warning counts plus path, included labels, bytes, and duration
- JSONL event `backup_pruned` with `{deleted_count, keep_last}`

### Security

- Validate destination paths (no traversal, no symlink escapes, and no
  destination nested inside a copied source tree).
- Apply the endpoint allowlist before constructing a Qdrant client.
- Never log secrets from `.env` or API keys.

## Testing strategy

### Unit

- Rotation logic (create N+2 dirs, prune to N).
- Manifest completeness, wrong artifact types, path containment, same-second
  creation, failed-workspace cleanup, and symlink-safe pruning.

### Integration

- Exercise real SQLite and DuckDB databases with committed data still present in
  their WALs, custom quoted paths, and simultaneous cache and analytics copies.
- Mock only the Qdrant transport boundary; verify the downloaded snapshot and
  server version are recorded in the manifest. Reject truncated bodies and
  verify the server-advertised SHA-256 when Qdrant supplies one.
- Follow the operations guide's restore procedure in release smoke testing and
  verify identity, uploads, chat, cache, analytics, storage, artifacts, and
  staged Qdrant counts.

## Rollout / Migration

- No data migrations.
- Ship CLI first; UI button is additive.
- Restore remains a documented, deliberate operator procedure because it
  replaces live state and requires writer quiescence.

## RTM updates

`FR-027` is tracked in `docs/specs/traceability.md` with code + test coverage.
