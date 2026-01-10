---
spec: SPEC-037
title: Local Backup & Retention (Snapshots, Cache, Analytics, Qdrant)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - FR-027: Users can create local backups with retention/restore steps.
  - NFR-SEC-001: Offline-first; no remote endpoints by default.
  - NFR-MAINT-003: No placeholder APIs; docs/specs/RTM match code.
related_adrs: ["ADR-033", "ADR-031", "ADR-030", "ADR-024", "ADR-038"]
---

## Goals

1. Provide a **manual**, local-only backup mechanism with rotation.
2. Back up the artifacts required to restore a working DocMind environment:
   - ingestion cache DB (DuckDB)
   - snapshots and manifests (`data/storage/`)
   - uploads (optional)
   - analytics DB (optional)
   - Qdrant collection snapshot (recommended)
3. Provide safe, documented restore steps.

## Non-goals

- Background schedulers/daemons for backups.
- Remote backup destinations (S3, GDrive) in the default posture.

## Design

### Primary Interface

- CLI script: `scripts/backup.py`
  - `--dest <path>` (optional; default under `data/backups/`)
  - `--include-uploads`
  - `--include-analytics`
  - `--include-logs`
  - `--keep-last <n>` (default `settings.backup_keep_last`)
  - `--qdrant-snapshot` (default true if Qdrant reachable)

Optional UI entry point (stretch, but recommended for a polished release):

- Settings page button: “Create backup now”

### Artifact Set

Minimum required:

- `settings.cache_dir/<filename>` (DuckDB cache)
- `settings.data_dir/storage/` (DocMind snapshot directories + CURRENT pointer)
- `.env` (optional but recommended)

Recommended:

- Qdrant collection snapshot (server-side)
- `settings.data_dir/analytics/analytics.duckdb` (if enabled)
- `settings.data_dir/uploads/` (if users want backups of raw files)

### Qdrant Snapshot

Use Qdrant’s snapshots API via `qdrant-client`:

1. `create_snapshot(collection_name=...)` to generate a snapshot.
2. Download the snapshot file via the Qdrant REST endpoint:
   - `GET /collections/{collection}/snapshots/{snapshot_name}`

Store the snapshot file under the backup directory with a manifest file describing:

- qdrant URL
- collection name
- snapshot file name
- timestamp

### Rotation

- Backups are stored under: `data/backups/backup_<YYYYMMDD_HHMMSS>/`
- Prune older backups beyond `keep_last` after a successful backup completes.
- Never delete a backup directory if any step failed.

### Observability

Emit:

- JSONL event `backup_created` with `{path, included: [...], bytes_written, duration_ms}`
- JSONL event `backup_pruned` with `{deleted_count, keep_last}`

### Security

- Validate destination paths (no traversal, no symlink escapes).
- Never log secrets from `.env` or API keys.

## Testing Strategy

### Unit

- Rotation logic (create N+2 dirs, prune to N).
- Manifest writing and path validation.

### Integration

- Run backup against a temp data_dir + temp cache_dir (fake files), with Qdrant snapshot step mocked.

## Rollout / Migration

- No data migrations.
- Ship CLI first; UI button is additive.

## RTM Updates

Update `docs/specs/traceability.md`:

- Add row `FR-027` with code + tests once implemented.
