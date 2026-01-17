---
spec: SPEC-039
title: Operational Metadata Store (SQLite WAL)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-015: Persist operational metadata via SQLite WAL.
  - FR-025: Background jobs record progress/cancellation safely.
  - NFR-REL-001: Recovery after restart without corruption.
  - NFR-SEC-001: Offline-first; local-only by default.
related_adrs: ["ADR-031", "ADR-033", "ADR-032", "ADR-024"]
notes: "ADR-055 and ADR-052 are currently 'Proposed' status (verified 2026-01-10); update references when they advance to 'Accepted'."
---

## Goals

1. Provide a **transactional, local-only ops metadata store** for DocMind.
   - This ops DB is separate from the LangGraph SQLite checkpointer + `SqliteStore` (ADR-057): separate file, schema, and connections; they may coexist under `settings.data_dir` but are operated independently.
2. Enable **background job UX** and reliable state recovery across restarts:
   - job lifecycle (queued/running/succeeded/failed/cancelled)
   - progress updates (percent, stage, timestamps)
3. Keep responsibilities separate:
   - ops metadata: SQLite WAL (this spec)
   - ingestion cache: LlamaIndex IngestionCache (DuckDBKV) (ADR-030)
   - analytics: DuckDB (ADR-032)
4. Keep the system offline-first: no new network surfaces.

## Non-goals

- Replace analytics DB (DuckDB) with SQLite.
- Store raw prompts or raw document text in the ops DB.
- Multi-user / remote database access.

## Technical Design

### Storage Location

- DB file: `settings.database.sqlite_db_path` (default `./data/docmind.db`)
- Must remain under `settings.data_dir` by default.
- Ops DB is independent from the Chat DB used by LangGraph persistence; backup/restore and WAL handling are separate per DB file.

### Database Settings (WAL)

On every connection:

- `PRAGMA journal_mode=WAL;`
- `PRAGMA foreign_keys=ON;`
- `PRAGMA busy_timeout=5000;` (5-second timeout for single-writer contention; adjust per deployment)
- `PRAGMA synchronous=NORMAL;` (default for balance; use `FULL` if durability is critical and latency is acceptable)

### Schema (v1 minimal)

Use a minimal schema that supports background jobs and snapshot history.

#### `ops_job_run`

- `job_id TEXT PRIMARY KEY` (UUID)
- `job_type TEXT NOT NULL` (e.g., `ingestion`, `snapshot_rebuild`, `backup`)
- `status TEXT NOT NULL` (`queued|running|succeeded|failed|cancelled`)
- `created_at_ms INTEGER NOT NULL`
- `started_at_ms INTEGER` (nullable)
- `updated_at_ms INTEGER NOT NULL`
- `finished_at_ms INTEGER` (nullable)
- `progress_pct REAL` (0..100; nullable)
- `stage TEXT` (nullable, short label)
- `message TEXT` (nullable; must be metadata-only, no raw doc content)
- `error_code TEXT` (nullable)

Indexes:

- `(job_type, status)`
- `(updated_at_ms)`

#### `ops_snapshot_event`

- `id INTEGER PRIMARY KEY`
- `snapshot_id TEXT NOT NULL` (directory name / timestamp)
- `event_type TEXT NOT NULL` (`created|gc_pruned|restored|failed`)
- `created_at_ms INTEGER NOT NULL`
- `meta_json TEXT` (nullable; JSON string, metadata-only)

Index:

- `(snapshot_id, created_at_ms)`

### Migrations

Implement migrations via `PRAGMA user_version`:

- Ship SQL migration files under `src/persistence/migrations/ops_db/`
  - Use sequential numeric prefix: `0001_init.sql`, `0002_add_column.sql`, etc.
  - Each file must be idempotent within its version
- `ops_db.apply_migrations()` applies sequentially within a transaction
- **Rollback**: Not supported; migrations are forward-only (document schema changes in ADR if needed)

### Code Modules

Add:

- `src/persistence/ops_db.py`
  - `connect_ops_db(path: Path) -> sqlite3.Connection`
  - `apply_migrations(conn) -> None`
  - `OpsDB` thin wrapper (single-record operations; batch not needed for v1):
    - `create_job(job_type: str) -> str` — creates and returns job_id
    - `update_job(job_id: str, progress_pct: float | None, stage: str | None, message: str | None) -> None`
    - `finish_job(job_id: str, status: str, error_code: str | None = None) -> None`
    - `list_jobs(limit: int = 50) -> list[dict]` — returns recent jobs ordered by `updated_at_ms` DESC
    - `record_snapshot_event(snapshot_id: str, event_type: str, meta: dict | None) -> None`

Integrate:

- `src/ui/background_jobs.py` writes job lifecycle state to ops DB (best-effort; fail-open: if ops DB write fails, log a warning and continue; do not abort the job).
- Snapshot actions record `ops_snapshot_event` entries where relevant.

### Observability

Emit JSONL events (local-only) for:

- `ops_db_migrated` with `{from_version, to_version}`
- `job_state_changed` with `{job_id, job_type, status, progress_pct}`

Never include raw prompt/document text.

### Security

- Validate DB path (no traversal/symlink escape) and default under `settings.data_dir`.
  - Use `pathlib.Path.resolve().relative_to(settings.data_dir)` to ensure the path is within the allowed directory; raise `ValueError` if not.
  - Check for symlinks with `Path.is_symlink()` if stricter isolation is required.
- Store only metadata; no secrets; no raw content.
- Use bounded retries for `SQLITE_BUSY` and avoid unbounded loops.

## Testing Strategy

### Unit

- `tests/unit/persistence/test_ops_db_migrations.py`
  - initializes empty DB, applies migrations, checks `user_version`
- `tests/unit/persistence/test_ops_db_jobs.py`
  - create/update/finish job, list jobs ordering, status transitions

### Integration

- `tests/integration/ui/test_background_job_ops_db.py`
  - start a background job and verify job lifecycle rows exist (temp data dir)

All tests must run offline and deterministic.

## Rollout / Migration

- Default: ops DB is created automatically at startup if missing.
- No user action required; existing installs get migrations applied.

## RTM Updates

Update `docs/specs/traceability.md`:

- Add/expand `FR-015` mapping to `src/persistence/ops_db.py` + tests above.
