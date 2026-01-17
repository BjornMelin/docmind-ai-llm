---
spec: SPEC-039
title: Operational Metadata Store (SQLite WAL)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - FR-015: Persist operational metadata via SQLite WAL.
  - FR-025: Background jobs record progress/cancellation safely.
  - NFR-REL-001: Recovery after restart without corruption.
  - NFR-SEC-001: Offline-first; local-only by default.
related_adrs: ["ADR-055", "ADR-052", "ADR-031", "ADR-033", "ADR-032", "ADR-024"]
notes: "ADR-055 and ADR-052 are currently 'Proposed' status (verified 2026-01-17); update references when they advance to 'Accepted'."
---

## Goals

1. Provide a **transactional, local-only ops metadata store** for DocMind.
   - This ops DB is separate from the LangGraph SQLite checkpointer + `SqliteStore` (ADR-057): separate file, schema, and connections; they may coexist under `settings.data_dir` but are operated independently.
2. Enable **background job UX** and reliable state recovery across restarts:
   - job lifecycle (queued/running/done/failed; add cancelled via later migration)
   - progress updates (percent, stage, timestamps)
3. Keep responsibilities separate:
   - ops metadata: SQLite WAL (this spec)
   - ingestion cache: LlamaIndex IngestionCache (DuckDBKV) (ADR-030)
   - analytics: DuckDB (ADR-032)
4. Keep the system offline-first: no new network surfaces.

## Implementation Status (2026-01-17)

Current codebase coverage:

- WAL setup exists for SQLite connections in `src/persistence/chat_db.py` and `src/persistence/memory_store.py`.
- Settings include `settings.database.sqlite_db_path` and `settings.database.enable_wal_mode`.
- Startup directory creation for the ops DB path exists in `src/config/integrations.py`.

Missing for this spec to be implemented:

- Ops DB schema and migrations (`jobs`, `snapshots`, `ui_state`) with `PRAGMA user_version` upgrades.
- Ops DB initializer that enables WAL/foreign keys/busy_timeout and applies migrations at startup.
- Ops DB API surface for job/snapshot/ui state reads/writes.
- Unit/integration tests for migrations and ops DB behavior.

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

Use a minimal schema that supports background jobs, snapshot history, and UI
state. This mirrors the v1 schema described in ADR-055.

#### `jobs`

- `id TEXT PRIMARY KEY`
- `status TEXT NOT NULL` (`queued|running|done|failed`)
- `payload TEXT` (nullable; metadata-only JSON, can include progress/stage)
- `created_at_ms INTEGER NOT NULL`
- `updated_at_ms INTEGER NOT NULL`

#### `snapshots`

- `id TEXT PRIMARY KEY`
- `job_id TEXT` (nullable; FK to `jobs.id`)
- `data TEXT` (nullable; metadata-only JSON)
- `created_at_ms INTEGER NOT NULL`

Foreign key:

- `FOREIGN KEY(job_id) REFERENCES jobs(id)`

#### `ui_state`

- `key TEXT PRIMARY KEY`
- `value TEXT` (nullable; metadata-only JSON)
- `updated_at_ms INTEGER NOT NULL`

### Migrations

Implement migrations via `PRAGMA user_version`:

- Ship SQL migration files under `src/persistence/migrations/ops_db/`
  - Use sequential numeric prefix: `0001_init.sql`, `0002_add_column.sql`, etc.
  - Each file must be idempotent within its version
- `ops_db.apply_migrations()` applies sequentially within a transaction
- **Rollback**: Not supported; migrations are forward-only (document schema changes in ADR if needed)

### Code Modules (Planned)

Add:

- `src/persistence/ops_db.py`
  - `connect_ops_db(path: Path) -> sqlite3.Connection`
  - `apply_migrations(conn) -> None`
  - `OpsDB` thin wrapper (single-record operations; batch not needed for v1):
    - `create_job(job_id: str, status: str, payload: dict | None) -> None`
    - `update_job(job_id: str, status: str, payload: dict | None = None) -> None`
    - `record_snapshot(snapshot_id: str, job_id: str | None, data: dict | None) -> None`
    - `upsert_ui_state(key: str, value: dict | None) -> None`
    - `get_ui_state(key: str) -> dict | None`

Integrate:

- `src/ui/background_jobs.py` writes job lifecycle state to ops DB (best-effort; fail-open: if ops DB write fails, log a warning and continue; do not abort the job).
- Snapshot actions record `snapshots` entries where relevant.

### Observability

Emit JSONL events (local-only) for:

- `ops_db_migrated` with `{from_version, to_version}`
- `job_state_changed` with `{job_id, status}`

Never include raw prompt/document text.

### Security

- Validate DB path (no traversal/symlink escape) and default under `settings.data_dir`.
  - Use `pathlib.Path.resolve().relative_to(settings.data_dir)` to ensure the path is within the allowed directory; raise `ValueError` if not.
  - Check for symlinks with `Path.is_symlink()` if stricter isolation is required.
- Store only metadata; no secrets; no raw content.
- Use bounded retries for `SQLITE_BUSY` and avoid unbounded loops.

## Testing Strategy (Planned)

### Unit

- `tests/unit/persistence/test_ops_db_migrations.py`
  - initializes empty DB, applies migrations, checks `user_version`
- `tests/unit/persistence/test_ops_db_jobs.py`
  - create/update job, verify status transitions and timestamps

### Integration

- `tests/integration/ui/test_background_job_ops_db.py`
  - start a background job and verify job lifecycle rows exist (temp data dir)

All tests must run offline and deterministic.

## Rollout / Migration

- Default: ops DB is created automatically at startup if missing.
- No user action required; existing installs get migrations applied.

## RTM Updates

Update `docs/specs/traceability.md`:

- Add/expand `FR-015` mapping to `src/persistence/ops_db.py` + tests above once implemented.
