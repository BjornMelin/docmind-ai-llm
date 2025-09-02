# ADR-033: Local Backup & Retention

## Metadata

**Status:** Proposed  
**Version/Date:** v1.0 / 2025-09-02

## Title

Manual Local Backups with Simple Rotation for Core Artifacts

## Description

Provide a simple, manual backup mechanism for key local artifacts with a small rotation policy. No background daemons or external services. Backups are created on demand and include the cache DB, Qdrant local data, and (optionally) the analytics DB and documents directory.

## Context

- Archived ADR-007 proposed an automated BackupManager with scheduled backups and retention.  
- ADR-030/ADR-031 aim for minimalism and separation of concerns; automated services add complexity.  
- Users still benefit from a documented, local backup path that’s easy to run and restore.

## Decision Drivers

- Local-first, offline operation.
- Keep it simple: single command/script, no daemons.
- Predictable restore steps; safe file copy semantics.

## Alternatives

- Automated background tasks (rejected: complexity and surprise cost in local-first app).  
- External backup services (rejected by default: violate local-only principle).  
- No backup (acceptable for defaults; this ADR offers an opt-in path).

## Decision

Implement a CLI/script that creates a timestamped backup directory with copies of selected artifacts and prunes older backups beyond a configured limit.

## Design

### Artifacts

- Cache DB: `settings.cache_dir / "docmind.duckdb"`
- Qdrant local data directory
- Optional analytics DB: `settings.data_dir / "analytics" / "analytics.duckdb"`
- Optional documents directory (if configured)

### Destination

- `settings.data_dir / "backups" / backup_{YYYYMMDD_HHMMSS}`

### Rotation

- Keep last N backups (default: 7).  
- Remove oldest beyond the limit.  
- Never delete the currently creating backup on failure.

### Safety

- Ensure handles are closed before copy.  
- Best-effort behavior; partial backups are identifiable.  
- Documented restore procedure.

## Configuration

- `backup.enabled`: bool (default: False)  
- `backup.keep_last`: int (default: 7)

## Testing

- **Unit**: rotation logic (dry-run path); skip if destinations missing.  
- **Doc-tests**: Restore instructions verified in documentation.

## Dependencies

- **stdlib only**: `pathlib`, `shutil`, `datetime`

## Related Decisions

- **ADR-031** (Persistence Architecture): Defines which stores exist and remain separate.  
- **ADR-030** (Cache Unification): Identifies the cache DB to include in backup.  
- **ADR-032** (Local Analytics): Include analytics DB when enabled.

## Consequences

- **Positive**: Clear, predictable local backups; simple restore steps.  
- **Trade-offs**: Manual burden; not continuous protection.

## Changelog

- **v1.0 (2025-09-02)**: Initial proposal for manual backups with simple rotation.
