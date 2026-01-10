---
spec: SPEC-035
title: Configuration Surface Pruning (Remove Unused/No-op Knobs)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Superseded
related_requirements:
  - NFR-MAINT-003: No drift between docs/specs/RTM and shipped code
related_adrs: ["ADR-054", "ADR-024"]
---

> **Supersession Notice (2026-01-09):** This spec is superseded by ADR-050 (Config Discipline & Env Bridges) and SPEC-031. The current direction is to implement the existing configuration surface (analysis modes, backups, semantic cache, ops DB) rather than pruning it.

## Goals

1. Remove unused settings knobs that have no runtime implementation.
2. Update docs/config references so v1 does not advertise non-existent behavior.
3. Ensure backward compatibility: unknown env vars are ignored and do not break startup.

## Non-goals

- Implement backup/semantic cache/analysis/deadline propagation for v1.
- Rename or add new env vars (handled by config discipline work).

## Scope

Remove from `src/config/settings.py` (and any docs/UI references):

- `backup_enabled`, `backup_keep_last`
- `SemanticCacheConfig` and `DocMindSettings.semantic_cache`
- `AnalysisConfig` and `DocMindSettings.analysis`
- `AgentConfig.enable_deadline_propagation`
- `AgentConfig.enable_router_injection`
- `ChatConfig.sqlite_path`
- `DatabaseConfig.sqlite_db_path`
- `DatabaseConfig.enable_wal_mode`

## Design

1. Delete the unused fields/models and update any exports in `__all__`.
2. Update docs:
   - `docs/developers/configuration-reference.md` (remove sections)
   - ADR index or affected ADRs (mark out-of-scope for v1)
3. Compatibility:
   - rely on `extra="ignore"` in Pydantic settings
   - optionally log a one-time warning if deprecated env var prefixes are present.

## Testing Strategy

- Unit test: set deprecated env vars and assert `DocMindSettings()` loads and ignores them.
- Docs drift check (WP08) should pass after removing doc references.

## Rollout / Migration

- No data migrations.
- Rollback by restoring the removed settings fields and docs references.

## RTM Updates

Update `docs/specs/traceability.md` row `NFR-MAINT-003` to include this work package’s code and tests.
