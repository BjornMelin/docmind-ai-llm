---
ADR: 054
Title: Configuration Surface Pruning (Remove Unused/No-op Knobs for v1)
Status: Superseded
Version: 1.2
Date: 2026-01-09
Supersedes:
Superseded-by: ADR-050
Related: 021, 023, 024, 013, 031, 035, 043
Tags: configuration, maintainability, release
---

## Description

Remove configuration knobs that exist in `src/config/settings.py` but have no implementation in `src/`, and update docs to avoid advertising non-existent functionality in the v1 release.

## Context

The settings model currently includes several fields that are either unreferenced in `src/` **or** have no functional implementation (only placeholder/plumbing code):

- `backup_enabled`, `backup_keep_last`
- `semantic_cache.*` (ADR‑035 references GPTCache/FAISS, but deps are not present)
- `analysis.*` (ADR‑023 analysis mode strategy not implemented)
- `AgentConfig.enable_deadline_propagation`, `AgentConfig.enable_router_injection`
- `ChatConfig.sqlite_path` (ADR‑021 prescribed SQLite chat store; v1 uses JSON `SimpleChatStore`, see ADR‑043)
- `DatabaseConfig.sqlite_db_path`, `DatabaseConfig.enable_wal_mode` (no SQLite operational store is implemented; `sqlite3` is unused in `src/`)

Shipping these as-is creates a misleading public configuration surface and increases maintenance and support burden (“why doesn’t this setting do anything?”).

> **Supersession Notice (2026-01-09):** This ADR is superseded by ADR-050 (Config Discipline & Env Bridges). The original "prune for v1" approach was reconsidered; the project direction is now to implement the existing configuration surface rather than removing fields. Advanced capabilities (analysis modes, backups, semantic cache) will be completed per ADR-050 and SPEC-031.

## Decision Drivers

- Coherent v1: config implies behavior
- Reduce drift between ADRs/specs/config reference and shipped code
- Avoid introducing large new features/dependencies late in v1 (semantic cache, analysis modes, deadline propagation)
- Preserve backward compatibility as much as feasible (unknown env vars ignored)

## Alternatives

- A: Remove unused settings fields and update docs/ADRs for v1 scope (Selected)
- B: Implement all missing features so knobs become real (high scope; new deps; risky)
- C: Keep knobs but mark as no-op/unsupported (misleading; increases cognitive load)

### Decision Framework (≥9.0)

Weights: Complexity 40% · Risk Reduction 30% · Alignment 30% (10 = best)

| Option                         | Complexity (40%) | Risk (30%) | Alignment (30%) |   Total |
| ------------------------------ | ---------------: | ---------: | --------------: | ------: |
| **A: Remove + document scope** |              9.5 |        9.5 |             9.5 | **9.5** |
| B: Implement features          |              3.0 |        6.0 |             5.0 |     4.5 |
| C: Keep no-op                  |              8.5 |        5.5 |             6.0 |     7.0 |

## Decision

> **Note:** This decision was not implemented; the ADR is superseded. The original plan was to:

1. Remove the unused fields from `DocMindSettings` and nested models.

2. Update documentation to reflect v1 scope:

   - configuration reference
   - ADR index / affected ADRs (mark as superseded/out-of-scope for v1)

3. Preserve backward compatibility:

   - `DocMindSettings` already sets `extra=\"ignore\"`, so unknown env vars do not break startup.
   - Add a single startup warning (optional) if deprecated env vars are detected, to avoid silent confusion.

## Security & Privacy

- Reduces risk of users enabling a knob that they believe changes privacy/security behavior when it does not.

## Testing

- Unit test that settings load even when deprecated env vars are set (they should be ignored).
- Doc drift check (WP08) should no longer reference removed config fields as v1 behavior.

## Consequences

### Positive Outcomes

- Smaller, truthful configuration surface for v1.
- Reduced cognitive load and doc drift risk.

### Trade-offs

- Future reintroduction of these features will require deliberate ADR/SPEC work and adding dependencies where needed.

## Changelog

- 1.2 (2026-01-10): Superseded by ADR-050; updated metadata and Decision framing.
- 1.1 (2026-01-09): Added supersession notice (same-day update).
- 1.0 (2026-01-09): Proposed for v1 coherence and maintainability (initial draft).
