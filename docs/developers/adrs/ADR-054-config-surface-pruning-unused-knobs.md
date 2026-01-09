---
ADR: 054
Title: Configuration Surface Pruning (Remove Unused/No-op Knobs for v1)
Status: Proposed
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 024, 013
Tags: configuration, maintainability, release
---

## Description

Remove configuration knobs that exist in `src/config/settings.py` but have no implementation in `src/`, and update docs to avoid advertising non-existent functionality in the v1 release.

## Context

The settings model currently includes several fields that are not referenced anywhere in `src/`:

- `backup_enabled`, `backup_keep_last`
- `semantic_cache.*` (ADR‑035 references GPTCache/FAISS, but deps are not present)
- `analysis.*` (ADR‑023 analysis mode strategy not implemented)
- `AgentConfig.enable_deadline_propagation`, `AgentConfig.enable_router_injection`

Shipping these as-is creates a misleading public configuration surface and increases maintenance and support burden (“why doesn’t this setting do anything?”).

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

| Option | Complexity (40%) | Risk (30%) | Alignment (30%) | Total |
|---|---:|---:|---:|---:|
| **A: Remove + document scope** | 9.5 | 9.5 | 9.5 | **9.5** |
| B: Implement features | 3.0 | 6.0 | 5.0 | 4.5 |
| C: Keep no-op | 8.5 | 5.5 | 6.0 | 7.0 |

## Decision

We will:

1) Remove the unused fields from `DocMindSettings` and nested models.

2) Update documentation to reflect v1 scope:

- configuration reference
- ADR index / affected ADRs (mark as superseded/out-of-scope for v1)

3) Preserve backward compatibility:

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

- 1.0 (2026-01-09): Proposed for v1 coherence and maintainability.

