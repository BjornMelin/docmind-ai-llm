---
ADR: 064
Title: Python 3.13-Only Baseline (Primary 3.13.11)
Status: Accepted
Version: 1.0
Date: 2026-01-17
Supersedes: ADR-062
Superseded-by:
Related: ADR-042, ADR-062, ADR-063
Tags: python, packaging, tooling, compatibility
References:
  - https://www.python.org/downloads/release/python-31311/
  - https://devguide.python.org/versions/
---

## Description

Standardize the repository on **Python 3.13 only** (primary dev/runtime: **CPython 3.13.11**) and target **py313** in Ruff/Pyright.

## Context

The project is pre-production with zero active users. Maintaining compatibility with multiple Python minor versions adds:

- Resolver and wheel-matrix churn for compiled dependencies.
- A larger CI matrix and slower feedback loops.
- Tooling ambiguity (which version should lint/type-check target).

The repository already pins `.python-version` to **3.13.11** and uses a Python 3.13-based Docker image. Moving to a Python 3.13-only baseline makes the upgrade coherent across:

- packaging (`requires-python`)
- tooling (Ruff/Pyright targeting)
- CI (single-version gate)
- documentation and user setup paths

## Decision Drivers

- **Maintainability**: reduce version-matrix complexity across dependencies, tooling, and CI.
- **DX**: a single baseline minimizes “it works on X but not Y” issues.
- **Ecosystem alignment**: Python 3.13 is the primary target for new features and performance work.
- **Operational safety**: fewer supported combinations reduces surprises in release hardening.

## Alternatives

- **A: Python 3.13-only baseline** (Selected)
- B: Python 3.13 primary, but support 3.11–3.13 (multi-version runtime support)
- C: Keep Python 3.11 as baseline

### Decision Framework (≥9.0)

Weights: Complexity 40% · Performance/Scale 30% · Ecosystem Alignment 30% (10 = best)

| Option | Complexity (40%) | Perf/Scale (30%) | Alignment (30%) | Total | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| **A: 3.13-only baseline** | 9.5 | 9.0 | 9.5 | **9.3** | ✅ Selected |
| B: 3.13 primary, support 3.11–3.13 | 8.5 | 9.0 | 8.5 | 8.7 | Rejected |
| C: 3.11 baseline | 8.0 | 7.5 | 7.0 | 7.6 | Rejected |

## Decision

1) Set `requires-python = ">=3.13,<3.14"`.

2) Set Ruff `target-version = "py313"` and Pyright `pythonVersion = "3.13"`.

3) Run CI on Python `3.13.11` only.

## Consequences

### Positive outcomes

- Reduced resolver churn and faster CI.
- Tooling configuration matches the runtime baseline (less confusion).
- Clears the way to use Python 3.13 typing and stdlib features without compatibility scaffolding.

### Trade-offs

- Users on Python 3.11/3.12 must upgrade to run DocMind.
