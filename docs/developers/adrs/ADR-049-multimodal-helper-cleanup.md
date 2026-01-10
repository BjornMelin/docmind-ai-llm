---
ADR: 049
Title: Multimodal Helper Cleanup — Remove Dead Test-only Utility From Production Package
Status: Proposed
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 037, 002
Tags: multimodal, cleanup, tests, maintainability
---

## Description

Remove the unused `src/utils/multimodal.py` helper (and its TODO placeholder) from the production package and keep any validation-only helpers in tests/eval code paths.

## Context

`src/utils/multimodal.py` is currently referenced only by unit tests and contains a TODO for “phase 2” graph traversal/LLM summarization.

Shipping unused “toy pipeline” utilities in the production package:

- increases cognitive load
- creates misleading affordances (“multimodal pipeline exists”)
- violates “no TODO in production code” release-readiness requirement

## Alternatives

- A: Keep module in `src/utils/` and simply remove TODO comment
- B: Move module to `src/eval/` as evaluation-only helper
- C: Delete module and related tests, relying on existing multimodal components (Selected)

### Decision Framework (≥9.0)

| Option                          | Complexity (40%) | Perf (30%) | Alignment (30%) |   Total |
| ------------------------------- | ---------------: | ---------: | --------------: | ------: |
| **C: Delete dead code + tests** |                9 |         10 |              10 | **9.6** |
| B: Move to eval                 |                8 |         10 |               8 |     8.8 |
| A: Keep in prod                 |               10 |         10 |               5 |     8.5 |

## Decision

We will:

1. Delete `src/utils/multimodal.py` and its dedicated test suite (`tests/unit/utils/multimodal/*`) if no production code imports it.

2. Update docs/architecture maps to remove references to the module.

## Consequences

### Positive Outcomes

- Removes unused code and TODO placeholders from production.
- Reduces maintenance burden and confusion for contributors.

### Trade-offs

- Removes a “toy” validation pipeline; if needed later, reintroduce under `src/eval/` with a proper spec.
