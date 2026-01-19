---
ADR: 049
Title: Multimodal Helper Cleanup — Remove Dead Test-only Utility From Production Package
Status: Implemented
Version: 1.1
Date: 2026-01-12
Supersedes:
Superseded-by:
Related: 037, 002
Tags: multimodal, cleanup, tests, maintainability
---

## Description

Remove the unused `src/utils/multimodal.py` helper (and its work-marker placeholder) from the production package and keep any validation-only helpers in tests/eval code paths.

## Context

`src/utils/multimodal.py` is currently referenced only by unit tests and contains a work-marker for “phase 2” graph traversal/LLM summarization.

Shipping unused “toy pipeline” utilities in the production package:

- increases cognitive load
- creates misleading affordances (“multimodal pipeline exists”)
- violates “no placeholder markers in production code” release-readiness requirement

## Alternatives

- A: Keep module in `src/utils/` and simply remove marker comment
- B: Move module to `src/eval/` as evaluation-only helper
- C: Delete module and related tests, relying on existing multimodal components (Selected)

### Decision Framework (≥9.0)

Weights: leverage 35%, application value 30%, maintenance 25%, adaptability 10%.

| Option | Leverage (35%) | App Value (30%) | Maintenance (25%) | Adaptability (10%) | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| **C: Delete dead code + tests (Selected)** | 9 | 9 | 10 | 9 | **9.25** |
| B: Move to eval/tests only | 5 | 3 | 6 | 5 | 4.65 |
| A: Keep in prod (remove marker only) | 2 | 1 | 2 | 3 | 1.80 |

## Decision

We will:

1. Delete `src/utils/multimodal.py` and its dedicated test suite (`tests/unit/utils/multimodal/*`) if no production code imports it.

2. Update docs/architecture maps to remove references to the module and point to the canonical multimodal implementation (`src/models/embeddings.py`, `src/retrieval/reranking.py`, `src/utils/vision_siglip.py`).

## Consequences

### Positive Outcomes

- Removes unused code and placeholder markers from production.
- Reduces maintenance burden and confusion for contributors.

### Trade-offs

- Removes a “toy” validation pipeline; if needed later, reintroduce under `src/eval/` with a proper spec.
