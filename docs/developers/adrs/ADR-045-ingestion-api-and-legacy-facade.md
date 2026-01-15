---
ADR: 045
Title: Unified Ingestion Architecture (Greenfield)
Status: Accepted
Version: 2.0
Date: 2026-01-15
Supersedes:
Superseded-by:
Related: 009, 024, 030
Tags: ingestion, api, architecture, cleanup
References:
  - https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/
  - SPEC-026
---

## Description

Establish `src/processing/ingestion_api.py` as the **sole canonical entry point** for file loading and ingestion input normalization. Fully deprecate and **delete** `src/utils/document.py`.

## Context

The codebase currently splits ingestion responsibilities:

- `src/processing/ingestion_pipeline.py`: Handles LlamaIndex pipeline assembly (chunking, embedding, storage).
- `src/utils/document.py`: Handles file loading (`UnstructuredReader`), path sanitization, and basic IO.

This split creates ambiguity and technical debt. `src/utils/document.py` was originally intended as a temporary home or a facade, but it has accumulated business logic (Unstructured fallback, logic for directory traversal).

To achieve a "production-ready, final-release" standard with no technical debt, we must unify the "Loader" layer.

## Decision Drivers

- **Modularity:** Ingestion logic should reside in one cohesive package (`src/processing`).
- **Maintainability:** Removing aliased paths prevents confusion for future developers.
- **Security:** Centralizing input validation in one module reduces the surface area for path traversal bugs.

## Alternatives

- **A: Keep Facade (ADR-045 v1.0)**: Move logic but keep `src/utils/document.py` as a proxy.
  - *Pros*: Backwards compatible.
  - *Cons*: Adds cognitive load; confusing "two ways to do one thing".
- **B: Strict Unification (Selected)**: Delete `src/utils/document.py` entirely.
  - *Pros*: Clean namespace; clear ownership; Greenfield best practice.
  - *Cons*: Requires updating all call sites (accepted cost).

### Decision Framework

| Model / Option                    | Maintainability (40%) | Architecture Clarity (30%) | Backwards Compatibility (10%) | Implementation Cost (20%) | Total Score | Decision        |
| --------------------------------- | --------------------- | -------------------------- | ----------------------------- | ------------------------- | ----------- | --------------- |
| **Strict Unification (Selected)** | 10                    | 10                         | 0                             | 6                         | **8.2**     | **Selected**    |
| Keep Facade                       | 5                     | 4                          | 10                            | 8                         | 5.8         | Rejected        |

## Decision

1. **create** `src/processing/ingestion_api.py`.
    - This module will house the logic previously in `src/utils/document.py` (loading, path validation, hashing).
    - It will serve as the "Input Layer" that feeds into `ingestion_pipeline.py`.
2. **Refactor & Move** all logic from `src/utils/document.py` to `src/processing/ingestion_api.py`.
3. **Delete** `src/utils/document.py`.
    - Zero backwards compatibility placeholders.
    - Codebase is updated atomically to use the new path.
4. **Enforce Strict Layering**:
    - `src/processing/ingestion_api.py` -> Loads Files -> Returns `IngestionInput` / `Document`.
    - `src/processing/ingestion_pipeline.py` -> Consumes `Document` -> Returns `Nodes` / `IngestionResult`.

## Related Requirements

### Functional Requirements

- **FR-024 (Canonical Ingestion API):** Developers must have a single, safe API for loading documents from disk.

### Non-Functional Requirements

- **NFR-MAINT-003 (Zero Debt):** No placeholder shims or deprecated modules allowed in v1.
- **NFR-SEC-001 (Offline First):** Ingestion must remain local-only; no implicit network calls.

## Consequences

### Positive Outcomes

- **Single Source of Truth**: All "ingestion" code lives in `src/processing/`.
- **Reduced Cognitive Load**: No need to check `utils` for core business logic.
- **Clean Namespace**: `src.utils` is reserved for true utilities (hashing, time), not domain logic.

### Trade-offs

- **Breaking Change**: Any external scripts or forks depending on `src.utils.document` will break. (Accepted per "Greenfield" mandate).

## Changelog

- 1.0 (2026-01-09): Proposed Facade approach.
- 2.0 (2026-01-15): Updated to Strict Unification (Delete Utils) based on deep architectural review and template compliance.
