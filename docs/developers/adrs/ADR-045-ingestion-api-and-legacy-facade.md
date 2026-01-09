---
ADR: 045
Title: Programmatic Ingestion API + Legacy Facade Replacement for src.utils.document
Status: Proposed
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 009, 024, 030
Tags: ingestion, api, legacy, docs
References:
  - https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/
---

## Description

Replace `src/utils/document.py` placeholder stubs with a **canonical programmatic ingestion API** under `src/processing/` and keep `src.utils.document` only as a thin forwarding facade (no duplicate logic).

## Context

`src/utils/document.py` currently contains async placeholders that raise `NotImplementedError` and include multiple TODOs. These symbols are referenced by:

- developer docs (`docs/developers/*`)
- some tests that patch the symbols

At the same time, the actual ingestion implementation is already present and library-first:

- `src/processing/ingestion_pipeline.py::ingest_documents_sync`
- `src/ui/_ingest_adapter_impl.py` for Streamlit uploads

Leaving placeholder stubs undermines ship readiness and documentation trust.

## Decision Drivers

- Zero broken public-facing examples in v1
- One definitive ingestion architecture (no parallel implementations)
- Avoid massive churn (update docs to preferred API, but keep compatibility path)

## Alternatives

- A: Delete `src.utils.document` and update all docs/tests immediately
- B: Replace stubs with a facade forwarding to canonical ingestion API (Selected)
- C: Keep stubs (unacceptable)

### Decision Framework (≥9.0)

| Option                             | Complexity (40%) | Perf (30%) | Alignment (30%) |   Total |
| ---------------------------------- | ---------------: | ---------: | --------------: | ------: |
| **B: Canonical API + thin facade** |                9 |          9 |               9 | **9.0** |
| A: Delete + churn                  |                6 |          9 |              10 |     8.1 |
| C: Keep stubs                      |               10 |          0 |               0 |     4.0 |

## Decision

1. Add/affirm a canonical programmatic ingestion API under `src/processing/` (typed functions that accept file paths/bytes and return `IngestionResult`).

2. Convert `src/utils/document.py` into:

- a thin forwarding layer to the canonical API
- no business logic beyond argument translation
- no TODO/NotImplemented stubs
- emits `DeprecationWarning` to encourage migration

1. Update developer docs to use the canonical API (not `src.utils.document`), while keeping `src.utils.document` functional for compatibility.

## Security & Privacy

- Ingestion API must validate file paths (no traversal) and remain local-only.
- No network access beyond explicit model backends (already gated by settings).

## Consequences

### Positive Outcomes

- Eliminates NotImplemented placeholders in production modules.
- Restores documentation correctness without requiring immediate global churn.

### Trade-offs

- Keeps a compatibility facade file in the tree until docs fully migrate.

## Changelog

- 1.0 (2026-01-09): Proposed for v1 ship readiness.
