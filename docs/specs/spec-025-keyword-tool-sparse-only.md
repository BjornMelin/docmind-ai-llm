---
spec: SPEC-025
title: Keyword Search Tool — Sparse-only Qdrant Retriever (text-sparse)
version: 2.0.0
date: 2026-07-13
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-023: Keyword/lexical tool for exact term lookups (disabled by default).
  - NFR-MAINT-001: Library-first; avoid new retrieval dependencies for v1.
related_adrs: ["ADR-044","ADR-024"]
---

## Objective

Provide one optional keyword tool that:

- performs **sparse-only** retrieval against Qdrant named vector `text-sparse`
- reuses existing sparse encoding utilities (FastEmbed BM42/BM25)
- is gated behind `settings.retrieval.enable_keyword_tool` (default false)

## Non-goals

- Adding a BM25 dependency and building a separate inverted index
- Bypassing the canonical LlamaIndex router

## Technical design

### Retriever

`src/retrieval/keyword.py::KeywordSparseRetriever` owns sparse-only retrieval:

- `retrieve(query: str | QueryBundle) -> list[NodeWithScore]`
- query encoding via `src.retrieval.sparse_query.encode_to_qdrant`
- Qdrant call:
  - `client.query_points(collection_name=..., query=sparse_vec, using="text-sparse", limit=top_k, with_payload=[...])`
- deterministic ordering: score desc + id asc
- fail-open behavior:
  - if sparse encoding or the Qdrant query fails, return an empty list and emit
    metadata-only telemetry
- async behavior:
  - query Qdrant with the native async client
  - run FastEmbed in the retriever-owned bounded executor
  - retain executor capacity until native work finishes, even if its waiter is cancelled

### Tool wiring

`src/retrieval/router_factory.py` wraps the retriever with LlamaIndex
`RetrieverQueryEngine.from_args(...)` and adds it to the canonical router only
when `settings.retrieval.enable_keyword_tool` is true. The agent registry exposes
one `retrieve_documents` tool; it does not register keyword retrieval as a second
agent-level path.

Tool description MUST clearly communicate intended use:

> "Exact keyword, identifier, and error-code lookup via sparse matching."

### Telemetry

Emit a JSONL event when the keyword tool:

- falls back (sparse unavailable)
- returns results count and latency (no raw query text)

## Testing strategy

- `tests/unit/retrieval/test_router_factory_contract.py` validates router tool
  composition.
- `tests/unit/retrieval/test_keyword_retriever.py` validates that the retriever:
  - encodes the query and calls `query_points(using="text-sparse")` through a mock client
  - orders results deterministically
  - returns an empty result and emits telemetry when sparse encoding is unavailable

## RTM updates (docs/specs/traceability.md)

FR-023 is implemented and recorded in the RTM:

- FR-023: “Keyword tool (sparse-only Qdrant)”
  - Code: `src/retrieval/keyword.py`, `src/retrieval/router_factory.py`
  - Tests: `tests/unit/retrieval/test_router_factory_contract.py`, `tests/unit/retrieval/test_keyword_retriever.py`
  - Verification: unit tests pass; optional Qdrant integration validation when available
  - Status: Implemented
