---
spec: SPEC-025
title: Keyword Search Tool — Sparse-only Qdrant Retriever (text-sparse)
version: 1.0.1
date: 2026-01-11
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-023: Keyword/lexical tool for exact term lookups (disabled by default).
  - NFR-MAINT-001: Library-first; avoid new retrieval dependencies for v1.
related_adrs: ["ADR-044","ADR-024"]
---

## Objective

Replace the placeholder keyword tool with a real implementation that:

- performs **sparse-only** retrieval against Qdrant named vector `text-sparse`
- reuses existing sparse encoding utilities (FastEmbed BM42/BM25)
- is gated behind `settings.retrieval.enable_keyword_tool` (default false)

## Non-goals

- Adding a BM25 dependency and building a separate inverted index
- Changing the default retrieval path (hybrid remains authoritative)

## Technical design

### Retriever

Add a small retriever class (suggested location: `src/retrieval/keyword.py`) with:

- `retrieve(query: str | QueryBundle) -> list[NodeWithScore]`
- query encoding via `src.retrieval.sparse_query.encode_to_qdrant`
- Qdrant call:
  - `client.query_points(collection_name=..., query=sparse_vec, using="text-sparse", limit=top_k, with_payload=[...])`
- deterministic ordering: score desc + id asc
- fail-open behavior:
  - if sparse encoder unavailable, return empty list (and log telemetry)

### Tool wiring

Update `src/agents/tool_factory.py::create_keyword_tool` to use the new sparse-only retriever and `build_retriever_query_engine`.

Tool description MUST clearly communicate intended use:

> “Exact keyword / ID / error-code lookup; lexical matching; not semantic.”

### Telemetry

Emit a JSONL event when the keyword tool:

- falls back (sparse unavailable)
- returns results count and latency (no raw query text)

## Testing strategy

- Unit: `tests/unit/agents/test_tool_factory_keyword.py` should validate that:
  - tool is only registered when flag true
  - tool name is `keyword_search`
  - telemetry event is emitted when sparse encoder is unavailable
  - description emphasizes lexical/keyword-only semantics
- Unit: new tests for sparse-only retriever behavior:
  - encodes query and calls `query_points(using="text-sparse")` (mock client)
  - deterministic ordering
  - empty return when sparse encoder returns None

## RTM updates (docs/specs/traceability.md)

FR-023 is implemented and recorded in the RTM:

- FR-023: “Keyword tool (sparse-only Qdrant)”
  - Code: `src/retrieval/keyword.py`, `src/agents/tool_factory.py`
  - Tests: `tests/unit/agents/test_tool_factory_keyword.py`, `tests/unit/retrieval/test_keyword_retriever.py`
  - Verification: unit tests pass; optional Qdrant integration validation when available
  - Status: Implemented
