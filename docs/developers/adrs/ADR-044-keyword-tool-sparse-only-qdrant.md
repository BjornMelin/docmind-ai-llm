---
ADR: 044
Title: Keyword Search Tool via Sparse-only Qdrant Query (No New BM25 Dependency)
Status: Implemented
Version: 1.1
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 003, 024, 028
Tags: retrieval, qdrant, hybrid, sparse
References:
  - https://qdrant.tech/documentation/
  - https://github.com/qdrant/qdrant-client
---

## Description

Implement DocMind’s optional `keyword_search` tool as a **sparse-only Qdrant query** using existing FastEmbed sparse encoders and the named sparse vector (`text-sparse`).

## Context

The current keyword tool is a placeholder (`src/agents/tool_factory.py`) and is disabled by default. DocMind already supports server-side hybrid retrieval (dense+sparse fusion) via Qdrant Query API, but agent routing benefits from a distinct “keyword/lexical” tool for:

- exact term/ID/error code lookups
- acronym-heavy corpora
- deterministic lexical matching behavior

Adding a separate BM25 dependency or building a parallel inverted index would violate KISS and increase maintenance.

## Decision Drivers

- No new dependencies for v1
- Keep Qdrant as the single retrieval source-of-truth
- Provide distinct tool semantics for agent routing
- Reuse existing sparse encoding (`src/retrieval/sparse_query.py`)

## Alternatives

- A: Remove keyword tool entirely — reduces tool diversity for routing
- B: Sparse-only Qdrant keyword tool (Selected)
- C: Add BM25 dependency and implement BM25 retriever — higher complexity and operational risk

### Decision Framework (≥9.0)

| Option                    | Complexity (40%) | Perf (30%) | Alignment (30%) |   Total |
| ------------------------- | ---------------: | ---------: | --------------: | ------: |
| **B: Sparse-only Qdrant** |                9 |          9 |              10 | **9.3** |
| A: Remove                 |               10 |          7 |               6 |     7.9 |
| C: BM25 dep               |                5 |          8 |               6 |     6.1 |

## Decision

Implement `keyword_search` as a retriever that:

- encodes the query to `qdrant_client.models.SparseVector` using existing FastEmbed sparse encoder
- queries Qdrant with `query_points(..., query=sparse_vec, using="text-sparse")`
- returns `NodeWithScore` results with payload fields required downstream

The tool remains **disabled by default** behind `settings.retrieval.enable_keyword_tool`.

## Security & Privacy

- No new network surfaces beyond the existing Qdrant dependency.
- Do not log raw query content; use safe summaries.

## Consequences

### Positive Outcomes

- Keyword tool becomes real and useful without new deps.
- Clearer tool differentiation for agent routing.

### Trade-offs

- Sparse-only search depends on sparse encoder availability; fail open when unavailable.

## Changelog

- 1.0 (2026-01-09): Proposed for v1 release hardening.
- 1.1 (2026-01-11): Implemented sparse-only Qdrant keyword tool and unit tests.
