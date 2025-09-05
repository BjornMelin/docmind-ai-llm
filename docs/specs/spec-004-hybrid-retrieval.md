---
spec: SPEC-004
title: Hybrid Retrieval: Qdrant Named Vectors (dense+sparse) + LanceDB BM25 Fallback with RRF Fusion
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-RET-001: Use Qdrant named vectors dense/sparse with server-side hybrid when available.
  - FR-RET-002: Provide LanceDB BM25 fallback and client-side RRF fusion.
  - FR-RET-003: Deterministic point IDs and idempotent upserts.
  - NFR-PERF-003: k=10 retrieval latency ≤ 120 ms on mid-GPU host.
related_adrs: ["ADR-005","ADR-006","ADR-010"]
---


## Objective

Implement hybrid retrieval with **Qdrant** named vectors `dense` and `sparse`. Enable server-side hybrid when supported; otherwise fallback to **LanceDB** BM25 + client **RRF** fusion.

## Libraries and Imports

```python
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.postprocessor import ReciprocalRerankFusion
```

## File Operations

### UPDATE

- `src/retrieval/query_engine.py`: add factory `build_hybrid_retriever()` returning either Qdrant hybrid retriever or LanceDB+RRF fusion.
- `src/models/storage.py`: deterministic ID helpers `sha256_id(text)`; upsert idempotent.

### CREATE

- `src/retrieval/optimization.py`: expose sliders/knobs for `alpha`, `k`, `rrf_k`, and top-k.

## Acceptance Criteria

```gherkin
Feature: Hybrid retrieval
  Scenario: Qdrant server hybrid
    Given Qdrant 1.10+ with hybrid support
    When I query with dense+sparse enabled
    Then results SHALL be returned from a single server-side call

  Scenario: LanceDB fallback with RRF
    Given Qdrant is unavailable
    Then LanceDB BM25 + dense vectors with RRF SHALL return fused results
```

## References

- Qdrant hybrid queries; LlamaIndex hybrid examples; LanceDB hybrid docs.
