---
spec: SPEC-004
title: Hybrid Retrieval: Qdrant Named Vectors (dense+sparse) with Server‑Side Fusion (RRF default)
version: 1.1.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-RET-001: Use Qdrant named vectors dense/sparse with server‑side hybrid fusion via Query API.
  - FR-RET-002: Deterministic point IDs and idempotent upserts.
  - FR-RET-003: Prefer FastEmbed BM42 for sparse; fallback to BM25 when unavailable.
  - NFR-PERF-003: fused_top_k=60; server‑side hybrid latency ≤ 120–200 ms (dataset/hardware dependent).
related_adrs: ["ADR-005","ADR-006","ADR-010","ADR-024"]
---


## Objective

Implement hybrid retrieval with **Qdrant** named vectors `text-dense` and `text-sparse`. Use the Qdrant **Query API** to perform server‑side fusion with **RRF** by default (optionally **DBSF** for experiments). Eliminate client‑side fusion paths and LanceDB fallbacks.

## Implementation Guidance

- Collection Schema
  - `text-dense`: VectorParams(size=1024, distance=COSINE) for BGE‑M3.
  - `text-sparse`: SparseVectorParams(index=SparseIndexParams(), modifier=models.Modifier.IDF) for FastEmbed BM42/BM25.
  - Precreate collections with these exact names if managing schema outside LlamaIndex.

- Query API (server‑side fusion)
  - Use `prefetch` for sparse and dense, then set `fusion` to `rrf` (default) or `dbsf` (experimental; env‑gated). Example:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

result = client.query_points(
    collection_name="doc_pages",
    prefetch=[
        models.Prefetch(
            query=models.SparseVector(indices=sp_idx, values=sp_vals),
            using="text-sparse",
            limit=400,
        ),
        models.Prefetch(
            query=models.VectorInput(vector=dense_vec),
            using="text-dense",
            limit=200,
        ),
    ],
    # env: HYBRID_FUSION_MODE=rrf|dbsf
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=60,
    with_payload=["doc_id", "page_id", "chunk_id", "text", "has_image"],
)
```

- Sparse Model
  - Prefer BM42 (`Qdrant/bm42-all-minilm-l6-v2-attentions`); fallback to BM25 when BM42 is unavailable.

- Fusion Choice
  - RRF is robust across distributions; DBSF can be evaluated per-corpus behind a feature flag.

## Development Notes

- De‑duplication: Collapse by `page_id` before the final fused cut (`limit`) to prevent over‑representing a single page. Use a fused candidate buffer (e.g., fused_top_k=60), dedup, then slice to top_k.
- Latency targets: p50 120–200 ms for fused_top_k=60 on typical local setup; tune prefetch limits to stay within SLOs.
- Telemetry: log prefetch sizes, fusion mode, fused_top_k, and query latency.

## Libraries and Imports

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, SparseVector, Fusion
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
```

## File Operations

### UPDATE

- `src/retrieval/query_engine.py`: implement server‑side hybrid queries using qdrant_client Query API with `prefetch` on `text-dense` and `text-sparse`, `fusion=Fusion.rrf`, `limit=fused_top_k`.
- `src/utils/storage.py`: precreate collections with named vectors `text-dense` (COSINE, BGE‑M3 1024D) and `text-sparse` (SparseIndexParams); prefer `fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions"` else `"Qdrant/bm25"`.
- `src/models/storage.py`: deterministic ID helpers `sha256_id(text)`; idempotent upserts.

## Acceptance Criteria

```gherkin
Feature: Hybrid retrieval (server‑side fusion)
  Scenario: Qdrant server hybrid (RRF)
    Given Qdrant 1.10+ with Query API and named vectors
    When I query with dense and sparse prefetch configured
    Then results SHALL be fused server‑side using RRF and returned in a single call

  Scenario: Qdrant server hybrid (DBSF experimental)
    Given the same setup
    When I switch fusion to DBSF
    Then results SHALL be fused server‑side using DBSF

  Scenario: Fallback dense-only when sparse unavailable
    Given fastembed is not installed or sparse embeddings are disabled
    When I query
    Then the system SHALL fall back to dense-only retrieval and log the fallback
```

## References

- Qdrant hybrid queries; LlamaIndex hybrid examples; LanceDB hybrid docs.
