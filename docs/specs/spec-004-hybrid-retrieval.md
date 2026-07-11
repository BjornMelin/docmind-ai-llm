---
spec: SPEC-004
title: Hybrid Retrieval: Qdrant Named Vectors (dense+sparse) with Server‑Side Fusion (RRF default)
version: 1.2.0
date: 2026-07-11
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-RET-001: Use Qdrant named vectors dense/sparse with server‑side hybrid fusion via Query API.
  - FR-RET-002: Deterministic point IDs and idempotent upserts.
  - FR-RET-003: Prefer FastEmbed BM42 for sparse; fallback to BM25 when unavailable.
related_adrs: ["ADR-005","ADR-006","ADR-010","ADR-024","ADR-034"]
---


## Objective

Implement hybrid retrieval with **Qdrant** named vectors `text-dense` and `text-sparse`. Use the Qdrant **Query API** to perform server‑side fusion with **RRF** by default (optionally **DBSF** for experiments). Eliminate client‑side fusion paths and LanceDB fallbacks.

## Implementation Guidance

- Collection Schema
  - `text-dense`: VectorParams(size=1024, distance=COSINE) for BGE‑M3.
  - `text-sparse`: SparseVectorParams(index=SparseIndexParams(), modifier=models.Modifier.IDF) for FastEmbed BM42/BM25.
  - Ensure collections are created with these exact names; code idempotently enforces schema before using the store.

- Query API (server‑side fusion)
  - Use `prefetch` for sparse and dense, then set `fusion` to `rrf` (default) or `dbsf` (experimental; env‑gated). Dense queries use `VectorInput`; sparse queries use `SparseVector` built via FastEmbed BM42/BM25 to align with index‑time sparse. Example:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

result = client.query_points_groups(
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
    # env: DOCMIND_RETRIEVAL__FUSION_MODE=rrf|dbsf
    query=models.RrfQuery(rrf=models.Rrf(k=60)),
    group_by="page_id",
    group_size=1,
    limit=60,
    with_payload=["doc_id", "page_id", "chunk_id", "text", "has_image"],
)
```

- Sparse Model
  - Prefer BM42 (`Qdrant/bm42-all-minilm-l6-v2-attentions`); fallback to BM25 when BM42 is unavailable.

- Fusion Choice
  - RRF is robust across distributions; DBSF can be evaluated per-corpus behind a feature flag.

## Development Notes

- De‑duplication: use Qdrant `query_points_groups()` with `group_by=page_id|doc_id`, `group_size=1`, and `limit=fused_top_k`. Do not overfetch or deduplicate fused points in Python.
- Settings: `retrieval.dedup_key=page_id|doc_id` selects the server grouping key. `retrieval.rrf_k` is passed to Qdrant through `RrfQuery(Rrf(k=...))`; DBSF uses `FusionQuery(Fusion.DBSF)`.
- Telemetry: log prefetch sizes, fusion mode, fused_top_k, query latency, and `retrieval.sparse_fallback=true` when sparse prefetch is skipped.

### Router Interop (Note)

- Compose router tools `[semantic_search, hybrid_search, knowledge_graph]` when a graph is present and healthy; otherwise `[semantic_search, hybrid_search]`. Health is determined via a shallow probe: `pg_index.as_retriever(include_text=False, path_depth=1, similarity_top_k=1).retrieve("health")` must return at least one result. Selector preference: `PydanticSingleSelector` when available, else `LLMSingleSelector`. See ADR‑038 and SPEC‑006.
- Rerank parity: RouterQueryEngine tools (vector/hybrid/KG) apply the same reranking policy via `node_postprocessors` when `DOCMIND_RETRIEVAL__USE_RERANKING=true`. See SPEC‑005 for reranking architecture and attachment points.

- Agent tool fallback: At the tool layer, when a hybrid query returns no documents for the primary query, the agent MUST fall back to vector search for that query (if available) and record a `hybrid_fallback` telemetry event.

## Libraries and Imports

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, SparseVector, Fusion
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
```

## File Operations

### UPDATE / CREATE

- `src/retrieval/hybrid.py`: implement `ServerHybridRetriever` using qdrant_client Query API with `prefetch` on `text-dense` and `text-sparse`, native RRF/DBSF fusion, and server grouping before `limit=fused_top_k`.
- `src/retrieval/router_factory.py`: register a `hybrid_search` tool by wrapping `ServerHybridRetriever` in `RetrieverQueryEngine`; compose with `semantic_search` and optional `knowledge_graph` tool.
- `src/utils/storage.py`: precreate collections with named vectors `text-dense` (COSINE, BGE‑M3 1024D) and `text-sparse` (SparseIndexParams); prefer `fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions"` else `"Qdrant/bm25"`.
- `src/models/processing.py`: own the parser-reserved
  `docmind_document_id` metadata key.
- `src/ui/_ingest_adapter_impl.py`: assign deterministic UUIDv5 point IDs from
  canonical document ID, page ID, and chunk position; after a successful
  upsert, delete only stale IDs captured for the affected documents.

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

- Qdrant hybrid queries; LlamaIndex hybrid examples.

## Changelog

- 1.1.1 (2025-09-09): Added Router interop note and cross‑link to GraphRAG spec/ADR

## Server‑Side Hybrid Only (Qdrant Query API)

- All hybrid queries MUST be executed server‑side via Qdrant Query API using `Prefetch` (sparse+dense), `RrfQuery` for RRF, and `FusionQuery` for DBSF.
- Default fusion mode: RRF; DBSF MAY be exposed behind an environment flag. No UI toggles for fusion modes.
- Named vectors MUST exist: `text-dense` and `text-sparse`; the sparse vector SHOULD use IDF modifier when supported.
- Grouping MUST occur in Qdrant by a configured key (default `page_id`) before the final fused cut.
- Telemetry MUST include: `retrieval.fusion_mode`, `retrieval.prefetch_*`, `retrieval.fused_limit`, `retrieval.return_count`, `retrieval.latency_ms`, `retrieval.sparse_fallback`, and `dedup.*`.

### Prohibited

- Client‑side fusion or weight mixing is NOT allowed in production paths.
- UI toggles for retrieval fusion/reranking are NOT allowed; retrieval knobs remain environment‑only for testing.
