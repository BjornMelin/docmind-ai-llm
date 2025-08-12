# ADR 002: Embedding Models and Hybrid Search

## Version/Date

v1.0 / July 22, 2025

## Status

Accepted

## Context

Effective retrieval requires accurate embeddings. Need support for multimodal/multilingual docs with good recall.

## Decision

- **Dense:** Jina v4 via HuggingFace (universal, multimodal, 10-15% better MTEB scores).
- **Sparse:** FastEmbed v0.7.1 with SPLADE++ (neural lexical for better keyword match).
- **Hybrid:** Qdrant integration with score boosting.
- **GPU:** Device mapping for acceleration.

## Rationale

- Jina v4 excels in benchmarks; FastEmbed fits for sparse (GPU support, no Jina v4 native).
- Hybrid combines semantic/lexical for superior RAG.

## Alternatives Considered

- Sentence-Transformers only: Less efficient for hybrid.
- BM25 sparse: Inferior to SPLADE++.

## Consequences

- Pros: Improved recall (15-20%), multilingual support.
- Cons: Slightly higher compute; mitigated by GPU.
