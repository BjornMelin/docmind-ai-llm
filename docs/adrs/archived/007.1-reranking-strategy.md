# ADR 007: Reranking Strategy for RAG

## Version/Date

v1.0 / July 22, 2025

## Status

Superseded by ColBERT Implementation (July 2025)

## Context

Retrieval-Augmented Generation (RAG) requires high-quality context. Reranking improves relevance and reduces redundancy.

## Decision

- Use **Jina Reranker v2** (sentence-transformers v5.0.0) via CrossEncoder for scoring retrieved documents.

- Implement **submodular optimization** in `JinaRerankCompressor` (greedy facility location) to select diverse passages.

- Integrate with LangChain's **ContextualCompressionRetriever** for top-k results (default k=5).

- Support GPU acceleration with device='cuda'.

## Rationale

- Jina Reranker v2 offers strong performance on MTEB.

- Submodular optimization reduces redundancy by 20-30% (per Jina article).

- ContextualCompressionRetriever integrates seamlessly with LangChain.

## Alternatives Considered

- No reranking: Lower context quality.

- BM25-based reranking: Less effective than neural models.

## Consequences

- Pros: Improved RAG quality, GPU-accelerated.

- Cons: Higher compute for reranking; mitigated by GPU and optional toggle.
