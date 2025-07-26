# ADR-013: RRF Hybrid Search

## Title

Reciprocal Rank Fusion for Hybrid Search

## Version/Date

2.0 / July 25, 2025

## Status

Accepted

## Context

Fuse dense/sparse results (weights 0.7/0.3, alpha=60) for balanced hybrid retrieval (dense semantic, sparse keyword).

## Related Requirements

- Phase 2.1: RRF with prefetch (limit*2).
- Configurable alpha via AppSettings.

## Alternatives

- Simple average: Less effective.
- Custom fusion: Error-prone.

## Decision

Use HybridFusionRetriever (fusion_type="rrf", alpha=AppSettings.rrf_fusion_alpha or 0.7) in LlamaIndex, with prefetch.

## Related Decisions

- ADR-002 (Dense/sparse embeds).
- ADR-006 (In pipeline chain).

## Design

- **Fusion**: In utils.py: from llama_index.core.retrievers import HybridFusionRetriever; retriever = HybridFusionRetriever(dense_retriever, sparse_retriever, fusion_type="rrf", alpha=AppSettings.rrf_fusion_alpha, prefetch_k=AppSettings.prefetch_factor or 2).
- **Integration**: Use in QueryPipeline chain=[retriever, ...]. Verify with verify_rrf_configuration(settings).
- **Implementation Notes**: Alpha=0.7 favors dense (semantic). Error if alpha not 0-1.
- **Testing**: tests/test_hybrid_search.py: def test_rrf_fusion(): results = retriever.retrieve("query"); assert len(results) > 0; assert fusion_scores descending; def test_alpha_toggle(): AppSettings.rrf_fusion_alpha = 0.5; assert retriever.alpha == 0.5.

## Consequences

- Balanced hybrid (better recall/precision).
- Configurable (tune alpha via AppSettings).

- Minor compute (fusion step).
- Deps: llama-index==0.12.52.

**Changelog:**  

- 2.0 (July 25, 2025): Switched to HybridFusionRetriever; Added alpha/prefetch toggle/integration with pipeline; Enhanced testing for dev.
