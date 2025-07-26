# ADR-007: Reranking Strategy

## Title

Document Reranking for Improved Relevance

## Version/Date

2.0 / July 25, 2025

## Status

Accepted

## Context

Rerank post-retrieval for precision (ColBERT late-interaction, top_n=5 from 20 retrieve).

## Related Requirements

- Offline/local (FastEmbed Colbert model).
- Integrate in QueryPipeline stage.

## Alternatives

- No rerank: Lower precision.
- LLM rerank: Slower, higher cost.

## Decision

Use ColbertRerank (model="colbert-ir/colbertv2.0", top_n=AppSettings.reranking_top_k or 5, keep_retrieval_score=True) as postprocessor in QueryPipeline.

## Related Decisions

- ADR-006 (In pipeline chain).
- ADR-001 (Post-hybrid retrieval).

## Design

- **Init**: In utils.py create_tools_from_index: from llama_index.postprocessor import ColbertRerank; reranker = ColbertRerank(top_n=AppSettings.reranking_top_k, keep_retrieval_score=True).
- **Integration**: Add to QueryPipeline chain=[retriever, reranker, ...]. For multimodal, combine with Jina m0 if images.
- **Implementation Notes**: Toggle via AppSettings.enable_colbert_reranking. Fuse scores (0.5 *rerank + 0.5* retrieval).
- **Testing**: tests/test_reranking.py: def test_colbert_rerank(): results = reranker.postprocess_nodes(nodes, "query"); assert len(results) == 5; assert all(r.score > 0 for r in results); scores descending; def test_score_fusion(): assert fused_score == expected.

## Consequences

- Higher precision (token-level interaction).
- Local/offline (FastEmbed).

- Compute (GPU via AppSettings).
- Deps: llama-index-postprocessor-colbert-rerank>=0.3.0.

**Changelog:**  

- 2.0 (July 25, 2025): Integrated with QueryPipeline; Added toggle/score fusion; Enhanced testing for dev.
