# ADR-007: Reranking Strategy

## Title

Native ColBERT Reranking with QueryPipeline Integration

## Version/Date

3.1 / August 13, 2025

## Status

Accepted

## Description

Implements ColbertRerank postprocessor within QueryPipeline for token-level relevance scoring with local processing, selecting top-5 documents from retrieval pool.

## Context

Following ADR-021's Native Architecture Consolidation, document reranking uses native LlamaIndex ColbertRerank postprocessor integrated within QueryPipeline for precision improvement. ColBERT late-interaction provides token-level relevance scoring with efficient local processing, selecting top_n=5 documents from initial retrieval pool for superior relevance vs speed optimization.

## Related Requirements

- **Native Postprocessor Integration**: Direct QueryPipeline integration with LlamaIndex ColbertRerank

- **Offline/Local Processing**: FastEmbed ColBERT model for privacy-preserving reranking

- **Performance Optimization**: Balance between precision improvement and processing speed

- **Score Fusion**: Combine retrieval and reranking scores for optimal relevance

- **Configurable Parameters**: Flexible top_k selection based on use case requirements

## Alternatives

- **No reranking**: Faster but significantly lower precision, reduced relevance quality

- **LLM-based reranking**: Higher accuracy but much slower, token cost implications

- **Custom reranking logic**: Maintenance-heavy, violates library-first principle

- **External reranking services**: Violates offline/privacy requirements

## Decision

Use native LlamaIndex ColbertRerank postprocessor integrated within QueryPipeline for optimal precision-performance balance with local processing.

**Integration Simplification:**

- **BEFORE**: Custom reranking implementations with separate pipeline integration

- **AFTER**: Native ColbertRerank postprocessor with zero custom reranking code

## Related Decisions

- ADR-021 (LlamaIndex Native Architecture Consolidation - enables native postprocessors)

- ADR-020 (LlamaIndex Settings Migration - unified reranking configuration)

- ADR-022 (Tenacity Resilience Integration - robust reranking with retry patterns)

- ADR-006 (Analysis Pipeline - QueryPipeline chain integration)

- ADR-013 (RRF Hybrid Search - post-retrieval reranking stage)

- ADR-001 (Architecture Overview - post-hybrid retrieval precision improvement)

- ADR-003 (GPU Optimization - provides GPU acceleration for ColBERT reranking operations)

## Design

**Native ColbertRerank Integration:**

```python

# In utils.py: Native postprocessor with QueryPipeline
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.query_pipeline import QueryPipeline

# Native simplification: Native postprocessor integration
reranker = ColbertRerank(
    model="colbert-ir/colbertv2.0",  # FastEmbed local model
    top_n=AppSettings.reranking_top_k or 5,
    keep_retrieval_score=True,  # Enable score fusion
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Direct QueryPipeline integration
qp = QueryPipeline(
    chain=[
        HybridFusionRetriever(dense_retriever, sparse_retriever),
        reranker,  # Native postprocessor stage
        synthesizer
    ],
    async_mode=True,
    parallel=True
)
```

**Advanced Score Fusion Configuration:**

```python

# In models.py: Enhanced reranking settings
class RerankingSettings(BaseModel):
    enable_colbert_reranking: bool = Field(
        default=True, 
        description="Enable ColBERT late-interaction reranking"
    )
    reranking_top_k: int = Field(
        default=5, 
        description="Number of documents to return after reranking"
    )
    retrieval_top_k: int = Field(
        default=20, 
        description="Initial retrieval pool size (rerank from top 20 → top 5)"
    )
    score_fusion_weight: float = Field(
        default=0.7, 
        description="Weight for reranking score (0.7 rerank + 0.3 retrieval)"
    )
    
    @validator("score_fusion_weight")
    def validate_fusion_weight(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score fusion weight must be between 0.0 and 1.0")
        return v

# Adaptive reranking configuration
def create_adaptive_reranker(settings: RerankingSettings) -> ColbertRerank:
    """Create ColBERT reranker with adaptive configuration."""
    return ColbertRerank(
        model="colbert-ir/colbertv2.0",
        top_n=settings.reranking_top_k,
        keep_retrieval_score=True,
        device="cuda" if settings.gpu_acceleration else "cpu"
    )
```

**QueryPipeline Integration with Multimodal Support:**

```python

# Enhanced pipeline with multimodal reranking considerations
async def create_reranked_query_pipeline(settings: AppSettings) -> QueryPipeline:
    """Create QueryPipeline with optimized reranking integration."""
    
    # Primary text reranking
    text_reranker = ColbertRerank(
        model="colbert-ir/colbertv2.0",
        top_n=settings.reranking_top_k,
        keep_retrieval_score=True
    )
    
    # For multimodal: Separate reranking for text vs image results
    if settings.multimodal_enabled:
        # Text documents through ColBERT reranking
        text_pipeline = QueryPipeline(
            chain=[text_retriever, text_reranker, text_synthesizer]
        )
        
        # Image documents use similarity-based ranking (CLIP ViT-B/32 scores)
        image_pipeline = QueryPipeline(
            chain=[image_retriever, similarity_reranker, image_synthesizer]
        )
        
        # Combine results with score normalization
        return MultimodalQueryPipeline(text_pipeline, image_pipeline)
    
    else:
        # Standard text-only reranking pipeline
        return QueryPipeline(
            chain=[hybrid_retriever, text_reranker, synthesizer],
            async_mode=True
        )
```

**Performance-Optimized Implementation:**

```python

# Batched reranking for efficiency
class OptimizedColbertRerank(ColbertRerank):
    """Performance-optimized ColBERT reranking with batching."""
    
    def __init__(self, *args, batch_size: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
    
    async def arun(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """Async batched reranking for large document sets."""
        if len(nodes) <= self.batch_size:
            return await super().arun(nodes, query)
        
        # Process in batches for memory efficiency
        reranked_batches = []
        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i:i + self.batch_size]
            reranked_batch = await super().arun(batch, query)
            reranked_batches.extend(reranked_batch)
        
        # Re-sort combined results by score
        return sorted(reranked_batches, key=lambda x: x.score, reverse=True)[:self.top_n]
```

**Configuration-Based Toggle System:**

```python

# In app.py: Dynamic reranking configuration
def configure_reranking_pipeline(settings: AppSettings) -> QueryPipeline:
    """Configure QueryPipeline with optional reranking based on settings."""
    
    base_chain = [hybrid_retriever]
    
    # Optional ColBERT reranking stage
    if settings.enable_colbert_reranking:
        reranker = create_adaptive_reranker(settings.reranking)
        base_chain.append(reranker)
        logger.info(f"ColBERT reranking enabled: top_{settings.reranking.reranking_top_k}")
    else:
        logger.info("ColBERT reranking disabled - using retrieval scores only")
    
    base_chain.append(synthesizer)
    
    return QueryPipeline(chain=base_chain, async_mode=True)
```

**Testing Strategy:**

```python

# In tests/test_reranking.py: Comprehensive reranking validation
async def test_native_colbert_reranking():
    """Test native ColbertRerank postprocessor integration."""
    reranker = ColbertRerank(
        model="colbert-ir/colbertv2.0",
        top_n=5,
        keep_retrieval_score=True
    )
    
    # Test reranking quality improvement
    query = "machine learning algorithms"
    initial_nodes = await retriever.aretrieve(query)  # 20 results
    reranked_nodes = await reranker.arun(initial_nodes, query)  # Top 5
    
    assert len(reranked_nodes) == 5
    assert all(hasattr(node, 'score') for node in reranked_nodes)
    
    # Verify score improvement (reranked should be more relevant)
    assert reranked_nodes[0].score >= reranked_nodes[-1].score  # Descending order

async def test_query_pipeline_reranking_integration():
    """Test QueryPipeline with integrated ColBERT reranking."""
    pipeline = QueryPipeline(
        chain=[hybrid_retriever, reranker, synthesizer],
        async_mode=True
    )
    
    response = await pipeline.arun(query="complex technical query")
    
    # Verify pipeline processing
    assert response.response is not None
    assert len(response.source_nodes) <= 5  # Reranked to top 5
    assert all(node.score > 0.5 for node in response.source_nodes)  # High relevance

@pytest.mark.parametrize("top_k", [3, 5, 10])
async def test_configurable_reranking_parameters(top_k):
    """Test different reranking configurations."""
    reranker = ColbertRerank(top_n=top_k, keep_retrieval_score=True)
    
    nodes = await generate_test_nodes(20)  # Generate 20 test nodes
    reranked = await reranker.arun(nodes, "test query")
    
    assert len(reranked) == min(top_k, len(nodes))
    assert all(reranked[i].score >= reranked[i+1].score for i in range(len(reranked)-1))
```

## Consequences

### Positive Outcomes

- **Improved Precision**: ColBERT late-interaction provides token-level relevance matching vs simple similarity

- **Native Integration**: Zero custom reranking code through LlamaIndex postprocessor architecture

- **Offline Processing**: FastEmbed ColBERT model enables privacy-preserving local reranking

- **Flexible Configuration**: Configurable top_k, score fusion, and pipeline integration options

- **Performance Balance**: Efficient reranking from 20 → 5 documents vs processing all retrieved content

- **GPU Acceleration**: Automatic CUDA utilization for ColBERT inference when available

### Ongoing Considerations

- **Monitor Reranking Quality**: Track precision improvements vs retrieval-only results

- **GPU Memory Usage**: ColBERT model requires ~1-2GB VRAM for efficient processing

- **Processing Latency**: Balance reranking quality improvements vs query response time

- **Score Calibration**: Tune score fusion weights based on retrieval and reranking performance

- **Batch Size Optimization**: Optimize batch processing for large document collections

### Dependencies

- **Core**: llama-index-postprocessor-colbert-rerank>=0.3.0 (native postprocessor)

- **Model**: colbert-ir/colbertv2.0 via FastEmbed (local ColBERT model)

- **Native**: llama-index>=0.12.0 (QueryPipeline and postprocessor integration)

- **Enhanced**: Native postprocessor architecture (replaces custom reranking logic)

**Changelog:**  

- 3.1 (August 13, 2025): Added cross-references to GPU optimization (ADR-003) for ColBERT reranking acceleration. Removed marketing language for technical precision.

- 3.0 (August 13, 2025): Native postprocessor integration with LlamaIndex ColbertRerank. Updated multimodal support, performance optimization with batching, and score fusion configuration. Zero custom reranking code through native architecture. Aligned with ADR-021's Native Architecture Consolidation.

- 2.0 (July 25, 2025): Integrated with QueryPipeline; Added toggle/score fusion; Enhanced testing for dev.
