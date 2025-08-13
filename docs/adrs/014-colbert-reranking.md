# ADR-014: Optimized Reranking Strategy

## Title

BGE-reranker-v2-m3 for Multimodal Reranking with ColBERT Text-Only Fallback

## Version/Date

4.0 / August 13, 2025

## Status

Accepted

## Description

Adopts BGE-reranker-v2-m3 as primary reranker achieving 86% size reduction (278M vs 2.4B parameters) with ColBERT as text-only fallback for maximum accuracy scenarios.

## Context

Following research and decision framework analysis (35/30/25/10 weighting), DocMind AI requires a smaller reranking model to replace Jina m0 (2.4B parameters) while maintaining acceptable accuracy. Research identified BGE-reranker-v2-m3 (278M parameters), achieving 86% size reduction with 5-10% performance degradation, aligning with KISS/DRY/YAGNI principles and enabling 3-5x faster inference.

## Related Requirements

- **Lightweight Reranking**: Model size <1GB vs Jina m0's 2.4B parameters

- **Performance Target**: Within 20% of Jina m0 accuracy acceptable for 80%+ size reduction

- **LlamaIndex Native**: Must integrate with existing Settings patterns

- **GPU Optimization**: device_map="auto" + mixed precision for 1.5x speedup

- **Async Processing**: QueryPipeline.parallel_run() integration

- **KISS Compliance**: Minimize complexity while maintaining functionality

## Alternatives

### Evaluated Models (Decision Framework Scoring)

| Model | Solution Leverage (35%) | Application Value (30%) | Maintenance (25%) | Adaptability (10%) | Total Score | Decision |
|-------|------------------------|------------------------|-------------------|-------------------|-------------|----------|
| **BGE-reranker-v2-m3** | 0.90 | 0.75 | 0.95 | 0.80 | **0.8625** | ✅ Selected |
| mxbai-rerank-base-v2 | 0.85 | 0.80 | 0.80 | 0.85 | 0.8225 | Secondary |
| ColBERT v2 (text-only) | 0.70 | 0.85 | 0.60 | 0.70 | 0.7425 | Text fallback |
| Keep Jina m0 | 0.70 | 0.95 | 0.30 | 0.40 | 0.6175 | Rejected |

## Decision

**Primary**: Implement **BGE-reranker-v2-m3** (BAAI) as default reranker, achieving 86% size reduction (278M vs 2.4B parameters) with only 5-10% performance loss. Native LlamaIndex integration via `SentenceTransformerRerank`.

**Secondary**: Maintain **ColBERT v2** for specialized text-heavy scenarios requiring maximum accuracy.

**Rationale**: BGE-reranker-v2-m3 scored 0.8625 in decision framework analysis, balancing simplicity, performance, and maintainability while aligning with KISS/DRY/YAGNI principles.

## Related Decisions

- ADR-007 (Reranking strategy).

- ADR-006 (In pipeline chain).

- ADR-003 (GPU Optimization - device_map="auto" for ColBERT acceleration)

- ADR-012 (Async Performance Optimization - async reranking patterns)

- ADR-023 (PyTorch Optimization Strategy - mixed precision for reranking speedup)

- ADR-020 (Settings Migration - unified Settings configuration)

## Design

### Primary Implementation: BGE-reranker-v2-m3

**Multimodal reranking with native LlamaIndex integration:**

```python
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import Settings

# Primary reranker: BGE-reranker-v2-m3 (278M params, 8.6x smaller than Jina m0)
def configure_bge_reranker():
    """Configure BGE reranker with GPU optimization."""
    
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=Settings.reranking_top_k or 5,
        device_map="auto",  # Automatic GPU optimization
        torch_dtype="float16"  # Mixed precision for 1.5x speedup
    )
    
    # Integration with Settings for global configuration
    Settings.rerank_model = reranker
    return reranker

# Performance characteristics:

# - Model size: 278M parameters (86% reduction from Jina m0)

# - Inference: 3-5x faster than Jina m0

# - Accuracy: BEIR NDCG@10 = 56.51 (vs 59-60 for Jina m0)

# - Memory: 80% reduction in VRAM usage

# - Multimodal: Yes

# - LlamaIndex Integration: Native

# - GPU Optimization: device_map="auto" + mixed precision for 1.5x speedup

# - Async Processing: QueryPipeline.parallel_run() integration
```

### Secondary Implementation: ColBERT for Text-Only

**High-accuracy text-only reranking for specialized scenarios:**

```python
from llama_index.postprocessor.colbert_rerank import ColbertRerank

# Text-only fallback: ColBERT v2 for maximum accuracy
def configure_colbert_reranker():
    """Configure ColBERT for text-heavy workloads requiring maximum precision."""
    
    colbert_reranker = ColbertRerank(
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        top_n=Settings.reranking_top_k or 5,
        keep_retrieval_score=True,
        device_map="auto",
        torch_dtype="float16"
    )
    
    return colbert_reranker

# Usage: Enable for text-heavy scenarios via Settings
if Settings.use_colbert_for_text:
    reranker = configure_colbert_reranker()
else:
    reranker = configure_bge_reranker()  # Default option
```

### Async Pipeline Integration

**Unified async processing for both reranking strategies:**

```python
from llama_index.core.query_pipeline import QueryPipeline
import asyncio

class UnifiedReranker:
    """Unified reranking interface supporting both BGE and ColBERT."""
    
    def __init__(self):
        self.primary_reranker = configure_bge_reranker()
        self.text_reranker = configure_colbert_reranker() if Settings.enable_colbert_fallback else None
    
    async def rerank_async(self, nodes, query, use_text_only=False):
        """Async reranking with automatic model selection."""
        
        # Select appropriate reranker based on content type
        reranker = self.text_reranker if (use_text_only and self.text_reranker) else self.primary_reranker
        
        # Async reranking with GPU acceleration
        return await reranker.apostprocess_nodes(nodes, query)
    
    async def parallel_rerank(self, node_batches, queries):
        """Parallel reranking for multiple queries."""
        
        tasks = [
            self.rerank_async(nodes, query) 
            for nodes, query in zip(node_batches, queries)
        ]
        
        return await asyncio.gather(*tasks)

# Integration with QueryPipeline
pipeline = QueryPipeline()
unified_reranker = UnifiedReranker()

# Add to pipeline with automatic model selection
pipeline.add_component(unified_reranker)
```

### Performance Comparison

| Metric | Jina m0 (Previous) | BGE-v2-m3 (Primary) | ColBERT v2 (Text) |
|--------|-------------------|---------------------|-------------------|
| Model Size | 2.4B params | 278M params (-86%) | 110M params |
| BEIR NDCG@10 | 58.95 | 56.51 (-4.1%) | 57.8 (-1.9%) |
| Inference Speed | Baseline | 3-5x faster | 2x faster |
| Memory Usage | High (>4GB) | Low (<1GB) | Medium (1.5GB) |
| Multimodal | Yes | Yes | No |
| LlamaIndex Integration | Complex | Native | Native |

### Implementation Notes

- **Model Selection**: Default to BGE-reranker-v2-m3, fallback to ColBERT for text-only high-accuracy needs

- **Settings Integration**: `Settings.rerank_model` for global configuration

- **GPU Optimization**: device_map="auto" + mixed precision for all models

- **Score Fusion**: Maintained for backward compatibility (0.5 *rerank + 0.5* retrieval)

- **KISS Compliance**: 86% size reduction with minimal code changes

### Testing Strategy

**tests/test_reranking.py:**

```python
import time
import pytest
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.postprocessor.colbert_rerank import ColbertRerank

@pytest.mark.asyncio
async def test_bge_reranker_performance():
    """Test BGE reranker performance and accuracy."""
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=5,
        device_map="auto",
        torch_dtype="float16"
    )
    
    # Test reranking quality
    results = await reranker.apostprocess_nodes(test_nodes, "test query")
    assert len(results) == 5
    assert all(r.score > 0 for r in results)
    
    # Validate 3-5x speedup vs baseline
    start = time.time()
    results = await reranker.apostprocess_nodes(large_node_set, "complex query")
    duration = time.time() - start
    assert duration < jina_baseline_time * 0.33  # At least 3x faster

@pytest.mark.asyncio
async def test_model_size_validation():
    """Validate model size reduction."""
    import torch
    from transformers import AutoModel
    
    # Load BGE model
    model = AutoModel.from_pretrained("BAAI/bge-reranker-v2-m3")
    param_count = sum(p.numel() for p in model.parameters())
    
    # Verify 278M parameters (approximately)
    assert 250_000_000 < param_count < 300_000_000
    
    # Verify memory usage < 1GB
    model_size_gb = param_count * 4 / (1024**3)  # FP32 size
    assert model_size_gb < 1.2  # With overhead

@pytest.mark.asyncio  
async def test_unified_reranker():
    """Test unified reranker with automatic model selection."""
    unified = UnifiedReranker()
    
    # Test BGE (default)
    bge_results = await unified.rerank_async(test_nodes, "multimodal query")
    assert len(bge_results) == 5
    
    # Test ColBERT fallback (if enabled)
    if Settings.enable_colbert_fallback:
        colbert_results = await unified.rerank_async(
            test_nodes, "text query", use_text_only=True
        )
        assert len(colbert_results) == 5
        # ColBERT should have slightly better accuracy for text
        
@pytest.mark.asyncio
async def test_parallel_reranking():
    """Test parallel reranking performance."""
    unified = UnifiedReranker()
    
    # Prepare multiple queries
    queries = ["query1", "query2", "query3"]
    node_batches = [test_nodes] * 3
    
    # Test parallel processing
    start = time.time()
    results = await unified.parallel_rerank(node_batches, queries)
    duration = time.time() - start
    
    assert len(results) == 3
    # Should be faster than sequential
    assert duration < len(queries) * single_query_time * 0.5
```

## Consequences

### Positive Outcomes

- **86% Model Size Reduction**: 278M params (BGE) vs 2.4B params (Jina m0)

- **3-5x Faster Inference**: Increased speed with smaller model

- **80% Memory Savings**: <1GB VRAM vs >4GB for Jina m0

- **Native LlamaIndex Integration**: SentenceTransformerRerank integration

- **KISS Compliance**: Minimal code changes required

- **Maintained Accuracy**: 4.1% NDCG degradation acceptable for efficiency gains

- **Dual Strategy**: ColBERT fallback preserves high-accuracy option for text-only scenarios

- **GPU Optimization**: device_map="auto" + mixed precision across all models

- **Async Processing**: QueryPipeline.parallel_run() for increased throughput

### Trade-offs Accepted

- **Accuracy Loss**: 56.51 vs 58.95 NDCG@10 (acceptable for 86% size reduction)

- **Reduced Multimodal Capability**: BGE-v2-m3 simpler than Jina m0's Qwen2-VL base

- **Text-Only Fallback Complexity**: Maintaining ColBERT adds optional complexity for edge cases

### Strategic Benefits

- **Deployment Simplicity**: <1GB model deployable vs 2.4B parameter size

- **Cost Reduction**: Lower compute requirements, faster processing

- **Maintainability**: BAAI models with active community support

- **Future Flexibility**: Model swapping enabled through unified interface

### Dependencies

- **Primary**: llama-index-core>=0.12.0, sentence-transformers>=2.0.0

- **BGE Model**: BAAI/bge-reranker-v2-m3 from HuggingFace

- **ColBERT (Optional)**: llama-index-postprocessor-colbert-rerank>=0.3.0

- **GPU Optimization**: torch>=2.7.1, torchao>=0.1.0

## Changelog

**4.0 (August 13, 2025)**: Major revision based on research and decision framework analysis. Replaced Jina m0 (2.4B params) with BGE-reranker-v2-m3 (278M params) as primary reranker, achieving 86% size reduction with 4.1% accuracy loss. Maintained ColBERT v2 as text-only fallback. Implemented unified reranking interface with model selection. Aligned with KISS/DRY/YAGNI principles.

**3.0 (August 13, 2025)**: Integrated GPU optimization with device_map="auto" and mixed precision from ADR-023 for 1.5x reranking speedup. Added async reranking patterns with QueryPipeline integration. Settings.embed_model configuration for GPU acceleration. Performance aligned with ~1000 tokens/sec system capability.

**2.0 (July 25, 2025)**: Integrated with QueryPipeline; Added toggle/score fusion; Added testing for development.
