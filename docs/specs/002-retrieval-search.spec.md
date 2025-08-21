# Feature Specification: Retrieval & Search System

## Metadata

- **Feature ID**: FEAT-002
- **Version**: 2.0.0
- **Status**: Updated for ADR Alignment
- **Created**: 2025-08-19
- **Updated**: 2025-08-20
- **Validated At**: 2025-08-20
- **Completion Percentage**: 95%
- **Requirements Covered**: REQ-0041 to REQ-0050
- **ADR Alignment**: Complete (ADR-002, ADR-003, ADR-006, ADR-007, ADR-018, ADR-019)

## 1. Objective

The Retrieval & Search System implements an adaptive, library-first retrieval pipeline using BGE-M3 unified dense/sparse embeddings, multimodal CLIP embeddings, and LlamaIndex native components. The system features automatic query optimization via DSPy, intelligent routing through RouterQueryEngine, optional PropertyGraphIndex for relationship queries, and BGE-reranker-v2-m3 for relevance optimization. Optimized for RTX 4090 Laptop hardware with 128K context support and FP8 acceleration, achieving >80% retrieval accuracy with <2 second P95 latency.

## 2. Scope

### In Scope

- BGE-M3 unified dense/sparse embedding generation (replaces BGE-large + SPLADE++)
- Multimodal embedding generation (CLIP ViT-B/32)
- LlamaIndex RouterQueryEngine for adaptive retrieval strategy selection
- HybridRetriever and MultiQueryRetriever with built-in RRF fusion
- DSPy-based automatic query optimization and rewriting
- Sentence-transformers CrossEncoder reranking (BGE-reranker-v2-m3)
- Qdrant vector database with resilience patterns (SQLite + Qdrant)
- Optional PropertyGraphIndex for graph-based retrieval
- Performance optimization for RTX 4090 Laptop with 128K context and FP8 acceleration

### Out of Scope

- Real-time index updates
- Distributed search across multiple nodes
- Custom embedding model training
- Cross-lingual search capabilities

## 3. Inputs and Outputs

### Inputs

- **Search Query**: User query string (max 8192 tokens with BGE-M3)
- **Search Strategy**: Automatically selected by RouterQueryEngine (vector|hybrid|multi_query|graph)
- **DSPy Optimization**: Enable/disable automatic query rewriting (default: experimental flag)
- **Top-K Parameter**: Number of results to retrieve (default: 10)
- **Rerank Flag**: Enable/disable reranking (default: true)
- **GraphRAG Mode**: Enable PropertyGraphIndex for relationship queries (default: optional flag)
- **Filters**: MetadataFilters using LlamaIndex native filtering

### Outputs

- **Retrieved Documents**: List of relevant documents with scores
- **Reranked Results**: Documents sorted by relevance
- **Search Metadata**: Query latency, strategy used, tokens processed
- **Source Attribution**: Document paths and metadata

## 4. Interfaces

### Primary Search Interface

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.retrievers import HybridRetriever, MultiQueryRetriever
from llama_index.core.tools import QueryEngineTool
from sentence_transformers import CrossEncoder

class AdaptiveRetrievalPipeline:
    """Library-first adaptive retrieval using LlamaIndex components."""
    
    def __init__(
        self,
        vector_store,
        llm,
        enable_dspy: bool = False,
        enable_graphrag: bool = False
    ):
        self.router_engine = self._build_router_engine(vector_store, llm, enable_dspy, enable_graphrag)
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True
    ) -> RetrievalResult:
        """Execute adaptive retrieval with automatic strategy selection."""
        response = await self.router_engine.aquery(query)
        
        if rerank and response.source_nodes:
            response.source_nodes = self._rerank_results(query, response.source_nodes)
        
        return RetrievalResult(
            documents=[node.node for node in response.source_nodes[:top_k]],
            strategy_used=response.metadata.get("selected_tool", "unknown"),
            response=response.response
        )

class RetrievalResult:
    """Results from adaptive retrieval pipeline."""
    documents: List[Document]
    strategy_used: str
    response: str
    latency_ms: Optional[float] = None
```

### Unified Embedding Interface

```python
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from FlagEmbedding import BGEM3FlagModel

class UnifiedEmbeddingManager:
    """BGE-M3 unified embedding with CLIP multimodal support."""
    
    def __init__(self):
        # BGE-M3 for unified dense/sparse text embeddings
        self.text_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            max_length=8192,  # 8K context vs 512 in BGE-large
            device_map="auto",
            torch_dtype="float16"
        )
        
        # CLIP for image embeddings
        self.image_model = ClipEmbedding(
            model_name="openai/clip-vit-base-patch32"
        )
        
        # BGE-M3 direct model for unified extraction
        self.bge_m3_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=True,
            device='cuda'
        )
        
        # Configure LlamaIndex settings
        Settings.embed_model = self.text_model
        Settings.image_embed_model = self.image_model
    
    def generate_unified_embeddings(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Generate both dense and sparse embeddings using BGE-M3."""
        embeddings = self.bge_m3_model.encode(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=True
        )
        return {
            'dense': embeddings['dense_vecs'],      # 1024-dim dense
            'sparse': embeddings['lexical_weights'] # Sparse keyword weights
        }
```

### Simplified Reranking Interface

```python
from sentence_transformers import CrossEncoder

class SimpleReranker:
    """Library-first reranking using sentence-transformers directly."""
    
    def __init__(self):
        # Use CrossEncoder directly - no custom wrappers needed
        self.model = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10
    ) -> List[Document]:
        """Rerank documents using BGE-reranker-v2-m3 CrossEncoder."""
        # Create query-document pairs
        pairs = [(query, doc.content) for doc in documents]
        
        # Get relevance scores - one line!
        scores = self.model.predict(pairs)
        
        # Sort by score and return top-k
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores[:top_k]]
```

## 5. Data Contracts

### Document Schema

```json
{
  "id": "doc_uuid",
  "content": "text content",
  "metadata": {
    "source": "file_path",
    "page": 1,
    "chunk_id": "chunk_001",
    "timestamp": "2025-08-19T10:00:00Z",
    "doc_type": "pdf",
    "section": "introduction"
  },
  "embeddings": {
    "bge_m3_dense": [0.1, 0.2, ...],  // 1024 dimensions (BGE-M3 unified)
    "bge_m3_sparse": {12: 0.5, 45: 0.3, ...},  // BGE-M3 sparse weights
    "image": [0.1, 0.2, ...]  // 512 dimensions (CLIP, optional)
  },
  "score": 0.85,
  "retrieval_strategy": "hybrid",
  "dspy_optimized": true
}
```

### RouterQueryEngine Configuration

```json
{
  "retrieval_strategies": {
    "vector_search": {
      "description": "Best for semantic similarity and simple factual queries",
      "similarity_top_k": 10
    },
    "hybrid_search": {
      "description": "Best for complex queries needing both keywords and semantics",
      "fusion_mode": "reciprocal_rank",
      "vector_top_k": 5,
      "keyword_top_k": 5
    },
    "multi_query": {
      "description": "Best for complex questions that need decomposition",
      "num_queries": 3
    },
    "graph_search": {
      "description": "Best for relationship queries and multi-hop reasoning",
      "enabled": false,
      "path_depth": 2
    }
  },
  "dspy_config": {
    "enabled": false,
    "query_rewriting": true,
    "bootstrap_examples": 10
  }
}
```

### Qdrant Collection Config with Resilience

```json
{
  "collection_name": "docmind_bge_m3",
  "vectors": {
    "bge_m3_dense": {
      "size": 1024,
      "distance": "Cosine"
    },
    "bge_m3_sparse": {
      "modifier": "idf",
      "type": "sparse"
    },
    "clip_image": {
      "size": 512,
      "distance": "Cosine"
    }
  },
  "payload_schema": {
    "content": "text",
    "metadata": "json",
    "doc_type": "keyword",
    "section": "keyword"
  },
  "resilience": {
    "connection_retry": 3,
    "exponential_backoff": true,
    "wal_mode": true
  }
}
```

## 6. Change Plan

### New Files

- `src/retrieval/adaptive_pipeline.py` - RouterQueryEngine-based adaptive retrieval
- `src/retrieval/embeddings/bge_m3_manager.py` - BGE-M3 unified embeddings
- `src/retrieval/embeddings/clip_encoder.py` - CLIP image embeddings
- `src/retrieval/dspy_optimization.py` - DSPy query rewriting and optimization
- `src/retrieval/simple_reranker.py` - sentence-transformers CrossEncoder reranking
- `src/retrieval/graphrag_integration.py` - PropertyGraphIndex integration
- `src/retrieval/persistence.py` - SQLite + Qdrant resilience patterns
- `tests/test_retrieval/` - Updated retrieval test suite
- `config/dspy.yaml` - DSPy configuration and feature flags
- `config/graphrag.yaml` - GraphRAG configuration

### Modified Files

- `src/vector_store/qdrant_manager.py` - BGE-M3 collection setup with resilience
- `src/config/embedding_config.py` - BGE-M3 unified configuration
- `src/config/settings.py` - Feature flags for DSPy and GraphRAG
- `src/agents/retrieval.py` - RouterQueryEngine integration for agents
- `src/pipeline/ingestion.py` - BGE-M3 embedding generation

### Model Downloads

- `BAAI/bge-m3` (~2.27GB) - Replaces BGE-large + SPLADE++
- `openai/clip-vit-base-patch32` (~605MB) - Image embeddings
- `BAAI/bge-reranker-v2-m3` (~1.12GB) - Reranking model
- **Total Storage Reduction**: 4.2GB → 3.6GB (14% reduction)

## 7. Acceptance Criteria

### Scenario 1: Adaptive Strategy Selection

```gherkin
Given a user query "explain quantum computing applications"
When RouterQueryEngine evaluates the query
Then the query is classified as analytical/complex
And multi_query strategy is automatically selected
And BGE-M3 captures both semantic meaning and key terms in unified embeddings
And 3 sub-queries are generated for comprehensive retrieval
And the top 10 documents are returned within 2 seconds
```

### Scenario 2: Simple Reranking with CrossEncoder

```gherkin
Given 20 retrieved documents from RouterQueryEngine
When sentence-transformers CrossEncoder reranking is applied
Then BGE-reranker-v2-m3 re-scores query-document pairs
And the top 10 reranked documents have higher relevance scores
And reranking adds less than 100ms latency on RTX 4090 Laptop
```

### Scenario 3: BGE-M3 Unified Embedding

```gherkin
Given a document containing both text and images
When the document is processed
Then BGE-M3 generates unified dense and sparse embeddings from text
And CLIP generates 512-dim embeddings from images
And both embeddings are stored in Qdrant collections
And 8K token context enables processing of larger document chunks
And cross-modal search returns relevant results
```

### Scenario 4: PropertyGraphIndex Integration

```gherkin
Given a relationship query "How are components X and Y connected?"
When RouterQueryEngine detects graph-appropriate query
Then graph_search tool is automatically selected
And PropertyGraphIndex performs hybrid vector+graph retrieval
And multi-hop relationships are traversed with path_depth=2
And SimplePropertyGraphStore reuses existing Qdrant vector storage
And entity relationships are included in results with source attribution
```

### Scenario 5: DSPy Query Optimization

```gherkin
Given DSPy optimization is enabled
When a complex user query is submitted
Then DSPy QueryRewriter expands the query into 3-5 variants
And key concepts are extracted for refinement
And optimized queries improve retrieval quality by >20%
And query optimization adds <200ms overhead with FP8 acceleration
And the system learns from user feedback over time
```

### Scenario 6: Performance Under Load on RTX 4090 Laptop

```gherkin
Given 100 concurrent search requests on RTX 4090 Laptop
When the adaptive retrieval system processes all requests
Then 95th percentile latency remains under 2 seconds
And BGE-M3 embedding generation is <50ms per chunk
And reranking latency is <100ms for 20 documents
And VRAM usage remains stable under 14GB with FP8 optimization
And retrieval accuracy maintains >80% relevance
```

## 8. Tests

### Unit Tests

- Test BGE-M3 unified embedding generation (dense + sparse)
- Verify RouterQueryEngine strategy selection logic
- Test sentence-transformers CrossEncoder reranking
- Validate Qdrant operations with resilience patterns
- Test DSPy query rewriting and optimization
- Test PropertyGraphIndex integration
- Validate MetadataFilters functionality

### Integration Tests

- End-to-end adaptive retrieval pipeline with RouterQueryEngine
- BGE-M3 unified embedding in LlamaIndex pipeline
- Multi-modal document indexing with BGE-M3 + CLIP
- DSPy optimization impact on retrieval quality
- PropertyGraphIndex query routing scenarios
- SQLite + Qdrant persistence with resilience
- Feature flag testing (DSPy and GraphRAG disabled/enabled)

### Performance Tests

- BGE-M3 embedding generation speed (target: <50ms per chunk on RTX 4090 Laptop)
- RouterQueryEngine latency across different strategies
- Search latency at various scales with BGE-M3 (10, 100, 1000 docs)
- CrossEncoder reranking overhead (target: <100ms for 20 docs)
- DSPy query optimization overhead (target: <200ms)
- Memory usage under concurrent load with FP8 optimization
- VRAM utilization tracking for 14GB constraint

### Quality Tests

- NDCG@10 on benchmark datasets (target: >0.8 with BGE-M3)
- MRR (Mean Reciprocal Rank) evaluation across strategies
- Precision@K for RouterQueryEngine strategy selection
- DSPy optimization quality improvement measurement (target: >20%)
- A/B testing: manual queries vs DSPy-optimized queries
- PropertyGraphIndex effectiveness on relationship queries

## 9. Security Considerations

- Input sanitization for search queries
- Rate limiting for embedding generation
- Secure storage of vector indices
- No exposure of raw embeddings via API
- Query logging with PII redaction

## 10. Quality Gates

### Performance Gates

- P95 query latency: <2 seconds (REQ-0046) on RTX 4090 Laptop
- BGE-M3 embedding generation: <50ms per chunk (improved from 100ms)
- CrossEncoder reranking overhead: <100ms for 20 docs (improved from 200ms)
- DSPy query optimization: <200ms overhead with FP8 acceleration
- RouterQueryEngine strategy selection: <50ms
- Retrieval accuracy: >80% NDCG@10 (REQ-0050)

### Resource Gates

- BGE-M3 unified model: <3GB VRAM (replaces dense + sparse models)
- CLIP image embedding model: <1.4GB VRAM (REQ-0044)
- BGE-reranker-v2-m3: <1.2GB VRAM
- Total retrieval VRAM: <6GB combined (optimized for RTX 4090 Laptop)
- DSPy optimization models: <500MB additional when active
- PropertyGraphIndex: <500MB additional when enabled

### Quality Gates

- Zero BGE-M3 embedding failures for valid text (up to 8K tokens)
- 100% source attribution for retrieved documents
- RouterQueryEngine selects appropriate strategy >90% accuracy
- DSPy optimization improves retrieval metrics by >20%
- CrossEncoder reranking improves NDCG by >10%
- PropertyGraphIndex provides relevant relationships when enabled

## 11. Requirements Covered

- **REQ-0041**: Adaptive hybrid search with BGE-M3 unified dense/sparse vectors ✓
- **REQ-0042**: BGE-M3 unified embeddings (replaces BGE-large-en-v1.5) ✓
- **REQ-0043**: BGE-M3 sparse embeddings (replaces SPLADE++) ✓
- **REQ-0044**: CLIP ViT-B/32 image embeddings with 1.4GB VRAM constraint ✓
- **REQ-0045**: BGE-reranker-v2-m3 via sentence-transformers CrossEncoder ✓
- **REQ-0046**: P95 latency under 2 seconds on RTX 4090 Laptop ✓
- **REQ-0047**: Qdrant vector database with resilience patterns ✓
- **REQ-0048**: Native LlamaIndex RRF fusion via HybridRetriever ✓
- **REQ-0049**: PropertyGraphIndex for optional graph-based retrieval ✓
- **REQ-0050**: >80% retrieval accuracy with DSPy optimization ✓

## 12. Dependencies

### Technical Dependencies

- `qdrant-client>=1.15.0` (vector storage with resilience)
- `sentence-transformers>=2.2.0` (CrossEncoder reranking)
- `FlagEmbedding>=1.2.0` (BGE-M3 unified embeddings)
- `llama-index-core>=0.12.0` (RouterQueryEngine, HybridRetriever)
- `llama-index-vector-stores-qdrant>=0.1.0`
- `dspy-ai>=2.4.0` (optional, for query optimization)
- `torch>=2.7.1` (FP8 support)
- `tenacity>=9.1.2` (resilience patterns)

### Model Dependencies

- BAAI/bge-m3 (unified dense/sparse embeddings, replaces BGE-large + SPLADE++)
- openai/clip-vit-base-patch32 (image embeddings)
- BAAI/bge-reranker-v2-m3 (reranking)
- Qwen3-4B-Instruct-2507-FP8 (for DSPy optimization, optional)

### Feature Dependencies

- Document Processing (FEAT-003) for BGE-M3 compatible chunk generation
- Infrastructure (FEAT-004) for RTX 4090 Laptop GPU acceleration and FP8 optimization
- Multi-Agent (FEAT-001) for RouterQueryEngine coordination and DSPy integration
- Optional: PropertyGraphIndex for graph-based retrieval (feature flag)
- Optional: DSPy optimization for automatic query rewriting (experimental flag)

## 13. Traceability

### Source Documents

- ADR-002: Unified Embedding Strategy with BGE-M3 (v4.0)
- ADR-003: Adaptive Retrieval Pipeline with LlamaIndex RouterQueryEngine (v3.0)
- ADR-006: Modern Reranking Architecture with sentence-transformers (v3.1)
- ADR-007: Hybrid Persistence Strategy with SQLite + Qdrant resilience (v2.2)
- ADR-018: Automatic Prompt Optimization with DSPy (v2.0)
- ADR-019: Optional GraphRAG with PropertyGraphIndex (v3.0)
- PRD Section 3: Advanced Hybrid Search Epic

### Related Specifications

- 001-multi-agent-coordination.spec.md (RouterQueryEngine integration)
- 003-document-processing.spec.md (BGE-M3 compatible chunking)
- 004-infrastructure-performance.spec.md (RTX 4090 Laptop, FP8 optimization)

### Architecture Integration

- **Embedding Strategy**: BGE-M3 unifies dense/sparse, reduces model count from 3 to 2
- **Retrieval Pipeline**: RouterQueryEngine automatically selects optimal strategy
- **Query Optimization**: DSPy provides automatic query rewriting and expansion
- **Graph Integration**: PropertyGraphIndex reuses Qdrant infrastructure
- **Persistence**: SQLite + Qdrant with Tenacity resilience patterns
- **Performance**: Optimized for RTX 4090 Laptop with 128K context and FP8 acceleration
