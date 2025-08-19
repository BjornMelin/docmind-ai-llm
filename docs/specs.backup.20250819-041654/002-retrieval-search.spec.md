# Feature Specification: Retrieval & Search System

## Metadata

- **Feature ID**: FEAT-002
- **Version**: 1.0.0
- **Status**: Draft
- **Created**: 2025-08-19
- **Requirements Covered**: REQ-0041 to REQ-0050

## 1. Objective

The Retrieval & Search System implements a sophisticated hybrid search pipeline combining dense semantic vectors, sparse keyword embeddings, and multimodal image embeddings. The system uses Reciprocal Rank Fusion (RRF) to merge results from multiple retrieval strategies, applies reranking for quality optimization, and optionally leverages GraphRAG for relationship-based queries, achieving >80% relevance accuracy with <2 second latency.

## 2. Scope

### In Scope

- Dense embedding generation (BGE-large-en-v1.5)
- Sparse embedding generation (SPLADE++)
- Multimodal embedding generation (CLIP ViT-B/32)
- Hybrid search with RRF fusion
- Document reranking (BGE-reranker-v2-m3)
- Qdrant vector database integration
- Optional GraphRAG PropertyGraphIndex
- Performance optimization and caching

### Out of Scope

- Real-time index updates
- Distributed search across multiple nodes
- Custom embedding model training
- Cross-lingual search capabilities

## 3. Inputs and Outputs

### Inputs

- **Search Query**: User query string (max 512 tokens)
- **Search Strategy**: vector|hybrid|graphrag
- **Top-K Parameter**: Number of results to retrieve (default: 10)
- **Rerank Flag**: Enable/disable reranking (default: true)
- **Filters**: Optional metadata filters (Dict[str, Any])

### Outputs

- **Retrieved Documents**: List of relevant documents with scores
- **Reranked Results**: Documents sorted by relevance
- **Search Metadata**: Query latency, strategy used, tokens processed
- **Source Attribution**: Document paths and metadata

## 4. Interfaces

### Primary Search Interface

```python
class HybridRetriever:
    """Main retrieval interface combining multiple strategies."""
    
    async def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 10,
        rerank: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Execute hybrid retrieval with specified strategy."""
        pass

class RetrievalResult:
    """Results from retrieval pipeline."""
    documents: List[Document]
    scores: List[float]
    strategy_used: str
    latency_ms: float
    tokens_processed: int
```

### Embedding Interfaces

```python
class EmbeddingManager:
    """Manages multiple embedding models."""
    
    def generate_dense_embedding(self, text: str) -> np.ndarray:
        """Generate 1024-dim dense embedding using BGE-large."""
        pass
    
    def generate_sparse_embedding(self, text: str) -> Dict[int, float]:
        """Generate sparse embedding using SPLADE++."""
        pass
    
    def generate_image_embedding(self, image: PIL.Image) -> np.ndarray:
        """Generate 512-dim image embedding using CLIP."""
        pass
```

### Reranking Interface

```python
class Reranker:
    """Document reranking for relevance optimization."""
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10
    ) -> List[Document]:
        """Rerank documents using BGE-reranker-v2-m3."""
        pass
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
    "timestamp": "2025-08-19T10:00:00Z"
  },
  "embeddings": {
    "dense": [0.1, 0.2, ...],  // 1024 dimensions
    "sparse": {12: 0.5, 45: 0.3, ...},  // term indices
    "image": [0.1, 0.2, ...]  // 512 dimensions (optional)
  },
  "score": 0.85
}
```

### RRF Fusion Parameters

```json
{
  "k": 60,
  "weights": {
    "dense": 0.5,
    "sparse": 0.3,
    "rerank": 0.2
  },
  "min_score_threshold": 0.5
}
```

### Qdrant Collection Config

```json
{
  "collection_name": "docmind_hybrid",
  "vectors": {
    "dense": {
      "size": 1024,
      "distance": "Cosine"
    },
    "sparse": {
      "size": null,
      "distance": "Dot"
    },
    "image": {
      "size": 512,
      "distance": "Cosine"
    }
  },
  "payload_schema": {
    "content": "text",
    "metadata": "json"
  }
}
```

## 6. Change Plan

### New Files

- `src/retrieval/hybrid_retriever.py` - Main retrieval orchestrator
- `src/retrieval/embeddings/dense_encoder.py` - BGE dense embeddings
- `src/retrieval/embeddings/sparse_encoder.py` - SPLADE++ sparse embeddings
- `src/retrieval/embeddings/image_encoder.py` - CLIP image embeddings
- `src/retrieval/fusion/rrf.py` - Reciprocal Rank Fusion implementation
- `src/retrieval/reranker.py` - BGE reranker integration
- `src/retrieval/graphrag_integration.py` - Optional GraphRAG support
- `tests/test_retrieval/` - Retrieval test suite

### Modified Files

- `src/vector_store/qdrant_manager.py` - Qdrant collection setup
- `src/config/embedding_config.py` - Model configurations
- `src/agents/retrieval.py` - Agent integration

### Model Downloads

- `BAAI/bge-large-en-v1.5` (~1.34GB)
- `naver/splade-cocondenser-ensembledistil` (~440MB)
- `openai/clip-vit-base-patch32` (~605MB)
- `BAAI/bge-reranker-v2-m3` (~1.12GB)

## 7. Acceptance Criteria

### Scenario 1: Hybrid Search Execution

```gherkin
Given a user query "explain quantum computing applications"
When hybrid search is executed with RRF fusion
Then dense embeddings capture semantic meaning
And sparse embeddings capture key terms like "quantum" and "computing"
And RRF fusion combines both result sets with k=60
And the top 10 documents are returned within 2 seconds
```

### Scenario 2: Document Reranking

```gherkin
Given 20 retrieved documents from initial search
When reranking is applied with BGE-reranker-v2-m3
Then documents are re-scored based on query relevance
And the top 10 reranked documents have higher NDCG scores
And reranking adds less than 200ms latency
```

### Scenario 3: Multimodal Search

```gherkin
Given a document containing both text and images
When the document is indexed
Then text generates dense and sparse embeddings
And images generate CLIP embeddings
And both text and image content are searchable
And cross-modal search returns relevant results
```

### Scenario 4: GraphRAG Integration

```gherkin
Given a relationship query "How are components X and Y connected?"
When GraphRAG is enabled and appropriate for the query
Then the PropertyGraphIndex is queried
And multi-hop relationships are traversed
And entity relationships are included in results
And confidence scores exceed 0.7 for graph results
```

### Scenario 5: Performance Under Load

```gherkin
Given 100 concurrent search requests
When the retrieval system processes all requests
Then 95th percentile latency remains under 2 seconds
And all embedding models stay loaded in memory
And VRAM usage remains stable under 14GB
And retrieval accuracy maintains >80% relevance
```

## 8. Tests

### Unit Tests

- Test each embedding model independently
- Verify RRF fusion algorithm with known inputs
- Test reranker scoring logic
- Validate Qdrant operations (insert, search, delete)
- Test GraphRAG query detection

### Integration Tests

- End-to-end hybrid search pipeline
- Multi-modal document indexing and retrieval
- Reranking impact on result quality
- GraphRAG fallback scenarios
- Cache hit/miss scenarios

### Performance Tests

- Embedding generation speed (target: <100ms per doc)
- Search latency at various scales (10, 100, 1000 docs)
- Reranking overhead measurement
- Memory usage under concurrent load
- VRAM utilization tracking

### Quality Tests

- NDCG@10 on benchmark datasets (target: >0.8)
- MRR (Mean Reciprocal Rank) evaluation
- Precision@K for different K values
- A/B testing dense vs hybrid strategies

## 9. Security Considerations

- Input sanitization for search queries
- Rate limiting for embedding generation
- Secure storage of vector indices
- No exposure of raw embeddings via API
- Query logging with PII redaction

## 10. Quality Gates

### Performance Gates

- P95 query latency: <2 seconds (REQ-0046)
- Embedding generation: <100ms per chunk
- Reranking overhead: <200ms for 20 docs
- Retrieval accuracy: >80% NDCG@10 (REQ-0050)

### Resource Gates

- Dense embedding model: <2GB VRAM
- Sparse embedding model: <1GB VRAM
- Image embedding model: <1.4GB VRAM (REQ-0044)
- Total retrieval VRAM: <5GB combined

### Quality Gates

- Zero embedding failures for valid text
- 100% source attribution for retrieved docs
- RRF fusion preserves top relevant results
- Reranking improves NDCG by >10%

## 11. Requirements Covered

- **REQ-0041**: Hybrid search with dense and sparse vectors ✓
- **REQ-0042**: BGE-large-en-v1.5 dense embeddings ✓
- **REQ-0043**: SPLADE++ sparse embeddings ✓
- **REQ-0044**: CLIP ViT-B/32 image embeddings ✓
- **REQ-0045**: BGE-reranker-v2-m3 reranking ✓
- **REQ-0046**: P95 latency under 2 seconds ✓
- **REQ-0047**: Qdrant vector database ✓
- **REQ-0048**: RRF with k=60 ✓
- **REQ-0049**: Optional GraphRAG support ✓
- **REQ-0050**: >80% retrieval accuracy ✓

## 12. Dependencies

### Technical Dependencies

- `qdrant-client==1.15.0`
- `sentence-transformers>=2.2.0`
- `transformers>=4.30.0`
- `torch>=2.0.0`
- `llama-index-vector-stores-qdrant>=0.1.0`

### Model Dependencies

- BGE-large-en-v1.5 (dense embeddings)
- SPLADE++ (sparse embeddings)
- CLIP ViT-B/32 (image embeddings)
- BGE-reranker-v2-m3 (reranking)

### Feature Dependencies

- Document Processing (FEAT-003) for chunk generation
- Infrastructure (FEAT-004) for GPU acceleration
- Multi-Agent (FEAT-001) for retrieval coordination

## 13. Traceability

### Source Documents

- ADR-002: Unified Embedding Strategy
- ADR-003: Adaptive Retrieval Pipeline
- ADR-006: Reranking Architecture
- ADR-019: Optional GraphRAG
- PRD Section 3: Advanced Hybrid Search Epic

### Related Specifications

- 001-multi-agent-coordination.spec.md
- 003-document-processing.spec.md
- 004-infrastructure-performance.spec.md
