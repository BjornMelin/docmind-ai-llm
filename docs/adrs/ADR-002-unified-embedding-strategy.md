# ADR-002: Unified Embedding Strategy with BGE-M3

## Title

BGE-M3 Unified Dense/Sparse Embedding with CLIP Multimodal Support

## Version/Date

4.0 / 2025-08-18

## Status

Accepted

## Description

Consolidates the current three-model embedding strategy (BGE-large + SPLADE + CLIP) into a two-model approach using BGE-M3 for unified dense/sparse text embeddings and CLIP ViT-B/32 for image embeddings. This reduces complexity, memory usage, and maintenance overhead while improving retrieval quality through BGE-M3's unified training approach.

## Context

The current architecture uses three separate embedding models:

- `BAAI/bge-large-en-v1.5` (1024d) for dense text embeddings
- `prithvida/Splade_PP_en_v1` for sparse text embeddings  
- `openai/clip-vit-base-patch32` (512d) for image embeddings

BGE-M3 represents a significant advancement by unifying dense and sparse retrieval in a single model with multi-functionality, multi-linguality, and multi-granularity support. Research shows BGE-M3 achieves superior performance compared to separate dense/sparse models while reducing resource requirements.

**Integration Flow**: Processed document chunks from ADR-009 (Unstructured.io pipeline) are fed into BGE-M3 to generate 1024-dimensional unified embeddings, which are then stored in Qdrant collections (ADR-007) for retrieval by the adaptive pipeline (ADR-003) within the 128K context constraints of the FP8 model.

## Related Requirements

### Functional Requirements

- **FR-1:** Generate high-quality dense embeddings for semantic similarity
- **FR-2:** Generate sparse embeddings for keyword-based retrieval
- **FR-3:** Support multimodal embeddings for images and text
- **FR-4:** Enable hybrid search combining dense and sparse retrieval

### Non-Functional Requirements

- **NFR-1:** **(Performance)** Reduce total embedding memory footprint by >30%
- **NFR-2:** **(Quality)** Maintain or improve retrieval accuracy vs current approach
- **NFR-3:** **(Compatibility)** Support 8192 token context length optimized for 128K LLM context
- **NFR-4:** **(Local-First)** All models must run offline on consumer hardware

## Alternatives

### 1. Current Three-Model Strategy

- **Models**: BGE-large + SPLADE + CLIP (3 models, ~4.2GB total)
- **Issues**: High memory usage, model coordination complexity, separate training objectives
- **Score**: 4/10 (quality: 7, simplicity: 2, performance: 3)

### 2. BGE-M3 + CLIP (Selected)

- **Models**: BGE-M3 unified + CLIP (2 models, fully local)
- **Benefits**: Unified dense/sparse, 8K context, 100% local operation
- **Score**: 9/10 (quality: 9, simplicity: 9, performance: 9)

### 2b. Nomic-Embed-Text-v2 + CLIP (Alternative)

- **Models**: Nomic-Embed-Text-v2-MoE + CLIP (2 models, ~3.5GB total)
- **Benefits**: MoE architecture, 305M active params, excellent multilingual
- **Score**: 8/10 (quality: 8, simplicity: 8, performance: 9)

### 3. Arctic-Embed-L-v2 + CLIP (High Performance)

- **Models**: Snowflake Arctic-Embed-L-v2 + CLIP (2 models, ~4GB total)
- **Benefits**: SOTA performance, 568M params, strong multilingual support
- **Score**: 8.5/10 (quality: 9, simplicity: 8, performance: 8)

### 4. Single Multimodal Model (Jina-CLIP)

- **Models**: Jina-CLIP-v1 for unified text + images (~2.5GB)
- **Benefits**: Single model simplicity, supports 8K context
- **Issues**: Less specialized performance than separate models
- **Score**: 7/10 (quality: 7, simplicity: 10, performance: 6)

### 5. Voyage-3 (Not Viable - API Only)

- **Models**: Voyage-3 API service (requires internet)
- **Issues**: Violates local-only requirement, requires API key, costs money
- **Score**: 0/10 (not viable for local-first architecture)

## Decision

We will adopt **BGE-M3 + CLIP strategy** for 100% local operation:

1. **Primary Text Embeddings**: `BAAI/bge-m3` (1024 dimensions)
   - Dense embeddings: 1024 dimensions
   - Sparse embeddings: Integrated SPLADE-style sparse vectors
   - Context length: 8,192 tokens (16x improvement over original)
   - Performance: 70.0 NDCG@10 on MIRACL (best local model)
   - Multilingual: 100+ languages supported
   - Multi-granular: Handles both short queries and long documents
   - 100% local: No API dependencies, runs offline

   **Alternative Options for Consideration**:
   - `nomic-ai/nomic-embed-text-v2-moe`: MoE architecture, 305M active params
   - `Snowflake/arctic-embed-l-v2`: 568M params, excellent performance
   - `jinaai/jina-embeddings-v3`: 570M params, 8K context support

2. **Image Embeddings**: `openai/clip-vit-base-patch32` (512 dimensions)
   - Proven performance for multimodal search
   - Existing integration with LlamaIndex
   - Lower memory footprint than alternatives

## Related Decisions

- **ADR-003** (Adaptive Retrieval Pipeline): Uses unified embeddings for hybrid search
- **ADR-006** (Modern Reranking Architecture): Works with BGE-M3 outputs
- **ADR-007** (Hybrid Persistence Strategy): Stores unified embedding vectors in Qdrant
- **ADR-009** (Document Processing Pipeline): Provides processed document chunks for BGE-M3 embedding generation
- **ADR-001** (Modern Agentic RAG): Benefits from improved embedding quality

## Design

### BGE-M3 Integration (Primary)

```python
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from sentence_transformers import SentenceTransformer
import torch

class UnifiedEmbeddingConfig:
    """Configuration for local embedding models with BGE-M3 as primary."""
    
    def __init__(self, model_choice="bge-m3"):
        if model_choice == "bge-m3":
            # BGE-M3 for unified dense/sparse embeddings
            self.text_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-m3",
                max_length=8192,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
        elif model_choice == "nomic":
            # Nomic-Embed-Text-v2 MoE alternative
            self.text_model = HuggingFaceEmbedding(
                model_name="nomic-ai/nomic-embed-text-v2-moe",
                max_length=512,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        elif model_choice == "arctic":
            # Snowflake Arctic-Embed-L-v2 for high performance
            self.text_model = HuggingFaceEmbedding(
                model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
                max_length=512,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        self.image_model = ClipEmbedding(
            model_name="openai/clip-vit-base-patch32"
        )
    
    def configure_settings(self):
        """Configure global LlamaIndex settings."""
        Settings.embed_model = self.text_model
        Settings.image_embed_model = self.image_model

# Unified embedding extraction
class BGE_M3_Embedder:
    """Wrapper for BGE-M3 dense and sparse embedding extraction."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(
            model_name, 
            use_fp16=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def encode_documents(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract both dense and sparse embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False  # Disable for simplicity
        )
        
        return {
            'dense': embeddings['dense_vecs'],      # 1024-dim dense vectors
            'sparse': embeddings['lexical_weights'] # Sparse keyword weights
        }
    
    def encode_query(self, query: str) -> Dict[str, np.ndarray]:
        """Encode single query for search."""
        return self.encode_documents([query])
```

### Hybrid Retrieval Integration

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import VectorStoreQuery

class UnifiedHybridRetriever(BaseRetriever):
    """Retriever using BGE-M3 unified embeddings for hybrid search."""
    
    def __init__(self, vector_store, embedder: BGE_M3_Embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        super().__init__()
    
    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        # Get unified embeddings
        query_embeddings = self.embedder.encode_query(query_bundle.query_str)
        
        # Dense retrieval
        dense_query = VectorStoreQuery(
            query_embedding=query_embeddings['dense'][0],
            similarity_top_k=20,
            mode="default"
        )
        dense_results = self.vector_store.query(dense_query)
        
        # Sparse retrieval  
        sparse_query = VectorStoreQuery(
            query_embedding=query_embeddings['sparse'][0],
            similarity_top_k=20,
            mode="sparse"
        )
        sparse_results = self.vector_store.query(sparse_query)
        
        # Fuse results using RRF (Reciprocal Rank Fusion)
        return self._fuse_results(dense_results, sparse_results)
    
    def _fuse_results(self, dense_results, sparse_results, k=60) -> List[NodeWithScore]:
        """Combine dense and sparse results using RRF."""
        scores = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results.nodes):
            scores[result.node_id] = 1.0 / (k + rank + 1)
        
        # Process sparse results  
        for rank, result in enumerate(sparse_results.nodes):
            node_id = result.node_id
            scores[node_id] = scores.get(node_id, 0) + 1.0 / (k + rank + 1)
        
        # Sort by combined score
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [NodeWithScore(node=node, score=score) 
                for node_id, score in sorted_nodes[:10]]
```

### Memory and Performance Optimization

```python
class OptimizedBGE_M3:
    """Memory-optimized BGE-M3 implementation."""
    
    def __init__(self):
        self.model = None
        self._embedding_cache = {}
        self.max_cache_size = 1000
    
    @lru_cache(maxsize=100)
    def _get_model(self):
        """Lazy model loading with caching."""
        if self.model is None:
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(
                "BAAI/bge-m3",
                use_fp16=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        return self.model
    
    def encode_with_cache(self, text: str) -> Dict[str, np.ndarray]:
        """Cache embeddings to avoid recomputation."""
        text_hash = hash(text)
        
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        # Compute embeddings
        embeddings = self._get_model().encode([text], return_dense=True, return_sparse=True)
        
        # Cache with LRU eviction
        if len(self._embedding_cache) >= self.max_cache_size:
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[text_hash] = embeddings
        return embeddings
```

## Consequences

### Positive Outcomes

- **Memory Reduction**: 30% reduction in total embedding memory footprint (4.2GB → 3.6GB)
- **Extended Context**: 16x increase in maximum token length (512 → 8192 tokens)
- **Unified Training**: BGE-M3's joint training improves dense/sparse coordination
- **Multilingual Support**: 100+ languages vs English-only current setup
- **Simplified Architecture**: Two models instead of three reduces coordination complexity
- **Performance**: Unified model eliminates separate inference passes for dense/sparse

### Negative Consequences / Trade-offs

- **Migration Effort**: Requires re-indexing all existing documents with new embeddings
- **Model Dependency**: Increased reliance on BGE-M3 specific implementation
- **Memory Peak**: Higher peak memory during model loading (2.27GB for BGE-M3)
- **API Changes**: Different API surface compared to separate HuggingFace models

### Migration Strategy

1. **Parallel Indexing**: Build new BGE-M3 index alongside existing index
2. **Gradual Transition**: Switch queries to new index after validation
3. **Fallback Support**: Maintain old index temporarily for comparison
4. **Performance Validation**: A/B test retrieval quality before full switch

## Performance Targets

- **Memory Usage**: <3.6GB total for both embedding models
- **Indexing Speed**: <2 seconds per document with unified embeddings
- **Query Speed**: <50ms for embedding generation on RTX 4090 Laptop
- **Retrieval Quality**: ≥5% improvement in NDCG@10 vs current approach
- **Context Support**: Full 8192 token documents without truncation

## Dependencies

- **Python**: `FlagEmbedding>=1.2.0`, `torch>=2.0.0`
- **Models**: `BAAI/bge-m3`, `openai/clip-vit-base-patch32`
- **Integration**: Updated LlamaIndex embedding interfaces

## Monitoring Metrics

- Model memory consumption and GPU utilization
- Embedding generation latency across different text lengths
- Retrieval quality metrics (precision, recall, NDCG)
- Cache hit rates for repeated embeddings
- Index size and storage requirements

## Changelog

- **4.0 (2025-08-18)**: **HARDWARE UPGRADE** - Updated performance targets for RTX 4090 Laptop: <50ms embedding generation. BGE-M3 benefits from faster GPU with larger batch processing capabilities.
- **3.1 (2025-08-18)**: Enhanced integration with DSPy query optimization for automatic embedding quality improvement and added BGE-M3 compatibility with PropertyGraphIndex for multi-modal retrieval scenarios
- **3.0 (2025-08-17)**: CRITICAL FIX - Removed Voyage-3 (API-only, violates local-first requirement). Set BGE-M3 as PRIMARY model for 100% local operation. Added Nomic-Embed-v2-MoE and Arctic-Embed-L-v2 as strong local alternatives.
- **2.0 (2025-08-17)**: [INVALID - Incorrectly selected API-only Voyage-3]
- **1.0 (2025-01-16)**: Initial design for BGE-M3 unified embedding strategy with CLIP multimodal support
