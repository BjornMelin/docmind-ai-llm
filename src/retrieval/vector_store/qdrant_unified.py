"""Unified Qdrant vector store for BGE-M3 dense + sparse embeddings.

This module implements a unified Qdrant vector store that supports both
dense (1024D) and sparse embeddings from BGE-M3 in a single collection,
with hybrid search and RRF score fusion capabilities per ADR-007.

Key features:
- Single collection supporting dense + sparse vectors
- Native sparse vector support with payload-based storage
- Hybrid search with configurable RRF fusion (α=0.7)
- Resilience patterns with tenacity retry logic
- Optimized for RTX 4090 performance
"""

from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        NamedSparseVector,
        PointStruct,
        SparseVector,
        SparseVectorParams,
        VectorParams,
    )
except ImportError:
    logger.error("qdrant-client not available. Install with: uv add qdrant-client")
    QdrantClient = None

from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from src.config.settings import settings

# Qdrant configuration constants
MAX_FUSION_LIMIT = 50
DEFAULT_COLLECTION_SIZE = 1000
# Retry configuration constants
RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 10
INIT_RETRY_MIN = 4
INIT_RETRY_MAX = 10
ADD_RETRY_MIN = 2
ADD_RETRY_MAX = 8
QUERY_RETRY_MAX = 4


class QdrantUnifiedVectorStore(BasePydanticVectorStore):
    """Unified Qdrant vector store for BGE-M3 dense + sparse embeddings.

    Supports both dense (1024D) and sparse embeddings in a single collection
    with hybrid search and RRF score fusion capabilities per ADR-007.

    Features:
    - Dense vector search (BGE-M3 1024D embeddings)
    - Sparse vector search (BGE-M3 sparse embeddings)
    - Hybrid search with configurable RRF fusion (α=0.7)
    - Metadata filtering and payload storage
    - Resilience patterns with exponential backoff
    - Optimized for RTX 4090 performance

    Performance targets (RTX 4090 Laptop):
    - <100ms query latency for 1K documents
    - <500ms query latency for 10K documents
    - Support for concurrent read/write operations
    """

    stores_text: bool = True
    is_embedding_query: bool = False

    qdrant_client: QdrantClient = Field(exclude=True)
    collection_name: str
    dense_vector_name: str = Field(default="dense")
    sparse_vector_name: str = Field(default="sparse")
    embedding_dim: int = Field(
        default=settings.bge_m3_embedding_dim
    )  # BGE-M3 dimension
    rrf_alpha: float = Field(
        default=settings.rrf_fusion_alpha
    )  # Dense/sparse fusion weight

    def __init__(
        self,
        *,
        client: QdrantClient | None = None,
        url: str = "http://localhost:6333",
        collection_name: str = "docmind_feat002_unified",
        embedding_dim: int = settings.bge_m3_embedding_dim,
        rrf_alpha: float = settings.rrf_fusion_alpha,
        **kwargs,
    ):
        """Initialize QdrantUnifiedVectorStore.

        Args:
            client: Optional existing QdrantClient instance
            url: Qdrant server URL
            collection_name: Name of the unified collection
            embedding_dim: BGE-M3 dense embedding dimension
            rrf_alpha: RRF fusion weight for dense/sparse (0.7 = 70% dense)
            **kwargs: Additional VectorStore arguments
        """
        if QdrantClient is None:
            raise ImportError(
                "qdrant-client not available. Install with: uv add qdrant-client"
            )

        self.qdrant_client = client or QdrantClient(url=url)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.dense_vector_name = "dense"
        self.sparse_vector_name = "sparse"
        self.rrf_alpha = rrf_alpha

        super().__init__(**kwargs)

        # Initialize unified collection with resilience
        self._init_collection_with_retry()

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=INIT_RETRY_MIN, max=INIT_RETRY_MAX),
        reraise=True,
    )
    def _init_collection_with_retry(self) -> None:
        """Initialize Qdrant collection with resilience patterns."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with dense + sparse vectors
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.dense_vector_name: VectorParams(
                            size=self.embedding_dim, distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        self.sparse_vector_name: SparseVectorParams()
                    },
                )
                logger.info("Created unified collection: %s", self.collection_name)
            else:
                logger.info("Using existing collection: %s", self.collection_name)

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error("Failed to initialize Qdrant collection: %s", e)
            raise

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=ADD_RETRY_MIN, max=ADD_RETRY_MAX),
        reraise=True,
    )
    def add(
        self,
        nodes: list[BaseNode],
        dense_embeddings: list[list[float]] | None = None,
        sparse_embeddings: list[dict[int, float]] | None = None,
        **_kwargs,
    ) -> list[str]:
        """Add nodes with unified dense + sparse embeddings.

        Args:
            nodes: List of nodes to add
            dense_embeddings: BGE-M3 dense embeddings (1024D)
            sparse_embeddings: BGE-M3 sparse embeddings (token weights)
            **kwargs: Additional arguments

        Returns:
            List of node IDs that were added
        """
        if not nodes:
            return []

        try:
            points = []
            node_ids = []

            for i, node in enumerate(nodes):
                node_id = node.node_id
                node_ids.append(node_id)

                # Prepare vectors dictionary
                vectors = {}

                # Dense vector (BGE-M3 1024D)
                if dense_embeddings and i < len(dense_embeddings):
                    vectors[self.dense_vector_name] = dense_embeddings[i]

                # Sparse vector (BGE-M3 token weights)
                named_sparse_vectors = {}
                if sparse_embeddings and i < len(sparse_embeddings):
                    sparse_dict = sparse_embeddings[i]
                    if sparse_dict:
                        indices = list(sparse_dict.keys())
                        values = list(sparse_dict.values())
                        named_sparse_vectors[self.sparse_vector_name] = SparseVector(
                            indices=indices, values=values
                        )

                # Prepare payload with node data and metadata
                payload = {
                    "text": node.get_content(),
                    "metadata": node.metadata or {},
                    "node_id": node_id,
                    "doc_id": getattr(node, "doc_id", ""),
                    "chunk_id": getattr(node, "chunk_id", ""),
                }

                points.append(
                    PointStruct(
                        id=node_id,
                        vector=vectors,
                        sparse_vector=named_sparse_vectors,
                        payload=payload,
                    )
                )

            # Batch upsert points with resilience
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )

            logger.info("Added %d nodes to unified collection", len(points))
            return node_ids

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error("Failed to add nodes to Qdrant: %s", e)
            raise

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=QUERY_RETRY_MAX),
        reraise=True,
    )
    def query(
        self,
        query: VectorStoreQuery,
        dense_embedding: list[float] | None = None,
        sparse_embedding: dict[int, float] | None = None,
        **kwargs,
    ) -> VectorStoreQueryResult:
        """Execute unified query with dense + sparse search.

        Automatically determines search strategy and applies RRF fusion
        for hybrid queries.

        Args:
            query: VectorStoreQuery with similarity parameters
            dense_embedding: BGE-M3 dense query embedding
            sparse_embedding: BGE-M3 sparse query embedding
            **kwargs: Additional query arguments

        Returns:
            VectorStoreQueryResult with reranked results
        """
        try:
            # Determine search strategy
            use_dense = dense_embedding is not None
            use_sparse = sparse_embedding is not None and sparse_embedding

            if use_dense and use_sparse:
                # Hybrid search with RRF fusion
                return self._hybrid_search(
                    query, dense_embedding, sparse_embedding, **kwargs
                )
            if use_dense:
                # Dense-only search
                return self._dense_search(query, dense_embedding, **kwargs)
            if use_sparse:
                # Sparse-only search
                return self._sparse_search(query, sparse_embedding, **kwargs)

            logger.warning("No embeddings provided for query")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
            logger.error("Unified query failed: %s", e)
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

    def _hybrid_search(
        self,
        query: VectorStoreQuery,
        dense_embedding: list[float],
        sparse_embedding: dict[int, float],
        **_kwargs,
    ) -> VectorStoreQueryResult:
        """Execute hybrid search with RRF score fusion.

        Performs both dense and sparse searches, then applies Reciprocal
        Rank Fusion (RRF) to combine results optimally.

        Args:
            query: VectorStoreQuery with parameters
            dense_embedding: BGE-M3 dense query embedding
            sparse_embedding: BGE-M3 sparse query embedding
            **kwargs: Additional arguments

        Returns:
            VectorStoreQueryResult with RRF-fused results
        """
        # Get more results for fusion (2x the requested amount)
        fusion_limit = min(query.similarity_top_k * 2, MAX_FUSION_LIMIT)

        # Execute dense search
        dense_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=(self.dense_vector_name, dense_embedding),
            limit=fusion_limit,
            with_payload=True,
            with_vectors=False,
        )

        # Execute sparse search
        sparse_indices = list(sparse_embedding.keys())
        sparse_values = list(sparse_embedding.values())
        sparse_vector = SparseVector(indices=sparse_indices, values=sparse_values)

        sparse_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=NamedSparseVector(
                name=self.sparse_vector_name, vector=sparse_vector
            ),
            limit=fusion_limit,
            with_payload=True,
            with_vectors=False,
        )

        # Apply RRF (Reciprocal Rank Fusion)
        fused_results = self._apply_rrf_fusion(dense_results, sparse_results)

        # Convert to VectorStoreQueryResult
        return self._convert_results(fused_results[: query.similarity_top_k])

    def _apply_rrf_fusion(
        self,
        dense_results: list[Any],
        sparse_results: list[Any],
        k: int = settings.rrf_k_constant,
    ) -> list[Any]:
        """Apply Reciprocal Rank Fusion to combine dense and sparse results.

        Uses RRF with configurable alpha weighting for dense/sparse balance.

        Args:
            dense_results: Results from dense vector search
            sparse_results: Results from sparse vector search
            k: RRF constant (default: 60)

        Returns:
            List of results ordered by fused relevance score
        """
        # Create score maps for RRF calculation
        dense_scores = {}
        sparse_scores = {}

        # Calculate RRF scores for dense results
        for rank, result in enumerate(dense_results, 1):
            dense_scores[result.id] = 1.0 / (k + rank)

        # Calculate RRF scores for sparse results
        for rank, result in enumerate(sparse_results, 1):
            sparse_scores[result.id] = 1.0 / (k + rank)

        # Combine scores with alpha weighting
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        fused_scores = {}

        for doc_id in all_ids:
            dense_score = dense_scores.get(doc_id, 0.0)
            sparse_score = sparse_scores.get(doc_id, 0.0)
            # Apply alpha weighting: α * dense + (1-α) * sparse
            fused_scores[doc_id] = (
                self.rrf_alpha * dense_score + (1 - self.rrf_alpha) * sparse_score
            )

        # Create result mapping and sort by fused score
        result_map = {}
        for result in dense_results + sparse_results:
            if result.id not in result_map:  # Avoid duplicates
                result_map[result.id] = result

        # Sort by fused score (descending) and update result scores
        sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        fused_results = []

        for doc_id, score in sorted_ids:
            if doc_id in result_map:
                result = result_map[doc_id]
                result.score = score  # Update with fused score
                fused_results.append(result)

        return fused_results

    def _dense_search(
        self, query: VectorStoreQuery, embedding: list[float], **_kwargs
    ) -> VectorStoreQueryResult:
        """Execute dense-only search using BGE-M3 dense embeddings."""
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=(self.dense_vector_name, embedding),
            limit=query.similarity_top_k,
            with_payload=True,
            with_vectors=False,
        )

        return self._convert_results(results)

    def _sparse_search(
        self, query: VectorStoreQuery, sparse_embedding: dict[int, float], **_kwargs
    ) -> VectorStoreQueryResult:
        """Execute sparse-only search using BGE-M3 sparse embeddings."""
        indices = list(sparse_embedding.keys())
        values = list(sparse_embedding.values())
        sparse_vector = SparseVector(indices=indices, values=values)

        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=NamedSparseVector(
                name=self.sparse_vector_name, vector=sparse_vector
            ),
            limit=query.similarity_top_k,
            with_payload=True,
            with_vectors=False,
        )

        return self._convert_results(results)

    def _convert_results(self, results: list[Any]) -> VectorStoreQueryResult:
        """Convert Qdrant results to VectorStoreQueryResult."""
        nodes = []
        similarities = []
        ids = []

        for result in results:
            node = TextNode(
                text=result.payload.get("text", ""),
                metadata=result.payload.get("metadata", {}),
                id_=result.payload.get("node_id"),
            )
            nodes.append(node)
            similarities.append(result.score)
            ids.append(str(result.id))

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def delete(self, ref_doc_id: str, **_delete_kwargs: Any) -> None:
        """Delete documents by reference document ID."""
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=ref_doc_id))
                    ]
                ),
            )
            logger.info("Deleted documents with doc_id: %s", ref_doc_id)
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error("Failed to delete documents: %s", e)
            raise

    def clear(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
            self._init_collection_with_retry()  # Recreate empty collection
            logger.info("Cleared collection: %s", self.collection_name)
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error("Failed to clear collection: %s", e)
            raise


def create_unified_qdrant_store(
    url: str = "http://localhost:6333",
    collection_name: str = "docmind_feat002_unified",
    embedding_dim: int = settings.bge_m3_embedding_dim,
    rrf_alpha: float = settings.rrf_fusion_alpha,
) -> QdrantUnifiedVectorStore:
    """Create unified Qdrant vector store for BGE-M3 embeddings.

    Factory function for easy instantiation with BGE-M3 optimized settings.

    Args:
        url: Qdrant server URL
        collection_name: Name of the unified collection
        embedding_dim: BGE-M3 dense embedding dimension
        rrf_alpha: RRF fusion weight (0.7 = 70% dense, 30% sparse)

    Returns:
        Configured QdrantUnifiedVectorStore for FEAT-002
    """
    return QdrantUnifiedVectorStore(
        url=url,
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        rrf_alpha=rrf_alpha,
    )
