"""Comprehensive unit tests for Qdrant unified vector store module.

Tests focus on hybrid search capabilities, RRF fusion, dense + sparse vector operations,
and error handling while mocking Qdrant client operations for fast execution.

Key testing areas:
- QdrantUnifiedVectorStore initialization and collection management
- Dense + sparse vector storage and retrieval
- Hybrid search with RRF score fusion
- Batch operations and performance optimization
- Error handling and resilience patterns
- Configuration validation and consistency
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient for testing."""
    client = Mock()

    # Mock collection management
    collections_response = Mock()
    collections_response.collections = []
    client.get_collections.return_value = collections_response
    client.create_collection.return_value = None
    client.delete_collection.return_value = None

    # Mock point operations
    client.upsert.return_value = None
    client.delete.return_value = None

    # Mock search results
    def create_mock_search_result(doc_id, score, text):
        result = Mock()
        result.id = doc_id
        result.score = score
        result.payload = {
            "text": text,
            "metadata": {"source": "test"},
            "node_id": doc_id,
            "doc_id": f"doc_{doc_id}",
            "chunk_id": f"chunk_{doc_id}",
        }
        return result

    # Mock dense search results
    client.search.return_value = [
        create_mock_search_result("1", 0.95, "First search result text"),
        create_mock_search_result("2", 0.87, "Second search result text"),
        create_mock_search_result("3", 0.82, "Third search result text"),
    ]

    return client


@pytest.fixture
def sample_nodes():
    """Sample TextNode objects for testing."""
    return [
        TextNode(
            text="Machine learning is transforming document processing capabilities.",
            metadata={"source": "doc1.pdf", "page": 1},
            id_="node_1",
        ),
        TextNode(
            text="Vector databases enable efficient semantic similarity search.",
            metadata={"source": "doc2.pdf", "page": 1},
            id_="node_2",
        ),
        TextNode(
            text="BGE-M3 provides unified dense and sparse embeddings for better retrieval.",
            metadata={"source": "doc3.pdf", "page": 1},
            id_="node_3",
        ),
    ]


@pytest.fixture
def sample_dense_embeddings():
    """Sample 1024-dimensional dense embeddings."""
    return [
        np.random.randn(1024).tolist(),
        np.random.randn(1024).tolist(),
        np.random.randn(1024).tolist(),
    ]


@pytest.fixture
def sample_sparse_embeddings():
    """Sample sparse embeddings (token ID -> weight mappings)."""
    return [
        {100: 0.8, 205: 0.6, 1024: 0.5, 2048: 0.4},
        {150: 0.9, 300: 0.7, 1100: 0.6, 2200: 0.3},
        {80: 0.7, 180: 0.8, 900: 0.5, 1800: 0.6},
    ]


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreInitialization:
    """Test vector store initialization and configuration."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_vector_store_initialization_defaults(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test vector store initialization with defaults."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        assert store.collection_name == "docmind_feat002_unified"
        assert store.embedding_dim == 1024
        assert store.dense_vector_name == "dense"
        assert store.sparse_vector_name == "sparse"
        assert store.rrf_alpha == 0.7
        assert store.stores_text is True

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_vector_store_initialization_custom_params(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test vector store initialization with custom parameters."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore(
            url="http://custom:6333",
            collection_name="custom_collection",
            embedding_dim=512,
            rrf_alpha=0.8,
        )

        assert store.collection_name == "custom_collection"
        assert store.embedding_dim == 512
        assert store.rrf_alpha == 0.8

        # Verify client initialized with custom URL
        mock_qdrant_client_class.assert_called_with(url="http://custom:6333")

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_vector_store_initialization_existing_client(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test vector store initialization with existing client."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        store = QdrantUnifiedVectorStore(client=mock_qdrant_client)

        # Should use provided client, not create new one
        assert store.qdrant_client == mock_qdrant_client
        mock_qdrant_client_class.assert_not_called()

    def test_vector_store_initialization_missing_qdrant(self):
        """Test vector store initialization fails when QdrantClient unavailable."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        with patch("src.retrieval.vector_store.QdrantClient", None):
            with pytest.raises(ImportError, match="qdrant-client not available"):
                QdrantUnifiedVectorStore()

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_collection_initialization_new_collection(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test collection initialization creates new collection."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        # Mock empty collections list
        collections_response = Mock()
        collections_response.collections = []
        mock_qdrant_client.get_collections.return_value = collections_response

        store = QdrantUnifiedVectorStore(collection_name="new_collection")

        # Verify collection creation was called
        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "new_collection"
        assert "dense" in call_kwargs["vectors_config"]
        assert "sparse" in call_kwargs["sparse_vectors_config"]

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_collection_initialization_existing_collection(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test collection initialization uses existing collection."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        # Mock existing collection
        existing_collection = Mock()
        existing_collection.name = "existing_collection"
        collections_response = Mock()
        collections_response.collections = [existing_collection]
        mock_qdrant_client.get_collections.return_value = collections_response

        store = QdrantUnifiedVectorStore(collection_name="existing_collection")

        # Should not create new collection
        mock_qdrant_client.create_collection.assert_not_called()

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_collection_initialization_retry_on_error(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test collection initialization retries on connection errors."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        # First call fails, second succeeds
        collections_response = Mock()
        collections_response.collections = []
        mock_qdrant_client.get_collections.side_effect = [
            ConnectionError("Connection failed"),
            collections_response,
        ]

        store = QdrantUnifiedVectorStore()

        # Should have retried
        assert mock_qdrant_client.get_collections.call_count == 2


@pytest.mark.unit
class TestVectorStoreAddOperations:
    """Test adding nodes with dense and sparse embeddings."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_add_nodes_with_embeddings(
        self,
        mock_qdrant_client_class,
        mock_qdrant_client,
        sample_nodes,
        sample_dense_embeddings,
        sample_sparse_embeddings,
    ):
        """Test adding nodes with both dense and sparse embeddings."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        node_ids = store.add(
            nodes=sample_nodes,
            dense_embeddings=sample_dense_embeddings,
            sparse_embeddings=sample_sparse_embeddings,
        )

        # Verify return values
        assert len(node_ids) == 3
        assert node_ids == ["node_1", "node_2", "node_3"]

        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args[1]
        assert call_args["collection_name"] == "docmind_feat002_unified"

        points = call_args["points"]
        assert len(points) == 3

        # Verify point structure
        for i, point in enumerate(points):
            assert point.id == f"node_{i + 1}"
            assert "dense" in point.vector
            assert "sparse" in point.sparse_vector
            assert "text" in point.payload
            assert "metadata" in point.payload

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_add_nodes_dense_only(
        self,
        mock_qdrant_client_class,
        mock_qdrant_client,
        sample_nodes,
        sample_dense_embeddings,
    ):
        """Test adding nodes with only dense embeddings."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        node_ids = store.add(
            nodes=sample_nodes,
            dense_embeddings=sample_dense_embeddings,
            sparse_embeddings=None,
        )

        assert len(node_ids) == 3

        # Verify points have dense vectors but no sparse
        call_args = mock_qdrant_client.upsert.call_args[1]
        points = call_args["points"]

        for point in points:
            assert "dense" in point.vector
            assert not point.sparse_vector  # Should be empty

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_add_nodes_sparse_only(
        self,
        mock_qdrant_client_class,
        mock_qdrant_client,
        sample_nodes,
        sample_sparse_embeddings,
    ):
        """Test adding nodes with only sparse embeddings."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        node_ids = store.add(
            nodes=sample_nodes,
            dense_embeddings=None,
            sparse_embeddings=sample_sparse_embeddings,
        )

        assert len(node_ids) == 3

        # Verify points have sparse vectors but no dense
        call_args = mock_qdrant_client.upsert.call_args[1]
        points = call_args["points"]

        for point in points:
            assert not point.vector  # Should be empty
            assert "sparse" in point.sparse_vector

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_add_empty_nodes_list(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test adding empty nodes list."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        node_ids = store.add(nodes=[], dense_embeddings=[], sparse_embeddings=[])

        assert node_ids == []
        mock_qdrant_client.upsert.assert_not_called()

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_add_nodes_retry_on_error(
        self,
        mock_qdrant_client_class,
        mock_qdrant_client,
        sample_nodes,
        sample_dense_embeddings,
    ):
        """Test add operation retries on connection errors."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        # First call fails, second succeeds
        mock_qdrant_client.upsert.side_effect = [
            ConnectionError("Connection failed"),
            None,
        ]

        store = QdrantUnifiedVectorStore()

        node_ids = store.add(
            nodes=sample_nodes, dense_embeddings=sample_dense_embeddings
        )

        assert len(node_ids) == 3
        assert mock_qdrant_client.upsert.call_count == 2

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_add_nodes_embedding_dimension_validation(
        self, mock_qdrant_client_class, mock_qdrant_client, sample_nodes
    ):
        """Test adding nodes validates embedding dimensions."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        # Valid 1024-dimensional embeddings
        valid_embeddings = [np.random.randn(1024).tolist() for _ in range(3)]

        node_ids = store.add(nodes=sample_nodes, dense_embeddings=valid_embeddings)

        assert len(node_ids) == 3

        # Verify embeddings were added correctly
        call_args = mock_qdrant_client.upsert.call_args[1]
        points = call_args["points"]

        for point in points:
            dense_vector = point.vector["dense"]
            assert len(dense_vector) == 1024


@pytest.mark.unit
class TestVectorStoreQueryOperations:
    """Test query operations including hybrid search and RRF fusion."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_dense_search_only(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test dense-only vector search."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        query = VectorStoreQuery(similarity_top_k=5)
        dense_embedding = np.random.randn(1024).tolist()

        result = store.query(
            query=query,
            dense_embedding=dense_embedding,
            sparse_embedding=None,
        )

        # Verify search was called with dense vector
        mock_qdrant_client.search.assert_called_once()
        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs["collection_name"] == "docmind_feat002_unified"
        assert call_kwargs["query_vector"][0] == "dense"
        assert call_kwargs["query_vector"][1] == dense_embedding
        assert call_kwargs["limit"] == 5

        # Verify result structure
        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) == 3
        assert len(result.similarities) == 3
        assert len(result.ids) == 3

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_sparse_search_only(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test sparse-only vector search."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        query = VectorStoreQuery(similarity_top_k=3)
        sparse_embedding = {100: 0.8, 200: 0.6, 500: 0.4}

        result = store.query(
            query=query,
            dense_embedding=None,
            sparse_embedding=sparse_embedding,
        )

        # Verify sparse search was called
        mock_qdrant_client.search.assert_called_once()
        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs["collection_name"] == "docmind_feat002_unified"

        # Verify sparse vector structure
        query_vector = call_kwargs["query_vector"]
        assert query_vector.name == "sparse"
        assert query_vector.vector.indices == [100, 200, 500]
        assert query_vector.vector.values == [0.8, 0.6, 0.4]

        assert isinstance(result, VectorStoreQueryResult)

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_hybrid_search_with_rrf_fusion(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test hybrid search with RRF score fusion."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        # Mock different results for dense and sparse searches
        def side_effect_search(*args, **kwargs):
            query_vector = kwargs.get("query_vector")
            if isinstance(query_vector, tuple) and query_vector[0] == "dense":
                # Dense search results
                return [
                    Mock(
                        id="1",
                        score=0.95,
                        payload={
                            "text": "Dense result 1",
                            "metadata": {},
                            "node_id": "1",
                        },
                    ),
                    Mock(
                        id="2",
                        score=0.87,
                        payload={
                            "text": "Dense result 2",
                            "metadata": {},
                            "node_id": "2",
                        },
                    ),
                    Mock(
                        id="4",
                        score=0.75,
                        payload={
                            "text": "Dense result 4",
                            "metadata": {},
                            "node_id": "4",
                        },
                    ),
                ]
            else:
                # Sparse search results
                return [
                    Mock(
                        id="3",
                        score=0.92,
                        payload={
                            "text": "Sparse result 3",
                            "metadata": {},
                            "node_id": "3",
                        },
                    ),
                    Mock(
                        id="1",
                        score=0.85,
                        payload={
                            "text": "Sparse result 1",
                            "metadata": {},
                            "node_id": "1",
                        },
                    ),
                    Mock(
                        id="5",
                        score=0.78,
                        payload={
                            "text": "Sparse result 5",
                            "metadata": {},
                            "node_id": "5",
                        },
                    ),
                ]

        mock_qdrant_client.search.side_effect = side_effect_search

        store = QdrantUnifiedVectorStore()

        query = VectorStoreQuery(similarity_top_k=3)
        dense_embedding = np.random.randn(1024).tolist()
        sparse_embedding = {100: 0.8, 200: 0.6}

        result = store.query(
            query=query,
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
        )

        # Should call search twice (dense + sparse)
        assert mock_qdrant_client.search.call_count == 2

        # Verify hybrid result with RRF fusion
        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) == 3  # Top 3 after fusion
        assert len(result.similarities) == 3
        assert len(result.ids) == 3

        # Results should be ordered by fused scores
        assert all(
            result.similarities[i] >= result.similarities[i + 1]
            for i in range(len(result.similarities) - 1)
        )

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_empty_query_handling(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test handling of query with no embeddings."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        query = VectorStoreQuery(similarity_top_k=5)

        result = store.query(
            query=query,
            dense_embedding=None,
            sparse_embedding=None,
        )

        # Should not call search
        mock_qdrant_client.search.assert_not_called()

        # Should return empty result
        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) == 0
        assert len(result.similarities) == 0
        assert len(result.ids) == 0

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_query_error_handling(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test query operation error handling."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client
        mock_qdrant_client.search.side_effect = ConnectionError("Connection failed")

        store = QdrantUnifiedVectorStore()

        query = VectorStoreQuery(similarity_top_k=5)
        dense_embedding = np.random.randn(1024).tolist()

        result = store.query(query=query, dense_embedding=dense_embedding)

        # Should return empty result on error
        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) == 0


@pytest.mark.unit
class TestRRFFusion:
    """Test Reciprocal Rank Fusion (RRF) algorithm implementation."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_rrf_fusion_algorithm(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test RRF fusion algorithm correctness."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore(rrf_alpha=0.7)  # 70% dense, 30% sparse

        # Create mock results with known rankings
        dense_results = [
            Mock(id="doc1", score=0.9),  # Rank 1
            Mock(id="doc2", score=0.8),  # Rank 2
            Mock(id="doc3", score=0.7),  # Rank 3
        ]

        sparse_results = [
            Mock(id="doc2", score=0.95),  # Rank 1
            Mock(id="doc4", score=0.85),  # Rank 2
            Mock(id="doc1", score=0.75),  # Rank 3
        ]

        fused_results = store._apply_rrf_fusion(dense_results, sparse_results, k=60)

        # Verify RRF calculation
        # doc1: dense_rank=1, sparse_rank=3 -> 0.7 * 1/61 + 0.3 * 1/63
        # doc2: dense_rank=2, sparse_rank=1 -> 0.7 * 1/62 + 0.3 * 1/61
        # doc3: dense_rank=3, sparse_rank=None -> 0.7 * 1/63 + 0.3 * 0
        # doc4: dense_rank=None, sparse_rank=2 -> 0.7 * 0 + 0.3 * 1/62

        # doc2 should have highest fused score (appears in both, high ranks)
        assert len(fused_results) == 4
        assert fused_results[0].id == "doc2"  # Should be ranked first

        # Verify scores were updated with fused values
        for result in fused_results:
            assert hasattr(result, "score")
            assert result.score > 0

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_rrf_fusion_alpha_weighting(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test RRF fusion alpha weighting affects results."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        # Test different alpha values
        dense_results = [Mock(id="doc1", score=0.9)]
        sparse_results = [Mock(id="doc1", score=0.8)]

        # High alpha (dense-weighted)
        store_dense = QdrantUnifiedVectorStore(rrf_alpha=0.9)
        result_dense = store_dense._apply_rrf_fusion(dense_results, sparse_results)

        # Low alpha (sparse-weighted)
        store_sparse = QdrantUnifiedVectorStore(rrf_alpha=0.1)
        result_sparse = store_sparse._apply_rrf_fusion(dense_results, sparse_results)

        # Both should have same document but different scores
        assert result_dense[0].id == result_sparse[0].id == "doc1"
        # Dense-weighted should have different score than sparse-weighted
        assert result_dense[0].score != result_sparse[0].score

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_rrf_fusion_k_constant(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test RRF fusion k constant affects scores."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        dense_results = [Mock(id="doc1", score=0.9)]
        sparse_results = [Mock(id="doc1", score=0.8)]

        # Different k values
        result_k30 = store._apply_rrf_fusion(dense_results, sparse_results, k=30)
        result_k90 = store._apply_rrf_fusion(dense_results, sparse_results, k=90)

        # Different k values should produce different scores
        assert result_k30[0].score != result_k90[0].score


@pytest.mark.unit
class TestVectorStoreDeleteOperations:
    """Test delete and cleanup operations."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_delete_by_doc_id(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test deleting documents by reference document ID."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        store.delete(ref_doc_id="doc_123")

        # Verify delete was called with correct filter
        mock_qdrant_client.delete.assert_called_once()
        call_kwargs = mock_qdrant_client.delete.call_args[1]
        assert call_kwargs["collection_name"] == "docmind_feat002_unified"

        # Verify filter structure
        points_selector = call_kwargs["points_selector"]
        assert hasattr(points_selector, "must")

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_delete_error_handling(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test delete operation error handling."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client
        mock_qdrant_client.delete.side_effect = ConnectionError("Connection failed")

        store = QdrantUnifiedVectorStore()

        with pytest.raises(ConnectionError):
            store.delete(ref_doc_id="doc_123")

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_clear_collection(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test clearing all documents from collection."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore(collection_name="test_collection")

        store.clear()

        # Verify collection was deleted
        mock_qdrant_client.delete_collection.assert_called_once_with(
            collection_name="test_collection"
        )

        # Verify collection was recreated (via retry mechanism)
        mock_qdrant_client.get_collections.assert_called()

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_clear_error_handling(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test clear operation error handling."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client
        mock_qdrant_client.delete_collection.side_effect = RuntimeError("Delete failed")

        store = QdrantUnifiedVectorStore()

        with pytest.raises(RuntimeError):
            store.clear()


@pytest.mark.unit
class TestVectorStoreConfiguration:
    """Test configuration classes and factory functions."""

    def test_retrieval_config_defaults(self):
        """Test RetrievalConfig default values."""
        from src.retrieval.vector_store import RetrievalConfig

        config = RetrievalConfig()

        assert config.strategy == "hybrid"
        assert config.top_k == 10
        assert config.use_reranking is True
        assert config.reranking_top_k == 5
        assert config.reranker_model == "BAAI/bge-reranker-v2-m3"
        assert config.rrf_alpha == 0.7
        assert config.rrf_k_constant == 60

    def test_retrieval_config_validation(self):
        """Test RetrievalConfig validation."""
        from src.retrieval.vector_store import RetrievalConfig

        config = RetrievalConfig(
            strategy="dense",
            top_k=20,
            use_reranking=False,
            rrf_alpha=0.8,
            rrf_k_constant=30,
        )

        assert config.strategy == "dense"
        assert config.top_k == 20
        assert config.use_reranking is False
        assert config.rrf_alpha == 0.8
        assert config.rrf_k_constant == 30

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig defaults in vector store context."""
        from src.retrieval.vector_store import EmbeddingConfig

        config = EmbeddingConfig()

        assert config.model_name == "BAAI/bge-m3"
        assert config.dimension == 1024
        assert config.max_length == 8192
        assert config.batch_size_gpu == 12

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_create_unified_qdrant_store_factory(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test factory function for creating unified vector store."""
        from src.retrieval.vector_store import create_unified_qdrant_store

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = create_unified_qdrant_store(
            url="http://custom:6333",
            collection_name="custom_collection",
            embedding_dim=512,
            rrf_alpha=0.6,
        )

        assert store.collection_name == "custom_collection"
        assert store.embedding_dim == 512
        assert store.rrf_alpha == 0.6

        # Verify client creation
        mock_qdrant_client_class.assert_called_with(url="http://custom:6333")


@pytest.mark.unit
class TestVectorStorePerformance:
    """Test performance optimization and batch operations."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_batch_add_performance(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test batch add operations for performance."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        # Large batch of nodes
        large_batch_nodes = [
            TextNode(text=f"Document {i} content", id_=f"node_{i}") for i in range(100)
        ]
        large_batch_embeddings = [np.random.randn(1024).tolist() for _ in range(100)]

        node_ids = store.add(
            nodes=large_batch_nodes, dense_embeddings=large_batch_embeddings
        )

        assert len(node_ids) == 100

        # Should call upsert once with all points (batch operation)
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args[1]
        assert len(call_args["points"]) == 100

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_fusion_limit_optimization(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test fusion limit optimization for hybrid search."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        # Mock search to return varied results
        mock_qdrant_client.search.return_value = [
            Mock(
                id=f"doc_{i}",
                score=0.9 - i * 0.1,
                payload={"text": f"Text {i}", "metadata": {}, "node_id": f"doc_{i}"},
            )
            for i in range(10)
        ]

        store = QdrantUnifiedVectorStore()

        query = VectorStoreQuery(similarity_top_k=5)  # Want 5 results
        dense_embedding = np.random.randn(1024).tolist()
        sparse_embedding = {100: 0.8}

        result = store.query(
            query=query,
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
        )

        # Should call search twice with fusion_limit (2 * similarity_top_k)
        assert mock_qdrant_client.search.call_count == 2

        # Each search call should request more results for fusion
        for call in mock_qdrant_client.search.call_args_list:
            call_kwargs = call[1]
            assert call_kwargs["limit"] == 10  # 2 * similarity_top_k

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_fusion_limit_maximum(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test fusion limit respects maximum."""
        from src.retrieval.vector_store import (
            MAX_FUSION_LIMIT,
            QdrantUnifiedVectorStore,
        )

        mock_qdrant_client_class.return_value = mock_qdrant_client
        mock_qdrant_client.search.return_value = []

        store = QdrantUnifiedVectorStore()

        # Request more than MAX_FUSION_LIMIT
        large_query = VectorStoreQuery(similarity_top_k=30)  # Would be 60 for fusion

        store.query(
            query=large_query,
            dense_embedding=np.random.randn(1024).tolist(),
            sparse_embedding={100: 0.8},
        )

        # Should be limited to MAX_FUSION_LIMIT
        for call in mock_qdrant_client.search.call_args_list:
            call_kwargs = call[1]
            assert call_kwargs["limit"] <= MAX_FUSION_LIMIT


@pytest.mark.unit
class TestVectorStoreDimensionValidation:
    """Test dimension validation and consistency."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_embedding_dimension_consistency(
        self, mock_qdrant_client_class, mock_qdrant_client, sample_nodes
    ):
        """Test embedding dimension consistency with BGE-M3."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore(embedding_dim=1024)

        # Test with correct 1024-dimensional embeddings
        embeddings_1024 = [np.random.randn(1024).tolist() for _ in range(3)]

        node_ids = store.add(nodes=sample_nodes, dense_embeddings=embeddings_1024)

        assert len(node_ids) == 3

        # Verify collection was created with 1024 dimensions
        create_call = mock_qdrant_client.create_collection.call_args[1]
        vectors_config = create_call["vectors_config"]
        assert vectors_config["dense"].size == 1024

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_query_embedding_dimension_validation(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test query embedding dimension validation."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client
        mock_qdrant_client.search.return_value = []

        store = QdrantUnifiedVectorStore(embedding_dim=1024)

        query = VectorStoreQuery(similarity_top_k=5)

        # Test with correct dimension
        correct_embedding = np.random.randn(1024).tolist()
        result = store.query(query=query, dense_embedding=correct_embedding)

        assert isinstance(result, VectorStoreQueryResult)

        # Verify search was called with correct embedding
        call_kwargs = mock_qdrant_client.search.call_args[1]
        query_vector = call_kwargs["query_vector"]
        assert len(query_vector[1]) == 1024


@pytest.mark.unit
class TestVectorStoreErrorHandling:
    """Test comprehensive error handling and recovery."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_connection_error_resilience(
        self,
        mock_qdrant_client_class,
        mock_qdrant_client,
        sample_nodes,
        sample_dense_embeddings,
    ):
        """Test resilience to connection errors with retry logic."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        # Simulate intermittent connection failures
        mock_qdrant_client.upsert.side_effect = [
            ConnectionError("Connection lost"),
            ConnectionError("Still failing"),
            None,  # Finally succeeds
        ]

        store = QdrantUnifiedVectorStore()

        node_ids = store.add(
            nodes=sample_nodes, dense_embeddings=sample_dense_embeddings
        )

        # Should eventually succeed after retries
        assert len(node_ids) == 3
        assert mock_qdrant_client.upsert.call_count == 3

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_timeout_error_handling(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test timeout error handling."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client
        mock_qdrant_client.search.side_effect = TimeoutError("Request timeout")

        store = QdrantUnifiedVectorStore()

        query = VectorStoreQuery(similarity_top_k=5)
        result = store.query(
            query=query, dense_embedding=np.random.randn(1024).tolist()
        )

        # Should return empty result gracefully
        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) == 0

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_runtime_error_handling(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test runtime error handling."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client
        mock_qdrant_client.get_collections.side_effect = RuntimeError(
            "Unexpected error"
        )

        # Should raise the error (not recoverable)
        with pytest.raises(RuntimeError):
            QdrantUnifiedVectorStore()

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_malformed_query_handling(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test handling of malformed queries."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client
        mock_qdrant_client.search.side_effect = ValueError("Invalid query format")

        store = QdrantUnifiedVectorStore()

        query = VectorStoreQuery(similarity_top_k=5)
        result = store.query(
            query=query, dense_embedding=np.random.randn(1024).tolist()
        )

        # Should return empty result on malformed query
        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) == 0


@pytest.mark.unit
class TestVectorStoreIntegration:
    """Test integration with other components."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_llamaindex_compatibility(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test compatibility with LlamaIndex interfaces."""
        from llama_index.core.vector_stores.types import BasePydanticVectorStore

        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        # Should be instance of LlamaIndex base class
        assert isinstance(store, BasePydanticVectorStore)

        # Should have required attributes
        assert store.stores_text is True
        assert hasattr(store, "add")
        assert hasattr(store, "query")
        assert hasattr(store, "delete")

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_node_metadata_preservation(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test that node metadata is preserved through storage."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore()

        # Create node with rich metadata
        node = TextNode(
            text="Test document with metadata",
            metadata={
                "source": "test.pdf",
                "page": 5,
                "section": "Introduction",
                "author": "Test Author",
                "date": "2024-01-01",
            },
            id_="rich_node",
        )

        store.add(nodes=[node], dense_embeddings=[np.random.randn(1024).tolist()])

        # Verify metadata was preserved in payload
        call_args = mock_qdrant_client.upsert.call_args[1]
        point = call_args["points"][0]

        assert point.payload["text"] == "Test document with metadata"
        assert point.payload["node_id"] == "rich_node"

        # Original metadata should be nested
        metadata = point.payload["metadata"]
        assert metadata["source"] == "test.pdf"
        assert metadata["page"] == 5
        assert metadata["section"] == "Introduction"

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_result_conversion_accuracy(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test accurate conversion of Qdrant results to VectorStoreQueryResult."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        # Setup realistic mock search result
        mock_result = Mock()
        mock_result.id = "test_doc_1"
        mock_result.score = 0.92
        mock_result.payload = {
            "text": "Test document content for accuracy verification",
            "metadata": {"source": "accuracy_test.pdf", "page": 3},
            "node_id": "accurate_node_1",
        }
        mock_qdrant_client.search.return_value = [mock_result]

        store = QdrantUnifiedVectorStore()

        query = VectorStoreQuery(similarity_top_k=1)
        result = store.query(
            query=query, dense_embedding=np.random.randn(1024).tolist()
        )

        # Verify accurate conversion
        assert len(result.nodes) == 1
        assert len(result.similarities) == 1
        assert len(result.ids) == 1

        node = result.nodes[0]
        assert isinstance(node, TextNode)
        assert node.text == "Test document content for accuracy verification"
        assert node.metadata["source"] == "accuracy_test.pdf"
        assert node.id_ == "accurate_node_1"

        assert result.similarities[0] == 0.92
        assert result.ids[0] == "test_doc_1"
