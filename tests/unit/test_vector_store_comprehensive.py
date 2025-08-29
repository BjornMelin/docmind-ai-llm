"""Comprehensive unit tests for Qdrant unified vector store.

Tests focus on covering the 182 uncovered statements in vector_store.py
with emphasis on CRUD operations, hybrid search, RRF fusion, and error
handling using mocked Qdrant client operations.

Key areas:
- QdrantUnifiedVectorStore initialization and collection setup
- Vector store CRUD operations (add, query, delete, clear)
- Hybrid search with RRF score fusion
- Dense and sparse search strategies
- Error handling and retry logic
- Configuration validation and optimization
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from src.retrieval.vector_store import QdrantUnifiedVectorStore


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient for comprehensive testing."""
    client = Mock()

    # Mock collection operations
    mock_collections_response = Mock()
    mock_collections_response.collections = [
        Mock(name="existing_collection"),
        Mock(name="docmind_feat002_unified"),
    ]
    client.get_collections.return_value = mock_collections_response
    client.create_collection.return_value = None
    client.delete_collection.return_value = None

    # Mock search operations
    def mock_search(**kwargs):
        collection_name = kwargs.get("collection_name")
        limit = kwargs.get("limit", 10)

        # Generate mock search results
        results = []
        for i in range(limit):
            result = Mock()
            result.id = f"doc_{i}"
            result.score = 0.9 - (i * 0.1)
            result.payload = {
                "text": f"Document {i} content",
                "metadata": {"source": f"file_{i}.txt"},
                "node_id": f"node_{i}",
                "doc_id": f"document_{i}",
                "chunk_id": f"chunk_{i}",
            }
            results.append(result)
        return results

    client.search.side_effect = mock_search

    # Mock CRUD operations
    client.upsert.return_value = None
    client.delete.return_value = None

    return client


@pytest.fixture
def vector_store(mock_qdrant_client):
    """Create QdrantUnifiedVectorStore instance with mocked client."""
    with patch("src.retrieval.vector_store.QdrantClient") as mock_client_class:
        mock_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore(
            client=mock_qdrant_client,
            collection_name="test_collection",
            embedding_dim=1024,
            rrf_alpha=0.7,
        )
        return store


@pytest.fixture
def sample_nodes():
    """Create sample TextNode instances for testing."""
    nodes = []
    for i in range(3):
        node = TextNode(
            text=f"Sample document {i} with relevant content for testing",
            metadata={"source": f"test_file_{i}.txt", "page": i + 1},
            id_=f"node_{i}",
        )
        # Add custom attributes
        node.doc_id = f"document_{i}"
        node.chunk_id = f"chunk_{i}"
        nodes.append(node)
    return nodes


@pytest.fixture
def sample_embeddings():
    """Create sample dense and sparse embeddings."""
    dense_embeddings = [np.random.randn(1024).tolist() for _ in range(3)]

    sparse_embeddings = [
        {i: float(np.random.random()) for i in range(10, 20)} for _ in range(3)
    ]

    return dense_embeddings, sparse_embeddings


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreInitialization:
    """Test vector store initialization and configuration."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_initialization_with_default_client(self, mock_client_class):
        """Test initialization creates default client when none provided."""
        mock_client = Mock()
        mock_collections_response = Mock()
        mock_collections_response.collections = []
        mock_client.get_collections.return_value = mock_collections_response
        mock_client_class.return_value = mock_client

        store = QdrantUnifiedVectorStore()

        # Verify client was created with default URL
        mock_client_class.assert_called_once_with(url="http://localhost:6333")
        assert store.qdrant_client == mock_client
        assert store.collection_name == "docmind_feat002_unified"
        assert store.embedding_dim == 1024
        assert store.rrf_alpha == 0.7
        assert store.dense_vector_name == "dense"
        assert store.sparse_vector_name == "sparse"

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_initialization_with_custom_parameters(self, mock_client_class):
        """Test initialization with custom parameters."""
        mock_client = Mock()
        mock_collections_response = Mock()
        mock_collections_response.collections = []
        mock_client.get_collections.return_value = mock_collections_response
        mock_client_class.return_value = mock_client

        store = QdrantUnifiedVectorStore(
            url="http://custom:6333",
            collection_name="custom_collection",
            embedding_dim=512,
            rrf_alpha=0.8,
        )

        mock_client_class.assert_called_once_with(url="http://custom:6333")
        assert store.collection_name == "custom_collection"
        assert store.embedding_dim == 512
        assert store.rrf_alpha == 0.8

    def test_initialization_missing_qdrant_client(self):
        """Test initialization fails when QdrantClient is not available."""
        with patch("src.retrieval.vector_store.QdrantClient", None):
            with pytest.raises(ImportError, match="qdrant-client not available"):
                QdrantUnifiedVectorStore()

    def test_collection_creation_new_collection(self, mock_qdrant_client):
        """Test collection creation when collection doesn't exist."""
        # Mock collection doesn't exist
        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name="other_collection")]
        mock_qdrant_client.get_collections.return_value = mock_collections_response

        with patch("src.retrieval.vector_store.QdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            QdrantUnifiedVectorStore(
                client=mock_qdrant_client, collection_name="new_collection"
            )

            # Verify collection was created
            mock_qdrant_client.create_collection.assert_called_once()
            create_call = mock_qdrant_client.create_collection.call_args

            assert create_call[1]["collection_name"] == "new_collection"
            assert "dense" in create_call[1]["vectors_config"]
            assert "sparse" in create_call[1]["sparse_vectors_config"]

    def test_collection_reuse_existing_collection(self, mock_qdrant_client):
        """Test reusing existing collection."""
        # Mock collection exists
        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name="existing_collection")]
        mock_qdrant_client.get_collections.return_value = mock_collections_response

        with patch("src.retrieval.vector_store.QdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            QdrantUnifiedVectorStore(
                client=mock_qdrant_client, collection_name="existing_collection"
            )

            # Verify collection was not created
            mock_qdrant_client.create_collection.assert_not_called()

    def test_initialization_retry_on_connection_error(self, mock_qdrant_client):
        """Test initialization retry logic on connection errors."""
        # First call fails, second succeeds
        mock_qdrant_client.get_collections.side_effect = [
            ConnectionError("Connection failed"),
            Mock(collections=[]),
        ]

        with patch("src.retrieval.vector_store.QdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            with pytest.raises(ConnectionError):
                QdrantUnifiedVectorStore(client=mock_qdrant_client)

            # Verify retry was attempted
            assert (
                mock_qdrant_client.get_collections.call_count == 3
            )  # Initial + 2 retries


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreAdd:
    """Test vector store add operations."""

    def test_add_nodes_success(self, vector_store, sample_nodes, sample_embeddings):
        """Test successful addition of nodes with embeddings."""
        dense_embeddings, sparse_embeddings = sample_embeddings

        node_ids = vector_store.add(
            nodes=sample_nodes,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # Verify return values
        assert len(node_ids) == 3
        assert all(node_id.startswith("node_") for node_id in node_ids)

        # Verify upsert was called
        vector_store.qdrant_client.upsert.assert_called_once()
        upsert_call = vector_store.qdrant_client.upsert.call_args

        assert upsert_call[1]["collection_name"] == "test_collection"
        points = upsert_call[1]["points"]
        assert len(points) == 3

        # Verify point structure
        point = points[0]
        assert hasattr(point, "id")
        assert hasattr(point, "vector")
        assert hasattr(point, "sparse_vector")
        assert hasattr(point, "payload")

        # Verify payload structure
        payload = point.payload
        assert "text" in payload
        assert "metadata" in payload
        assert "node_id" in payload
        assert "doc_id" in payload
        assert "chunk_id" in payload

    def test_add_nodes_dense_only(self, vector_store, sample_nodes, sample_embeddings):
        """Test adding nodes with dense embeddings only."""
        dense_embeddings, _ = sample_embeddings

        node_ids = vector_store.add(
            nodes=sample_nodes,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=None,
        )

        assert len(node_ids) == 3
        vector_store.qdrant_client.upsert.assert_called_once()

        # Verify sparse vectors are empty
        points = vector_store.qdrant_client.upsert.call_args[1]["points"]
        point = points[0]
        assert point.sparse_vector == {}  # No sparse vectors

    def test_add_nodes_sparse_only(self, vector_store, sample_nodes, sample_embeddings):
        """Test adding nodes with sparse embeddings only."""
        _, sparse_embeddings = sample_embeddings

        node_ids = vector_store.add(
            nodes=sample_nodes,
            dense_embeddings=None,
            sparse_embeddings=sparse_embeddings,
        )

        assert len(node_ids) == 3
        vector_store.qdrant_client.upsert.assert_called_once()

        # Verify dense vectors are empty
        points = vector_store.qdrant_client.upsert.call_args[1]["points"]
        point = points[0]
        assert "dense" not in point.vector  # No dense vectors

    def test_add_empty_nodes(self, vector_store):
        """Test adding empty list of nodes."""
        node_ids = vector_store.add(nodes=[])

        assert node_ids == []
        vector_store.qdrant_client.upsert.assert_not_called()

    def test_add_nodes_mismatched_embeddings(
        self, vector_store, sample_nodes, sample_embeddings
    ):
        """Test adding nodes with mismatched embedding counts."""
        dense_embeddings, sparse_embeddings = sample_embeddings

        # Remove one embedding to create mismatch
        dense_embeddings = dense_embeddings[:2]

        node_ids = vector_store.add(
            nodes=sample_nodes,  # 3 nodes
            dense_embeddings=dense_embeddings,  # 2 embeddings
            sparse_embeddings=sparse_embeddings,  # 3 embeddings
        )

        # Should still succeed but only use available embeddings
        assert len(node_ids) == 3
        vector_store.qdrant_client.upsert.assert_called_once()

        points = vector_store.qdrant_client.upsert.call_args[1]["points"]
        # Third point should not have dense embedding
        assert "dense" not in points[2].vector

    def test_add_nodes_connection_error_retry(self, vector_store, sample_nodes):
        """Test add operation retry on connection error."""
        vector_store.qdrant_client.upsert.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Still failing"),
            ConnectionError("Final failure"),
        ]

        with pytest.raises(ConnectionError):
            vector_store.add(nodes=sample_nodes)

        # Verify retry was attempted (3 attempts total)
        assert vector_store.qdrant_client.upsert.call_count == 3


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreQuery:
    """Test vector store query operations."""

    def test_query_hybrid_search(self, vector_store):
        """Test hybrid search with dense and sparse embeddings."""
        query = VectorStoreQuery(similarity_top_k=5)
        dense_embedding = np.random.randn(1024).tolist()
        sparse_embedding = {10: 0.8, 15: 0.6, 20: 0.9}

        result = vector_store.query(
            query=query,
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
        )

        # Verify result structure
        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) > 0
        assert len(result.similarities) == len(result.nodes)
        assert len(result.ids) == len(result.nodes)

        # Verify both dense and sparse searches were called
        assert vector_store.qdrant_client.search.call_count == 2
        search_calls = vector_store.qdrant_client.search.call_args_list

        # First call should be dense search
        dense_call = search_calls[0][1]
        assert dense_call["collection_name"] == "test_collection"
        assert dense_call["query_vector"][0] == "dense"
        assert dense_call["query_vector"][1] == dense_embedding

        # Second call should be sparse search
        sparse_call = search_calls[1][1]
        assert sparse_call["collection_name"] == "test_collection"
        sparse_query_vector = sparse_call["query_vector"]
        assert sparse_query_vector.name == "sparse"

    def test_query_dense_only(self, vector_store):
        """Test dense-only search."""
        query = VectorStoreQuery(similarity_top_k=3)
        dense_embedding = np.random.randn(1024).tolist()

        result = vector_store.query(query=query, dense_embedding=dense_embedding)

        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) > 0

        # Verify only one search call (dense)
        assert vector_store.qdrant_client.search.call_count == 1
        search_call = vector_store.qdrant_client.search.call_args[1]
        assert search_call["query_vector"][0] == "dense"
        assert search_call["limit"] == 3

    def test_query_sparse_only(self, vector_store):
        """Test sparse-only search."""
        query = VectorStoreQuery(similarity_top_k=4)
        sparse_embedding = {5: 0.7, 10: 0.8, 15: 0.5}

        result = vector_store.query(query=query, sparse_embedding=sparse_embedding)

        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) > 0

        # Verify only one search call (sparse)
        assert vector_store.qdrant_client.search.call_count == 1
        search_call = vector_store.qdrant_client.search.call_args[1]
        sparse_query_vector = search_call["query_vector"]
        assert sparse_query_vector.name == "sparse"

    def test_query_no_embeddings(self, vector_store):
        """Test query with no embeddings provided."""
        query = VectorStoreQuery(similarity_top_k=5)

        result = vector_store.query(query=query)

        # Should return empty results
        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) == 0
        assert len(result.similarities) == 0
        assert len(result.ids) == 0

        # No search calls should be made
        vector_store.qdrant_client.search.assert_not_called()

    def test_query_empty_sparse_embedding(self, vector_store):
        """Test query with empty sparse embedding dict."""
        query = VectorStoreQuery(similarity_top_k=3)
        dense_embedding = np.random.randn(1024).tolist()
        sparse_embedding = {}  # Empty dict

        result = vector_store.query(
            query=query,
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
        )

        # Should fall back to dense-only search
        assert isinstance(result, VectorStoreQueryResult)
        assert vector_store.qdrant_client.search.call_count == 1
        search_call = vector_store.qdrant_client.search.call_args[1]
        assert search_call["query_vector"][0] == "dense"

    def test_query_connection_error(self, vector_store):
        """Test query error handling on connection failure."""
        vector_store.qdrant_client.search.side_effect = ConnectionError(
            "Connection failed"
        )

        query = VectorStoreQuery(similarity_top_k=5)
        dense_embedding = np.random.randn(1024).tolist()

        result = vector_store.query(query=query, dense_embedding=dense_embedding)

        # Should return empty results on error
        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) == 0
        assert len(result.similarities) == 0
        assert len(result.ids) == 0


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreRRFFusion:
    """Test RRF (Reciprocal Rank Fusion) implementation."""

    def test_rrf_fusion_algorithm(self, vector_store):
        """Test RRF fusion algorithm with mock results."""
        # Create mock dense results
        dense_results = []
        for i in range(3):
            result = Mock()
            result.id = f"doc_{i}"
            result.score = 0.9 - (i * 0.1)  # 0.9, 0.8, 0.7
            result.payload = {"text": f"Dense doc {i}"}
            dense_results.append(result)

        # Create mock sparse results (different order)
        sparse_results = []
        for i in [2, 0, 1]:  # Different ranking
            result = Mock()
            result.id = f"doc_{i}"
            result.score = 0.8 - (i * 0.1)  # Different scores
            result.payload = {"text": f"Sparse doc {i}"}
            sparse_results.append(result)

        # Test RRF fusion
        fused_results = vector_store._apply_rrf_fusion(
            dense_results, sparse_results, k=60
        )

        # Verify fusion was applied
        assert len(fused_results) == 3  # Unique documents

        # Verify scores were updated with fusion
        for result in fused_results:
            assert hasattr(result, "score")
            assert 0.0 < result.score <= 1.0

        # Verify results are sorted by fused score (descending)
        scores = [result.score for result in fused_results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_fusion_alpha_weighting(self, vector_store):
        """Test RRF alpha weighting affects fusion results."""
        vector_store.rrf_alpha = 0.8  # Heavy dense weighting

        # Mock results where dense and sparse have different top results
        dense_results = [Mock(id="doc_0", score=0.9, payload={"text": "Dense top"})]
        sparse_results = [Mock(id="doc_1", score=0.9, payload={"text": "Sparse top"})]

        fused_results = vector_store._apply_rrf_fusion(dense_results, sparse_results)

        # With alpha=0.8, dense should be weighted higher
        doc_0_score = next(r.score for r in fused_results if r.id == "doc_0")
        doc_1_score = next(r.score for r in fused_results if r.id == "doc_1")

        # doc_0 should have higher fused score due to dense weighting
        assert doc_0_score > doc_1_score

    def test_rrf_fusion_no_overlap(self, vector_store):
        """Test RRF fusion with no overlapping documents."""
        dense_results = [
            Mock(id="dense_doc", score=0.9, payload={"text": "Dense only"})
        ]
        sparse_results = [
            Mock(id="sparse_doc", score=0.8, payload={"text": "Sparse only"})
        ]

        fused_results = vector_store._apply_rrf_fusion(dense_results, sparse_results)

        # Should include both documents
        assert len(fused_results) == 2
        result_ids = {result.id for result in fused_results}
        assert result_ids == {"dense_doc", "sparse_doc"}

    def test_rrf_fusion_complete_overlap(self, vector_store):
        """Test RRF fusion with complete document overlap."""
        # Same documents in both results
        dense_results = [
            Mock(id="doc_0", score=0.9, payload={"text": "Doc 0"}),
            Mock(id="doc_1", score=0.8, payload={"text": "Doc 1"}),
        ]
        sparse_results = [
            Mock(id="doc_1", score=0.95, payload={"text": "Doc 1"}),  # Different rank
            Mock(id="doc_0", score=0.85, payload={"text": "Doc 0"}),
        ]

        fused_results = vector_store._apply_rrf_fusion(dense_results, sparse_results)

        # Should have only 2 unique documents
        assert len(fused_results) == 2
        result_ids = {result.id for result in fused_results}
        assert result_ids == {"doc_0", "doc_1"}

        # Verify no duplicates
        assert len(set(r.id for r in fused_results)) == len(fused_results)


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreSearchStrategies:
    """Test individual search strategy implementations."""

    def test_dense_search_implementation(self, vector_store):
        """Test dense search implementation."""
        query = VectorStoreQuery(similarity_top_k=3)
        dense_embedding = np.random.randn(1024).tolist()

        result = vector_store._dense_search(query, dense_embedding)

        # Verify search call
        vector_store.qdrant_client.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=("dense", dense_embedding),
            limit=3,
            with_payload=True,
            with_vectors=False,
        )

        # Verify result conversion
        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.nodes) > 0

    def test_sparse_search_implementation(self, vector_store):
        """Test sparse search implementation."""
        query = VectorStoreQuery(similarity_top_k=4)
        sparse_embedding = {10: 0.8, 20: 0.6, 30: 0.9}

        result = vector_store._sparse_search(query, sparse_embedding)

        # Verify search call
        vector_store.qdrant_client.search.assert_called_once()
        search_call = vector_store.qdrant_client.search.call_args[1]

        assert search_call["collection_name"] == "test_collection"
        assert search_call["limit"] == 4
        assert search_call["with_payload"] is True
        assert search_call["with_vectors"] is False

        # Verify sparse vector structure
        sparse_query = search_call["query_vector"]
        assert sparse_query.name == "sparse"
        assert hasattr(sparse_query.vector, "indices")
        assert hasattr(sparse_query.vector, "values")

    def test_hybrid_search_fusion_limit(self, vector_store):
        """Test hybrid search respects fusion limit."""
        query = VectorStoreQuery(similarity_top_k=5)
        dense_embedding = np.random.randn(1024).tolist()
        sparse_embedding = {10: 0.8}

        vector_store._hybrid_search(query, dense_embedding, sparse_embedding)

        # Verify fusion limit is applied (2x requested, max 50)
        search_calls = vector_store.qdrant_client.search.call_args_list

        # Both dense and sparse searches should use fusion limit
        dense_limit = search_calls[0][1]["limit"]
        sparse_limit = search_calls[1][1]["limit"]

        expected_limit = min(5 * 2, 50)  # 2x top_k, max 50
        assert dense_limit == expected_limit
        assert sparse_limit == expected_limit

    def test_hybrid_search_max_fusion_limit(self, vector_store):
        """Test hybrid search respects max fusion limit."""
        query = VectorStoreQuery(similarity_top_k=30)  # High number
        dense_embedding = np.random.randn(1024).tolist()
        sparse_embedding = {10: 0.8}

        vector_store._hybrid_search(query, dense_embedding, sparse_embedding)

        # Should be capped at MAX_FUSION_LIMIT (50)
        search_calls = vector_store.qdrant_client.search.call_args_list
        dense_limit = search_calls[0][1]["limit"]
        sparse_limit = search_calls[1][1]["limit"]

        assert dense_limit == 50
        assert sparse_limit == 50


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreResultConversion:
    """Test result conversion from Qdrant to VectorStoreQueryResult."""

    def test_convert_results_success(self, vector_store):
        """Test successful result conversion."""
        # Create mock Qdrant results
        mock_results = []
        for i in range(3):
            result = Mock()
            result.id = f"result_{i}"
            result.score = 0.9 - (i * 0.1)
            result.payload = {
                "text": f"Document {i} content",
                "metadata": {"source": f"file_{i}.txt", "page": i + 1},
                "node_id": f"node_{i}",
            }
            mock_results.append(result)

        converted_result = vector_store._convert_results(mock_results)

        # Verify conversion
        assert isinstance(converted_result, VectorStoreQueryResult)
        assert len(converted_result.nodes) == 3
        assert len(converted_result.similarities) == 3
        assert len(converted_result.ids) == 3

        # Verify node properties
        for i, node in enumerate(converted_result.nodes):
            assert isinstance(node, TextNode)
            assert node.text == f"Document {i} content"
            assert node.metadata["source"] == f"file_{i}.txt"
            assert node.id_ == f"node_{i}"

        # Verify similarities are in correct order
        expected_similarities = [0.9, 0.8, 0.7]
        assert converted_result.similarities == expected_similarities

        # Verify IDs
        expected_ids = ["result_0", "result_1", "result_2"]
        assert converted_result.ids == expected_ids

    def test_convert_results_empty(self, vector_store):
        """Test conversion of empty results."""
        converted_result = vector_store._convert_results([])

        assert isinstance(converted_result, VectorStoreQueryResult)
        assert len(converted_result.nodes) == 0
        assert len(converted_result.similarities) == 0
        assert len(converted_result.ids) == 0

    def test_convert_results_missing_payload_fields(self, vector_store):
        """Test conversion handles missing payload fields gracefully."""
        mock_result = Mock()
        mock_result.id = "test_id"
        mock_result.score = 0.85
        mock_result.payload = {}  # Missing fields

        converted_result = vector_store._convert_results([mock_result])

        # Should handle missing fields gracefully
        assert len(converted_result.nodes) == 1
        node = converted_result.nodes[0]

        assert node.text == ""  # Default for missing text
        assert node.metadata == {}  # Default for missing metadata
        assert node.id_ is None  # Default for missing node_id


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreDelete:
    """Test vector store delete operations."""

    def test_delete_by_doc_id(self, vector_store):
        """Test deletion by document ID."""
        doc_id = "test_document_123"

        vector_store.delete(ref_doc_id=doc_id)

        # Verify delete was called with correct filter
        vector_store.qdrant_client.delete.assert_called_once()
        delete_call = vector_store.qdrant_client.delete.call_args[1]

        assert delete_call["collection_name"] == "test_collection"

        # Verify filter structure
        points_selector = delete_call["points_selector"]
        assert hasattr(points_selector, "must")
        filter_condition = points_selector.must[0]
        assert filter_condition.key == "doc_id"
        assert filter_condition.match.value == doc_id

    def test_delete_connection_error(self, vector_store):
        """Test delete error handling."""
        vector_store.qdrant_client.delete.side_effect = ConnectionError(
            "Connection failed"
        )

        with pytest.raises(ConnectionError):
            vector_store.delete(ref_doc_id="test_doc")

        vector_store.qdrant_client.delete.assert_called_once()


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreClear:
    """Test vector store clear operations."""

    def test_clear_collection(self, vector_store):
        """Test clearing entire collection."""
        vector_store.clear()

        # Verify collection was deleted and recreated
        vector_store.qdrant_client.delete_collection.assert_called_once_with(
            collection_name="test_collection"
        )

        # Note: Recreation is handled by _init_collection_with_retry which we've mocked

    def test_clear_connection_error(self, vector_store):
        """Test clear error handling."""
        vector_store.qdrant_client.delete_collection.side_effect = ConnectionError(
            "Connection failed"
        )

        with pytest.raises(ConnectionError):
            vector_store.clear()

        vector_store.qdrant_client.delete_collection.assert_called_once()


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreConfiguration:
    """Test configuration and settings validation."""

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
        """Test RetrievalConfig field validation."""
        from pydantic import ValidationError

        from src.retrieval.vector_store import RetrievalConfig

        # Test valid configuration
        config = RetrievalConfig(
            strategy="dense", top_k=15, rrf_alpha=0.8, rrf_k_constant=40
        )

        assert config.strategy == "dense"
        assert config.top_k == 15
        assert config.rrf_alpha == 0.8
        assert config.rrf_k_constant == 40

        # Test validation limits
        with pytest.raises(ValidationError):
            RetrievalConfig(top_k=0)  # Below minimum

        with pytest.raises(ValidationError):
            RetrievalConfig(top_k=100)  # Above maximum

        with pytest.raises(ValidationError):
            RetrievalConfig(rrf_alpha=-0.1)  # Below minimum

        with pytest.raises(ValidationError):
            RetrievalConfig(rrf_alpha=1.1)  # Above maximum

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig default values."""
        from src.retrieval.vector_store import EmbeddingConfig

        config = EmbeddingConfig()

        assert config.model_name == "BAAI/bge-m3"
        assert config.dimension == 1024
        assert config.max_length == 8192
        assert config.batch_size_gpu == 12
        assert config.batch_size_cpu == 4

    def test_embedding_config_validation(self):
        """Test EmbeddingConfig field validation."""
        from pydantic import ValidationError

        from src.retrieval.vector_store import EmbeddingConfig

        # Test valid configuration
        config = EmbeddingConfig(
            model_name="custom/model", dimension=512, max_length=4096, batch_size_gpu=8
        )

        assert config.model_name == "custom/model"
        assert config.dimension == 512
        assert config.max_length == 4096
        assert config.batch_size_gpu == 8

        # Test validation limits
        with pytest.raises(ValidationError):
            EmbeddingConfig(dimension=100)  # Below minimum

        with pytest.raises(ValidationError):
            EmbeddingConfig(max_length=256)  # Below minimum


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreFactory:
    """Test factory function for vector store creation."""

    @patch("src.retrieval.vector_store.QdrantUnifiedVectorStore")
    def test_create_unified_qdrant_store_defaults(self, mock_store_class):
        """Test factory function with default parameters."""
        from src.retrieval.vector_store import create_unified_qdrant_store

        mock_store = Mock()
        mock_store_class.return_value = mock_store

        store = create_unified_qdrant_store()

        # Verify factory was called with defaults
        mock_store_class.assert_called_once_with(
            url="http://localhost:6333",
            collection_name="docmind_feat002_unified",
            embedding_dim=1024,
            rrf_alpha=0.7,
        )

        assert store == mock_store

    @patch("src.retrieval.vector_store.QdrantUnifiedVectorStore")
    def test_create_unified_qdrant_store_custom_params(self, mock_store_class):
        """Test factory function with custom parameters."""
        from src.retrieval.vector_store import create_unified_qdrant_store

        mock_store = Mock()
        mock_store_class.return_value = mock_store

        store = create_unified_qdrant_store(
            url="http://custom:6333",
            collection_name="custom_collection",
            embedding_dim=512,
            rrf_alpha=0.8,
        )

        # Verify factory was called with custom params
        mock_store_class.assert_called_once_with(
            url="http://custom:6333",
            collection_name="custom_collection",
            embedding_dim=512,
            rrf_alpha=0.8,
        )

        assert store == mock_store


@pytest.mark.unit
class TestQdrantUnifiedVectorStoreIntegration:
    """Test integration scenarios and edge cases."""

    def test_vector_store_properties(self, vector_store):
        """Test vector store required properties."""
        assert vector_store.stores_text is True
        assert vector_store.is_embedding_query is False

        # Test BasePydanticVectorStore interface compliance
        required_attrs = ["stores_text", "is_embedding_query"]
        for attr in required_attrs:
            assert hasattr(vector_store, attr)

    def test_large_batch_add_operation(self, vector_store):
        """Test adding large batch of nodes."""
        # Create large batch of nodes
        large_batch_nodes = []
        for i in range(100):
            node = TextNode(
                text=f"Large batch document {i}",
                metadata={"batch_id": "large_test"},
                id_=f"large_node_{i}",
            )
            node.doc_id = f"large_doc_{i}"
            node.chunk_id = f"large_chunk_{i}"
            large_batch_nodes.append(node)

        # Create corresponding embeddings
        dense_embeddings = [np.random.randn(1024).tolist() for _ in range(100)]

        node_ids = vector_store.add(
            nodes=large_batch_nodes, dense_embeddings=dense_embeddings
        )

        assert len(node_ids) == 100
        vector_store.qdrant_client.upsert.assert_called_once()

        # Verify all 100 points were included
        points = vector_store.qdrant_client.upsert.call_args[1]["points"]
        assert len(points) == 100

    def test_query_performance_with_small_top_k(self, vector_store):
        """Test query performance optimization with small top_k."""
        query = VectorStoreQuery(similarity_top_k=1)  # Very small
        dense_embedding = np.random.randn(1024).tolist()
        sparse_embedding = {10: 0.8}

        vector_store._hybrid_search(query, dense_embedding, sparse_embedding)

        # Even with small top_k, fusion should use minimum reasonable limit
        search_calls = vector_store.qdrant_client.search.call_args_list
        fusion_limit = search_calls[0][1]["limit"]

        assert fusion_limit == 2  # 2x the top_k for fusion

    def test_rrf_constants_usage(self, vector_store):
        """Test RRF constants are used correctly."""
        from src.retrieval.vector_store import (
            DEFAULT_COLLECTION_SIZE,
            MAX_FUSION_LIMIT,
            RETRY_ATTEMPTS,
        )

        # Verify constants are reasonable values
        assert MAX_FUSION_LIMIT == 50
        assert DEFAULT_COLLECTION_SIZE == 1000
        assert RETRY_ATTEMPTS == 3

        # Test MAX_FUSION_LIMIT is respected
        query = VectorStoreQuery(similarity_top_k=100)
        dense_embedding = np.random.randn(1024).tolist()
        sparse_embedding = {10: 0.8}

        vector_store._hybrid_search(query, dense_embedding, sparse_embedding)

        search_calls = vector_store.qdrant_client.search.call_args_list
        fusion_limit = search_calls[0][1]["limit"]

        assert fusion_limit <= MAX_FUSION_LIMIT
