"""Integration tests for vector store search functionality.

Tests focus on search workflows, document indexing integration, hybrid search scenarios,
and real-world usage patterns. These complement unit tests by testing integration
between vector storage, embeddings, and search quality with realistic data.

Key testing areas:
- Vector store initialization with real-like configuration
- Document indexing and retrieval workflows
- Hybrid search with dense + sparse embeddings
- Search result ranking and relevance validation
- Query processing with different embedding types
- Performance characteristics with realistic data volumes
- Cache behavior and optimization patterns
"""

from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from src.retrieval.vector_store import (
    EmbeddingConfig,
    QdrantUnifiedVectorStore,
    RetrievalConfig,
    create_unified_qdrant_store,
)


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient for integration testing with realistic responses."""
    client = Mock()

    # Mock collection management operations
    collections_response = Mock()
    collections_response.collections = [
        Mock(name="existing_collection"),
        Mock(name="docmind_test_collection"),
    ]
    client.get_collections.return_value = collections_response
    client.create_collection.return_value = None
    client.delete_collection.return_value = None

    # Mock realistic search responses with varied relevance scores
    def create_search_response(limit: int = 10, base_score: float = 0.9) -> list[Any]:
        """Create realistic search response with diverse document content."""
        documents = [
            ("machine learning transforms document processing workflows", "ml.pdf", 1),
            (
                "artificial intelligence enhances semantic search capabilities",
                "ai.pdf",
                2,
            ),
            ("vector databases enable efficient similarity search", "vectors.pdf", 1),
            ("natural language processing improves text understanding", "nlp.pdf", 3),
            ("deep learning models provide better embeddings", "dl.pdf", 1),
            ("information retrieval systems handle large scale data", "ir.pdf", 2),
            ("knowledge graphs represent structured information", "kg.pdf", 1),
            ("semantic similarity enables better search results", "search.pdf", 4),
            ("document analysis requires advanced algorithms", "analysis.pdf", 2),
            ("text mining extracts insights from unstructured data", "mining.pdf", 1),
        ]

        results = []
        for i, (text, source, page) in enumerate(documents[:limit]):
            result = Mock()
            result.id = f"doc_{i}"
            result.score = max(0.1, base_score - (i * 0.05))  # Decreasing relevance
            result.payload = {
                "text": text,
                "metadata": {
                    "source": source,
                    "page": page,
                    "category": "technical" if i % 2 == 0 else "general",
                    "word_count": len(text.split()),
                },
                "node_id": f"node_{i}",
                "doc_id": f"document_{i}",
                "chunk_id": f"chunk_{i}",
            }
            results.append(result)
        return results

    # Configure search to return realistic results
    client.search.side_effect = lambda **kwargs: create_search_response(
        kwargs.get("limit", 10),
        0.95 if "dense" in str(kwargs.get("query_vector", "")) else 0.85,
    )

    # Mock CRUD operations
    client.upsert.return_value = None
    client.delete.return_value = None

    return client


@pytest.fixture
def test_documents():
    """Create realistic test documents for search integration testing."""
    documents_data = [
        {
            "text": "Machine learning algorithms are transforming document processing capabilities. "
            "Modern embedding models like BGE-M3 provide unified dense and sparse representations "
            "that enable more accurate semantic search across large document collections.",
            "metadata": {
                "source": "ml_overview.pdf",
                "page": 1,
                "section": "introduction",
            },
        },
        {
            "text": "Vector databases such as Qdrant offer efficient storage and retrieval of high-dimensional "
            "embeddings. They support both dense and sparse vectors, enabling hybrid search "
            "strategies that combine semantic similarity with keyword matching.",
            "metadata": {
                "source": "vector_db_guide.pdf",
                "page": 3,
                "section": "architecture",
            },
        },
        {
            "text": "Information retrieval systems must balance precision and recall when processing user queries. "
            "Reranking techniques and reciprocal rank fusion help improve search quality by "
            "combining multiple ranking signals from different retrieval strategies.",
            "metadata": {
                "source": "ir_best_practices.pdf",
                "page": 7,
                "section": "optimization",
            },
        },
        {
            "text": "Natural language processing techniques enable better understanding of document semantics. "
            "Advanced models can capture context, handle synonyms, and identify relevant "
            "passages even when query terms don't appear exactly in the source text.",
            "metadata": {
                "source": "nlp_techniques.pdf",
                "page": 12,
                "section": "applications",
            },
        },
        {
            "text": "Performance optimization in search systems requires careful tuning of embedding dimensions, "
            "batch sizes, and caching strategies. GPU acceleration can significantly improve "
            "throughput for both embedding computation and vector similarity calculations.",
            "metadata": {
                "source": "performance_guide.pdf",
                "page": 5,
                "section": "optimization",
            },
        },
    ]

    nodes = []
    for i, doc_data in enumerate(documents_data):
        # Include doc_id and chunk_id in metadata for compatibility
        metadata = doc_data["metadata"].copy()
        metadata["doc_id"] = f"integration_doc_{i}"
        metadata["chunk_id"] = f"integration_chunk_{i}"

        node = TextNode(
            text=doc_data["text"],
            metadata=metadata,
            id_=f"integration_node_{i}",
        )
        nodes.append(node)

    return nodes


@pytest.fixture
def test_embeddings():
    """Create realistic embeddings using MockEmbedding for consistency."""
    MockEmbedding(embed_dim=1024)

    # Create diverse embeddings that will produce different similarity scores
    dense_embeddings = []
    sparse_embeddings = []

    # Generate 5 diverse embeddings with different patterns
    for i in range(5):
        # Create dense embeddings with different patterns
        base_vector = np.random.randn(1024)
        # Add domain-specific patterns
        if i == 0:  # ML-focused
            base_vector[:100] += 0.5
        elif i == 1:  # Vector DB-focused
            base_vector[100:200] += 0.5
        elif i == 2:  # IR-focused
            base_vector[200:300] += 0.5

        dense_embeddings.append(base_vector.tolist())

        # Create sparse embeddings with domain-relevant tokens
        sparse_dict = {}
        base_tokens = [
            100 + i * 50,
            200 + i * 30,
            300 + i * 20,
        ]  # Different token patterns
        for j, token_id in enumerate(base_tokens):
            sparse_dict[token_id] = 0.8 - (j * 0.1)  # Decreasing weights

        sparse_embeddings.append(sparse_dict)

    return dense_embeddings, sparse_embeddings


@pytest.fixture
def integrated_vector_store(mock_qdrant_client):
    """Create integrated vector store for realistic testing scenarios."""
    with patch("src.retrieval.vector_store.QdrantClient") as mock_client_class:
        mock_client_class.return_value = mock_qdrant_client

        store = QdrantUnifiedVectorStore(
            collection_name="integration_test_collection",
            embedding_dim=1024,
            rrf_alpha=0.7,  # 70% dense, 30% sparse weighting
        )
        return store


@pytest.mark.integration
class TestVectorStoreSearchWorkflows:
    """Test complete search workflows from indexing to retrieval."""

    def test_document_indexing_workflow(
        self, integrated_vector_store, test_documents, test_embeddings
    ):
        """Test complete document indexing workflow with realistic data."""
        dense_embeddings, sparse_embeddings = test_embeddings

        # Index documents with both dense and sparse embeddings
        node_ids = integrated_vector_store.add(
            nodes=test_documents,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # Validate indexing results
        assert len(node_ids) == 5
        assert all(node_id.startswith("integration_node_") for node_id in node_ids)

        # Verify upsert was called with proper structure
        integrated_vector_store.qdrant_client.upsert.assert_called_once()
        upsert_call = integrated_vector_store.qdrant_client.upsert.call_args[1]

        assert upsert_call["collection_name"] == "integration_test_collection"
        points = upsert_call["points"]
        assert len(points) == 5

        # Validate point structure for search functionality
        for i, point in enumerate(points):
            assert point.id == f"integration_node_{i}"
            assert "dense" in point.vector
            assert "sparse" in point.vector
            assert point.payload["text"] == test_documents[i].text
            # Metadata should include the original plus doc_id and chunk_id
            expected_metadata = test_documents[i].metadata.copy()
            expected_metadata.update(
                {
                    "doc_id": f"integration_doc_{i}",
                    "chunk_id": f"integration_chunk_{i}",
                }
            )
            assert point.payload["metadata"] == expected_metadata

    def test_semantic_search_quality(self, integrated_vector_store):
        """Test semantic search quality with domain-specific queries."""
        # Test ML-focused query
        ml_query = VectorStoreQuery(similarity_top_k=3)
        ml_embedding = np.random.randn(1024).tolist()

        results = integrated_vector_store.query(
            query=ml_query,
            dense_embedding=ml_embedding,
            sparse_embedding=None,
        )

        # Validate search results quality
        assert isinstance(results, VectorStoreQueryResult)
        assert len(results.nodes) == 3
        assert len(results.similarities) == 3
        assert len(results.ids) == 3

        # Verify results are ranked by relevance (descending)
        assert all(
            results.similarities[i] >= results.similarities[i + 1]
            for i in range(len(results.similarities) - 1)
        )

        # Verify result content quality
        for node in results.nodes:
            assert isinstance(node, TextNode)
            assert len(node.text) > 50  # Meaningful content
            assert node.metadata is not None

    def test_hybrid_search_fusion_workflow(self, integrated_vector_store):
        """Test hybrid search with RRF fusion using realistic embeddings."""
        query = VectorStoreQuery(similarity_top_k=5)

        # Create query embeddings that should match different aspects
        dense_query = np.random.randn(1024).tolist()
        sparse_query = {150: 0.9, 250: 0.7, 350: 0.6}  # Overlapping with indexed tokens

        results = integrated_vector_store.query(
            query=query,
            dense_embedding=dense_query,
            sparse_embedding=sparse_query,
        )

        # Validate hybrid search execution
        assert (
            integrated_vector_store.qdrant_client.search.call_count == 2
        )  # Dense + sparse

        # Verify fusion results quality
        assert len(results.nodes) == 5
        assert len(results.similarities) == 5

        # Check that RRF fusion was applied (scores should be fusion scores)
        for similarity in results.similarities:
            assert 0.0 < similarity <= 1.0

        # Verify search calls used correct parameters
        search_calls = integrated_vector_store.qdrant_client.search.call_args_list

        # First call: dense search with fusion limit
        dense_call = search_calls[0][1]
        assert dense_call["query_vector"][0] == "dense"
        assert dense_call["limit"] == 10  # 2x similarity_top_k for fusion

        # Second call: sparse search with fusion limit
        sparse_call = search_calls[1][1]
        assert sparse_call["query_vector"].name == "sparse"
        assert sparse_call["limit"] == 10

    def test_search_result_diversity(self, integrated_vector_store):
        """Test search returns diverse, relevant results."""
        query = VectorStoreQuery(similarity_top_k=4)
        dense_embedding = np.random.randn(1024).tolist()

        results = integrated_vector_store.query(
            query=query,
            dense_embedding=dense_embedding,
        )

        # Verify result diversity
        assert len(results.nodes) == 4

        # Check content diversity (no exact duplicates)
        texts = [node.text for node in results.nodes]
        assert len(set(texts)) == len(texts)  # All unique

        # Check metadata diversity
        sources = [node.metadata.get("source", "") for node in results.nodes]
        assert len(set(sources)) >= 2  # At least 2 different sources

        # Verify quality scoring
        similarities = results.similarities
        assert max(similarities) >= 0.5  # At least one highly relevant result
        assert all(sim >= 0.1 for sim in similarities)  # All results minimally relevant

    def test_query_processing_edge_cases(self, integrated_vector_store):
        """Test query processing handles various edge cases gracefully."""
        # Test very small similarity_top_k
        small_query = VectorStoreQuery(similarity_top_k=1)
        results = integrated_vector_store.query(
            query=small_query,
            dense_embedding=np.random.randn(1024).tolist(),
        )
        assert len(results.nodes) == 1

        # Test large similarity_top_k (should be handled gracefully)
        large_query = VectorStoreQuery(similarity_top_k=50)
        results = integrated_vector_store.query(
            query=large_query,
            dense_embedding=np.random.randn(1024).tolist(),
        )
        assert len(results.nodes) > 0  # Should return available results

        # Test query with only sparse embedding
        sparse_only_query = VectorStoreQuery(similarity_top_k=3)
        results = integrated_vector_store.query(
            query=sparse_only_query,
            sparse_embedding={100: 0.8, 200: 0.6},
        )
        assert isinstance(results, VectorStoreQueryResult)
        assert len(results.nodes) >= 0

    def test_embedding_dimension_validation_workflow(self, integrated_vector_store):
        """Test embedding dimension validation in realistic workflows."""
        query = VectorStoreQuery(similarity_top_k=3)

        # Test correct 1024-dimensional embedding
        correct_embedding = np.random.randn(1024).tolist()
        results = integrated_vector_store.query(
            query=query,
            dense_embedding=correct_embedding,
        )

        assert isinstance(results, VectorStoreQueryResult)

        # Verify the search used the correct embedding dimension
        search_call = integrated_vector_store.qdrant_client.search.call_args[1]
        query_vector = search_call["query_vector"]
        assert len(query_vector[1]) == 1024  # Correct dimension passed through


@pytest.mark.integration
class TestVectorStoreConfiguration:
    """Test vector store configuration and initialization workflows."""

    def test_vector_store_initialization_workflow(self, mock_qdrant_client):
        """Test complete initialization workflow with custom configuration."""
        with patch("src.retrieval.vector_store.QdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            # Test custom configuration
            custom_config = {
                "url": "http://test:6333",
                "collection_name": "test_workflow_collection",
                "embedding_dim": 1024,
                "rrf_alpha": 0.8,  # Favor dense search
            }

            store = QdrantUnifiedVectorStore(**custom_config)

            # Verify configuration applied correctly
            assert store.collection_name == "test_workflow_collection"
            assert store.embedding_dim == 1024
            assert store.rrf_alpha == 0.8
            assert store.dense_vector_name == "dense"
            assert store.sparse_vector_name == "sparse"

            # Verify collection initialization was attempted
            mock_qdrant_client.get_collections.assert_called()

    def test_factory_function_integration(self, mock_qdrant_client):
        """Test factory function creates properly configured vector store."""
        with patch("src.retrieval.vector_store.QdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_qdrant_client

            # Test factory with custom parameters
            store = create_unified_qdrant_store(
                url="http://factory_test:6333",
                collection_name="factory_test_collection",
                embedding_dim=1024,
                rrf_alpha=0.6,
            )

            assert isinstance(store, QdrantUnifiedVectorStore)
            assert store.collection_name == "factory_test_collection"
            assert store.rrf_alpha == 0.6

    def test_configuration_classes_integration(self):
        """Test configuration classes work correctly in integration context."""
        # Test RetrievalConfig with realistic parameters
        retrieval_config = RetrievalConfig(
            strategy="hybrid",
            top_k=15,
            use_reranking=True,
            rrf_alpha=0.75,
            rrf_k_constant=45,
        )

        assert retrieval_config.strategy == "hybrid"
        assert retrieval_config.top_k == 15
        assert retrieval_config.rrf_alpha == 0.75

        # Test EmbeddingConfig with BGE-M3 settings
        embedding_config = EmbeddingConfig(
            model_name="BAAI/bge-m3",
            dimension=1024,
            max_length=8192,
            batch_size_gpu=16,
        )

        assert embedding_config.model_name == "BAAI/bge-m3"
        assert embedding_config.dimension == 1024
        assert embedding_config.batch_size_gpu == 16


@pytest.mark.integration
class TestSearchPerformanceWorkflows:
    """Test search performance characteristics with realistic data volumes."""

    def test_batch_indexing_performance(self, integrated_vector_store):
        """Test batch indexing performance with larger document sets."""
        # Create batch of 20 documents
        batch_nodes = []
        for i in range(20):
            node = TextNode(
                text=f"Performance test document {i} with substantial content " * 10,
                metadata={
                    "batch": "performance_test",
                    "doc_id": f"perf_doc_{i}",
                    "chunk_id": f"perf_chunk_{i}",
                },
                id_=f"perf_node_{i}",
            )
            batch_nodes.append(node)

        # Create corresponding embeddings
        batch_embeddings = [np.random.randn(1024).tolist() for _ in range(20)]

        # Test batch indexing
        node_ids = integrated_vector_store.add(
            nodes=batch_nodes,
            dense_embeddings=batch_embeddings,
        )

        assert len(node_ids) == 20

        # Verify single batch operation (performance optimization)
        integrated_vector_store.qdrant_client.upsert.assert_called_once()
        points = integrated_vector_store.qdrant_client.upsert.call_args[1]["points"]
        assert len(points) == 20

    def test_fusion_limit_optimization(self, integrated_vector_store):
        """Test fusion limit optimization for performance."""
        # Test with different top_k values to verify optimization
        test_cases = [
            (5, 10),  # Normal case: 2x top_k
            (25, 50),  # At max fusion limit
            (30, 50),  # Above max fusion limit (should cap at 50)
        ]

        for top_k, expected_limit in test_cases:
            integrated_vector_store.qdrant_client.reset_mock()

            query = VectorStoreQuery(similarity_top_k=top_k)
            dense_embedding = np.random.randn(1024).tolist()
            sparse_embedding = {100: 0.8}

            integrated_vector_store._hybrid_search(
                query, dense_embedding, sparse_embedding
            )

            # Check fusion limit was applied correctly
            search_calls = integrated_vector_store.qdrant_client.search.call_args_list
            actual_limit = search_calls[0][1]["limit"]

            assert actual_limit == expected_limit, (
                f"top_k={top_k}, expected={expected_limit}, actual={actual_limit}"
            )

    def test_search_caching_behavior(self, integrated_vector_store):
        """Test search behavior that would benefit from caching."""
        query = VectorStoreQuery(similarity_top_k=5)
        dense_embedding = np.random.randn(1024).tolist()

        # Execute same search multiple times
        results1 = integrated_vector_store.query(
            query=query, dense_embedding=dense_embedding
        )
        results2 = integrated_vector_store.query(
            query=query, dense_embedding=dense_embedding
        )

        # Both should return valid results
        assert len(results1.nodes) == len(results2.nodes)
        assert isinstance(results1, VectorStoreQueryResult)
        assert isinstance(results2, VectorStoreQueryResult)

        # Each query should trigger search (no caching at vector store level)
        assert integrated_vector_store.qdrant_client.search.call_count == 2


@pytest.mark.integration
class TestSearchErrorRecovery:
    """Test search error handling and recovery in integrated workflows."""

    def test_search_resilience_workflow(self, integrated_vector_store):
        """Test search handles connection errors gracefully."""
        # Configure client to fail then succeed
        integrated_vector_store.qdrant_client.search.side_effect = [
            ConnectionError("Connection lost"),
            [
                Mock(
                    id="doc_1",
                    score=0.9,
                    payload={
                        "text": "Recovery test",
                        "metadata": {},
                        "node_id": "node_1",
                    },
                )
            ],
        ]

        query = VectorStoreQuery(similarity_top_k=1)
        dense_embedding = np.random.randn(1024).tolist()

        results = integrated_vector_store.query(
            query=query, dense_embedding=dense_embedding
        )

        # Should return empty results on error (graceful degradation)
        assert isinstance(results, VectorStoreQueryResult)
        assert len(results.nodes) == 0  # Empty due to error

    def test_malformed_query_recovery(self, integrated_vector_store):
        """Test recovery from malformed queries."""
        # Test with None embeddings (should handle gracefully)
        query = VectorStoreQuery(similarity_top_k=5)

        results = integrated_vector_store.query(
            query=query,
            dense_embedding=None,
            sparse_embedding=None,
        )

        # Should return empty results gracefully
        assert isinstance(results, VectorStoreQueryResult)
        assert len(results.nodes) == 0
        assert len(results.similarities) == 0
        assert len(results.ids) == 0

        # No search should be attempted
        integrated_vector_store.qdrant_client.search.assert_not_called()

    def test_partial_embedding_handling(self, integrated_vector_store, test_documents):
        """Test handling of partial or mismatched embeddings."""
        # Test with fewer embeddings than nodes
        dense_embeddings = [
            np.random.randn(1024).tolist() for _ in range(2)
        ]  # Only 2 embeddings
        sparse_embeddings = [{100: 0.8}]  # Only 1 sparse embedding

        node_ids = integrated_vector_store.add(
            nodes=test_documents[:3],  # 3 nodes
            dense_embeddings=dense_embeddings,  # 2 dense
            sparse_embeddings=sparse_embeddings,  # 1 sparse
        )

        # Should handle mismatched counts gracefully
        assert len(node_ids) == 3

        # Verify points were created appropriately
        points = integrated_vector_store.qdrant_client.upsert.call_args[1]["points"]
        assert len(points) == 3

        # First two points should have dense embeddings
        assert "dense" in points[0].vector
        assert "dense" in points[1].vector

        # Third point should not have dense embedding (graceful handling)
        assert points[2].vector == {} or "dense" not in points[2].vector


@pytest.mark.integration
class TestSearchQualityValidation:
    """Test search quality and relevance validation workflows."""

    def test_result_relevance_validation(self, integrated_vector_store):
        """Test search results maintain relevance ordering."""
        query = VectorStoreQuery(similarity_top_k=5)
        dense_embedding = np.random.randn(1024).tolist()

        results = integrated_vector_store.query(
            query=query,
            dense_embedding=dense_embedding,
        )

        # Validate relevance ordering
        similarities = results.similarities
        assert len(similarities) == 5

        # Should be in descending order (most relevant first)
        for i in range(len(similarities) - 1):
            assert similarities[i] >= similarities[i + 1], (
                f"Relevance ordering broken at index {i}"
            )

        # Top result should be significantly relevant
        assert similarities[0] >= 0.5, "Top result should have high relevance"

    def test_metadata_preservation_workflow(
        self, integrated_vector_store, test_documents, test_embeddings
    ):
        """Test metadata is preserved through complete indexing and retrieval workflow."""
        dense_embeddings, sparse_embeddings = test_embeddings

        # Index documents with rich metadata
        integrated_vector_store.add(
            nodes=test_documents,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # Search and verify metadata preservation
        query = VectorStoreQuery(similarity_top_k=3)
        results = integrated_vector_store.query(
            query=query,
            dense_embedding=np.random.randn(1024).tolist(),
        )

        # Validate metadata was preserved
        for node in results.nodes:
            assert isinstance(node.metadata, dict)
            assert len(node.metadata) > 0

            # Check for expected metadata fields
            if node.metadata.get("source"):
                assert node.metadata["source"].endswith(".pdf")
            if "page" in node.metadata:
                assert isinstance(node.metadata["page"], int)
                assert node.metadata["page"] > 0

    def test_text_content_quality(self, integrated_vector_store):
        """Test search returns quality text content."""
        query = VectorStoreQuery(similarity_top_k=3)
        dense_embedding = np.random.randn(1024).tolist()

        results = integrated_vector_store.query(
            query=query,
            dense_embedding=dense_embedding,
        )

        # Validate text content quality
        for node in results.nodes:
            assert isinstance(node.text, str)
            assert len(node.text.strip()) > 20  # Meaningful content length
            assert not node.text.strip().startswith(
                "Document "
            )  # Not just template text

            # Check for realistic content patterns
            word_count = len(node.text.split())
            assert word_count >= 5, "Content should have multiple words"

    def test_search_consistency_validation(self, integrated_vector_store):
        """Test search results are consistent across similar queries."""
        base_embedding = np.random.randn(1024).tolist()
        query = VectorStoreQuery(similarity_top_k=3)

        # Run same query multiple times
        results1 = integrated_vector_store.query(
            query=query, dense_embedding=base_embedding
        )
        results2 = integrated_vector_store.query(
            query=query, dense_embedding=base_embedding
        )

        # Results should be consistent (same IDs, same order)
        assert len(results1.ids) == len(results2.ids)
        assert results1.ids == results2.ids, "Search results should be consistent"

        # Similarities should match
        assert len(results1.similarities) == len(results2.similarities)
        for sim1, sim2 in zip(
            results1.similarities, results2.similarities, strict=False
        ):
            assert abs(sim1 - sim2) < 0.001, "Similarity scores should be consistent"
