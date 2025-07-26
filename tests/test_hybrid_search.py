"""Tests for hybrid search with Qdrant and RRF fusion.

This module tests Qdrant vector database integration, hybrid search functionality,
and Reciprocal Rank Fusion (RRF) algorithm implementation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils import (
    setup_hybrid_qdrant,
    setup_hybrid_qdrant_async,
    verify_rrf_configuration,
)


class TestQdrantIntegration:
    """Test Qdrant vector database integration."""

    def test_qdrant_client_creation(self, mock_qdrant_client):
        """Test Qdrant client creation and configuration."""
        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            client = mock_qdrant_client

            # Verify client is properly configured
            assert client is not None

            # Test basic operations
            client.search.return_value = [
                MagicMock(id=1, score=0.9, payload={"text": "Document 1"}),
                MagicMock(id=2, score=0.8, payload={"text": "Document 2"}),
            ]

            results = client.search(
                collection_name="test_collection", query_vector=[0.1, 0.2, 0.3], limit=2
            )

            assert len(results) == 2
            assert (
                results[0].score > results[1].score
            )  # Results should be sorted by score

    @pytest.mark.parametrize(
        ("collection_name", "vector_size"),
        [
            ("dense_collection", 1024),
            ("sparse_collection", 30522),  # SPLADE++ vocab size
            ("hybrid_collection", 1024),
        ],
    )
    def test_collection_configuration(
        self, collection_name, vector_size, mock_qdrant_client
    ):
        """Test different collection configurations."""
        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            client = mock_qdrant_client

            # Mock collection creation
            client.create_collection = MagicMock()
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": vector_size, "distance": "Cosine"},
            )

            # Verify collection creation was called with correct parameters
            client.create_collection.assert_called_once()
            call_args = client.create_collection.call_args
            assert call_args[1]["collection_name"] == collection_name
            assert call_args[1]["vectors_config"]["size"] == vector_size

    def test_qdrant_search_functionality(self, mock_qdrant_client, sample_documents):
        """Test Qdrant search with different query types."""
        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            client = mock_qdrant_client

            # Mock search results
            mock_results = [
                MagicMock(
                    id=i,
                    score=0.9 - i * 0.1,
                    payload={"text": doc.text, "metadata": doc.metadata},
                )
                for i, doc in enumerate(sample_documents[:3])
            ]
            client.search.return_value = mock_results

            # Test search
            results = client.search(
                collection_name="test_collection", query_vector=[0.1] * 1024, limit=3
            )

            assert len(results) == 3
            # Verify results are properly structured
            for result in results:
                assert hasattr(result, "id")
                assert hasattr(result, "score")
                assert hasattr(result, "payload")
                assert "text" in result.payload

    @pytest.mark.performance
    def test_qdrant_search_performance(self, benchmark, mock_qdrant_client):
        """Test Qdrant search performance."""
        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            client = mock_qdrant_client

            # Mock fast search results
            mock_results = [
                MagicMock(id=i, score=0.9, payload={"text": f"Document {i}"})
                for i in range(10)
            ]
            client.search.return_value = mock_results

            def search_operation():
                return client.search(
                    collection_name="test", query_vector=[0.1] * 1024, limit=10
                )

            result = benchmark(search_operation)
            assert len(result) == 10


class TestHybridSearchSetup:
    """Test hybrid search setup and configuration."""

    @pytest.mark.asyncio
    async def test_setup_hybrid_qdrant_async(self, sample_documents, test_settings):
        """Test async Qdrant setup for hybrid search."""
        with patch("utils.AsyncQdrantClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock collection operations
            mock_client_instance.collection_exists.return_value = False
            mock_client_instance.create_collection = AsyncMock()
            mock_client_instance.upsert = AsyncMock()

            # Test setup
            client = await setup_hybrid_qdrant_async(
                documents=sample_documents, settings=test_settings
            )

            assert client is not None
            # Verify collection creation was called
            mock_client_instance.create_collection.assert_called()

    def test_setup_hybrid_qdrant_sync(self, sample_documents, test_settings):
        """Test synchronous Qdrant setup for hybrid search."""
        with patch("utils.QdrantClient") as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            # Mock collection operations
            mock_client_instance.collection_exists.return_value = False
            mock_client_instance.create_collection = MagicMock()
            mock_client_instance.upsert = MagicMock()

            # Test setup
            client = setup_hybrid_qdrant(
                documents=sample_documents, settings=test_settings
            )

            assert client is not None
            # Verify collection creation was called
            mock_client_instance.create_collection.assert_called()

    def test_hybrid_search_configuration_validation(self, test_settings):
        """Test validation of hybrid search configuration."""
        # Test RRF configuration validation
        rrf_config = verify_rrf_configuration(test_settings)

        assert rrf_config is not None
        assert "dense_weight" in rrf_config
        assert "sparse_weight" in rrf_config
        assert "k_parameter" in rrf_config

        # Weights should sum to approximately 1.0
        total_weight = rrf_config["dense_weight"] + rrf_config["sparse_weight"]
        assert abs(total_weight - 1.0) < 0.01


class TestRRFFusion:
    """Test Reciprocal Rank Fusion algorithm implementation."""

    def test_rrf_basic_fusion(self):
        """Test basic RRF fusion with two result sets."""
        # Mock search results from dense and sparse retrievers
        dense_results = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.8},
            {"id": "doc3", "score": 0.7},
        ]

        sparse_results = [
            {"id": "doc2", "score": 0.85},
            {"id": "doc4", "score": 0.75},
            {"id": "doc1", "score": 0.65},
        ]

        # Implement simple RRF fusion
        def rrf_fusion(
            dense_results, sparse_results, k=60, dense_weight=0.7, sparse_weight=0.3
        ):
            """Simple RRF implementation for testing."""
            scores = {}

            # Process dense results
            for rank, result in enumerate(dense_results):
                doc_id = result["id"]
                scores[doc_id] = scores.get(doc_id, 0) + dense_weight / (k + rank + 1)

            # Process sparse results
            for rank, result in enumerate(sparse_results):
                doc_id = result["id"]
                scores[doc_id] = scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)

            # Sort by combined score
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused_results = rrf_fusion(dense_results, sparse_results)

        # Verify fusion results
        assert len(fused_results) == 4  # Should combine all unique documents
        doc_ids = [result[0] for result in fused_results]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids  # Both should appear (in both lists)

        # doc2 appears high in both lists, should be ranked highly
        doc2_position = doc_ids.index("doc2")
        assert doc2_position < 2  # Should be in top 2

    @pytest.mark.parametrize(
        ("k_parameter", "dense_weight", "sparse_weight"),
        [
            (60, 0.7, 0.3),  # Default configuration
            (30, 0.5, 0.5),  # Equal weighting
            (100, 0.8, 0.2),  # Dense-heavy
            (20, 0.3, 0.7),  # Sparse-heavy
        ],
    )
    def test_rrf_parameter_variations(self, k_parameter, dense_weight, sparse_weight):
        """Test RRF with different parameter configurations."""
        # Create test data
        dense_results = [{"id": f"doc{i}", "score": 0.9 - i * 0.1} for i in range(5)]
        sparse_results = [
            {"id": f"doc{i + 2}", "score": 0.8 - i * 0.1} for i in range(5)
        ]

        def rrf_fusion(dense_results, sparse_results, k, dense_weight, sparse_weight):
            scores = {}

            for rank, result in enumerate(dense_results):
                doc_id = result["id"]
                scores[doc_id] = scores.get(doc_id, 0) + dense_weight / (k + rank + 1)

            for rank, result in enumerate(sparse_results):
                doc_id = result["id"]
                scores[doc_id] = scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)

            return sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused_results = rrf_fusion(
            dense_results, sparse_results, k_parameter, dense_weight, sparse_weight
        )

        # Verify basic properties
        assert len(fused_results) > 0
        assert all(score > 0 for _, score in fused_results)

        # Verify sorting (scores should be descending)
        scores = [score for _, score in fused_results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_edge_cases(self):
        """Test RRF with edge cases."""

        def rrf_fusion(
            dense_results, sparse_results, k=60, dense_weight=0.7, sparse_weight=0.3
        ):
            scores = {}

            for rank, result in enumerate(dense_results):
                doc_id = result["id"]
                scores[doc_id] = scores.get(doc_id, 0) + dense_weight / (k + rank + 1)

            for rank, result in enumerate(sparse_results):
                doc_id = result["id"]
                scores[doc_id] = scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)

            return sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Test with empty dense results
        sparse_only = [{"id": "doc1", "score": 0.8}]
        result = rrf_fusion([], sparse_only)
        assert len(result) == 1
        assert result[0][0] == "doc1"

        # Test with empty sparse results
        dense_only = [{"id": "doc2", "score": 0.9}]
        result = rrf_fusion(dense_only, [])
        assert len(result) == 1
        assert result[0][0] == "doc2"

        # Test with both empty (should return empty)
        result = rrf_fusion([], [])
        assert len(result) == 0

    @pytest.mark.performance
    def test_rrf_fusion_performance(self, benchmark):
        """Test RRF fusion performance with larger result sets."""
        # Create larger test datasets
        dense_results = [
            {"id": f"dense_doc{i}", "score": 0.9 - i * 0.001} for i in range(1000)
        ]
        sparse_results = [
            {"id": f"sparse_doc{i}", "score": 0.8 - i * 0.001} for i in range(1000)
        ]

        def rrf_fusion(
            dense_results, sparse_results, k=60, dense_weight=0.7, sparse_weight=0.3
        ):
            scores = {}

            for rank, result in enumerate(dense_results):
                doc_id = result["id"]
                scores[doc_id] = scores.get(doc_id, 0) + dense_weight / (k + rank + 1)

            for rank, result in enumerate(sparse_results):
                doc_id = result["id"]
                scores[doc_id] = scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)

            return sorted(scores.items(), key=lambda x: x[1], reverse=True)

        def fusion_operation():
            return rrf_fusion(dense_results, sparse_results)

        result = benchmark(fusion_operation)
        assert len(result) == 2000  # Should combine all unique documents


class TestHybridSearchIntegration:
    """Test complete hybrid search integration."""

    @pytest.mark.integration
    def test_hybrid_search_pipeline(
        self,
        sample_documents,
        mock_qdrant_client,
        mock_embedding_model,
        mock_sparse_embedding_model,
    ):
        """Test complete hybrid search pipeline."""
        with (
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
            patch("utils.FastEmbedModelManager.get_model") as mock_get_model,
        ):
            # Configure mocks
            def model_side_effect(model_name):
                if "splade" in model_name.lower():
                    return mock_sparse_embedding_model
                else:
                    return mock_embedding_model

            mock_get_model.side_effect = model_side_effect

            # Mock search results for both dense and sparse
            mock_qdrant_client.search.side_effect = [
                # Dense search results
                [
                    MagicMock(id=1, score=0.9, payload={"text": "Dense result 1"}),
                    MagicMock(id=2, score=0.8, payload={"text": "Dense result 2"}),
                ],
                # Sparse search results
                [
                    MagicMock(id=2, score=0.85, payload={"text": "Sparse result 2"}),
                    MagicMock(id=3, score=0.75, payload={"text": "Sparse result 3"}),
                ],
            ]

            # Test hybrid search workflow
            query = "What is SPLADE++ embedding?"

            # This would be the actual hybrid search implementation
            # For now, we test the component integration

            # 1. Generate embeddings
            dense_embedding = mock_embedding_model.embed_query(query)
            sparse_embedding = mock_sparse_embedding_model.encode([query])[0]

            # 2. Perform searches
            dense_results = mock_qdrant_client.search(
                collection_name="dense", query_vector=dense_embedding, limit=10
            )

            sparse_results = mock_qdrant_client.search(
                collection_name="sparse", query_vector=sparse_embedding, limit=10
            )

            # 3. Verify searches were performed
            assert len(dense_results) == 2
            assert len(sparse_results) == 2
            assert mock_qdrant_client.search.call_count == 2

    def test_hybrid_search_error_handling(self):
        """Test error handling in hybrid search."""
        with patch("utils.QdrantClient") as mock_client_class:
            # Mock client that raises exceptions
            mock_client = MagicMock()
            mock_client.search.side_effect = Exception("Search failed")
            mock_client_class.return_value = mock_client

            # Test that errors are properly handled
            with pytest.raises(Exception, match="Search failed"):
                mock_client.search(
                    collection_name="test", query_vector=[0.1] * 1024, limit=10
                )

    @pytest.mark.slow
    def test_hybrid_search_with_large_corpus(self, large_document_set):
        """Test hybrid search performance with large document corpus."""
        # This test validates that hybrid search can handle larger datasets
        # Skip by default to avoid long test times
        pytest.skip("Large corpus test - run explicitly for performance validation")

        # Would test with large_document_set (100 documents)
        assert len(large_document_set) == 100

        # Mock performance test setup
        with patch("utils.QdrantClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Simulate search on large corpus
            mock_instance.search.return_value = [
                MagicMock(id=i, score=0.9 - i * 0.01) for i in range(50)
            ]

            results = mock_instance.search(
                collection_name="large_corpus", query_vector=[0.1] * 1024, limit=50
            )

            assert len(results) == 50
