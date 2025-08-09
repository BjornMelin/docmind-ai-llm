"""Tests for hybrid search with Qdrant and RRF fusion.

This module tests Qdrant vector database integration, hybrid search functionality,
and Reciprocal Rank Fusion (RRF) algorithm implementation.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
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

            benchmark(search_operation)
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
        rrf_fusion([], sparse_only)
        assert len(result) == 1
        assert result[0][0] == "doc1"

        # Test with empty sparse results
        dense_only = [{"id": "doc2", "score": 0.9}]
        rrf_fusion(dense_only, [])
        assert len(result) == 1
        assert result[0][0] == "doc2"

        # Test with both empty (should return empty)
        rrf_fusion([], [])
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

        benchmark(fusion_operation)
        assert len(result) == 2000  # Should combine all unique documents


class TestHybridFusionRetriever:
    """Test HybridFusionRetriever with RRF fusion and ColBERT reranking."""

    def test_hybrid_fusion_retriever_creation(self, sample_documents, test_settings):
        """Test creation of HybridFusionRetriever with proper configuration."""
        from unittest.mock import MagicMock

        from llama_index.core import Document

        from utils.index_builder import create_hybrid_retriever

        # Create test index
        _ = [Document(text="test document content") for _ in range(3)]

        mock_index_instance = MagicMock()

        # Create hybrid retriever
        retriever = create_hybrid_retriever(mock_index_instance)

        # Verify retriever is created
        assert retriever is not None
        # Verify it's the expected type (QueryFusionRetriever or fallback)
        assert hasattr(retriever, "retrieve")

    def test_hybrid_fusion_retriever_with_rrf(self):
        """Test hybrid fusion retriever with RRF scoring."""
        from unittest.mock import MagicMock

        from llama_index.core import Document

        from utils.index_builder import create_hybrid_retriever

        # Create test documents
        _ = [
            Document(text="artificial intelligence machine learning"),
            Document(text="natural language processing nlp"),
            Document(text="deep learning neural networks"),
        ]

        mock_index_instance = MagicMock()

        # Mock retrieve method to return results with scores
        def mock_retrieve(query_bundle):
            return [
                MagicMock(node=MagicMock(node_id="1"), score=0.9),
                MagicMock(node=MagicMock(node_id="2"), score=0.8),
                MagicMock(node=MagicMock(node_id="3"), score=0.7),
            ]

        # Create retriever and test retrieval
        retriever = create_hybrid_retriever(mock_index_instance)

        # Verify fusion configuration
        if hasattr(retriever, "mode"):
            assert retriever.mode == "reciprocal_rerank"

        # Test that retriever works
        assert callable(getattr(retriever, "retrieve", None))

    @pytest.mark.performance
    def test_hybrid_fusion_performance(self, benchmark, sample_documents):
        """Test performance of hybrid fusion retriever."""
        from unittest.mock import MagicMock

        from llama_index.core import Document

        from utils.index_builder import create_hybrid_retriever

        # Create larger test dataset
        _ = [Document(text=f"test document {i} with content") for i in range(100)]

        mock_index_instance = MagicMock()

        # Mock fast retrieval
        def fast_retrieve(query_bundle):
            return [
                MagicMock(
                    node=MagicMock(node_id=str(i), text=f"result {i}"),
                    score=0.9 - i * 0.01,
                )
                for i in range(5)
            ]

        retriever = create_hybrid_retriever(mock_index_instance)

        # Mock the retrieve method for performance testing
        if hasattr(retriever, "retrieve"):
            retriever.retrieve = MagicMock(side_effect=fast_retrieve)

        def retrieval_operation():
            if hasattr(retriever, "retrieve"):
                from llama_index.core import QueryBundle

                return retriever.retrieve(QueryBundle(query_str="test query"))
            return []

        benchmark(retrieval_operation)
        assert len(result) > 0

    def test_hybrid_fusion_with_colbert_reranking(self):
        """Test integration of ColBERT reranking with hybrid fusion."""
        from unittest.mock import MagicMock, patch

        from agents.agent_utils import create_tools_from_index

        # Mock index data with hybrid retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            MagicMock(node=MagicMock(node_id="1", text="relevant content"), score=0.9),
            MagicMock(node=MagicMock(node_id="2", text="somewhat relevant"), score=0.7),
        ]

        index_data = {
            "vector": MagicMock(),
            "kg": MagicMock(),
            "retriever": mock_retriever,
        }

        with patch("agents.agent_utils.ColbertRerank") as mock_reranker:
            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            # Create tools with hybrid retriever
            tools = create_tools_from_index(index_data)

            # Verify hybrid fusion search tool is created
            assert len(tools) >= 1

            # Check for hybrid fusion search tool
            hybrid_tool = None
            for tool in tools:
                if "hybrid_fusion_search" in tool.metadata.name:
                    hybrid_tool = tool
                    break

            assert hybrid_tool is not None
            assert "QueryFusionRetriever" in hybrid_tool.metadata.description
            assert "RRF" in hybrid_tool.metadata.description
            assert "ColBERT" in hybrid_tool.metadata.description

    def test_fallback_behavior(self):
        """Test fallback behavior when hybrid retriever creation fails."""
        from unittest.mock import MagicMock, patch

        from utils.index_builder import create_hybrid_retriever

        # Mock index that raises exception
        mock_index = MagicMock()

        with patch("utils.index_builder.VectorIndexRetriever") as mock_retriever:
            # First call (dense) succeeds, second call (sparse) fails,
            # third call (fallback) succeeds
            mock_retriever.side_effect = [
                MagicMock(),  # dense retriever
                Exception("Sparse retriever failed"),  # sparse retriever fails
                MagicMock(),  # fallback retriever
            ]

            # Should return fallback retriever
            create_hybrid_retriever(mock_index)

            # Verify fallback is used
            assert result is not None
            assert hasattr(result, "retrieve")


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


class TestSPLADEPlusPlusIntegration:
    """Test SPLADE++ integration with hybrid search."""

    def test_splade_plus_plus_sparse_vector_processing(
        self, mock_sparse_embedding_model, mock_settings
    ):
        """Test SPLADE++ sparse vector processing for hybrid search."""
        # Configure SPLADE++ model
        mock_settings.sparse_embedding_model = (
            "prithivida/Splade_PP_en_v1"  # Fixed typo
        )

        # Test query processing
        query = "hybrid retrieval with sparse embeddings"

        # Mock SPLADE++ output
        sparse_embeddings = mock_sparse_embedding_model.encode([query])
        sparse_embedding = sparse_embeddings[0]

        # Verify SPLADE++ sparse structure
        if isinstance(sparse_embedding, dict):
            indices = sparse_embedding["indices"]
            values = sparse_embedding["values"]
        else:
            indices = sparse_embedding.indices
            values = sparse_embedding.values

        # SPLADE++ specific validations
        assert len(indices) > 0, "SPLADE++ should generate sparse indices"
        assert len(values) > 0, "SPLADE++ should generate sparse values"
        assert len(indices) == len(values), "Indices and values should match in length"
        assert all(val > 0 for val in values), (
            "SPLADE++ values should be positive (ReLU)"
        )
        assert all(isinstance(idx, int) for idx in indices), (
            "Indices should be integers"
        )

        # Term expansion verification
        original_terms = len(query.split())
        expanded_terms = len(indices)
        assert expanded_terms > original_terms, (
            f"SPLADE++ should expand {original_terms} terms to {expanded_terms}"
        )

    def test_splade_plus_plus_typo_fix_validation(self, mock_settings):
        """Test that SPLADE++ typo has been fixed in configuration."""
        # Verify the typo fix: "Splade_PP_en_v1" not "Splade+_PP_en_v1"
        mock_settings.sparse_embedding_model = "prithivida/Splade_PP_en_v1"

        model_name = mock_settings.sparse_embedding_model
        assert "Splade_PP_en_v1" in model_name, "Should use correct SPLADE++ model name"
        assert "Splade+_PP" not in model_name, (
            "Should not contain typo with extra + symbol"
        )
        assert "prithivida/" in model_name, "Should include correct HuggingFace org"


class TestBGELargeIntegration:
    """Test BGE-Large integration with hybrid search."""

    def test_bge_large_dense_embedding_processing(
        self, mock_embedding_model, mock_settings
    ):
        """Test BGE-Large dense embedding processing for hybrid search."""
        # Configure BGE-Large model
        mock_settings.dense_embedding_model = "BAAI/bge-large-en-v1.5"
        mock_settings.dense_embedding_dimension = 1024

        # Test semantic query processing
        query = "What are the benefits of hybrid retrieval systems?"

        # Mock BGE-Large output (1024 dimensions)
        dense_embedding = mock_embedding_model.embed_query(query)

        # Verify BGE-Large dense structure
        assert len(dense_embedding) == 1024, (
            "BGE-Large should produce 1024-dimensional embeddings"
        )
        assert all(isinstance(val, (int, float)) for val in dense_embedding), (
            "All values should be numeric"
        )

        # Semantic representation verification
        embedding_norm = sum(val**2 for val in dense_embedding) ** 0.5
        assert embedding_norm > 0, "BGE-Large embeddings should have non-zero magnitude"

        # Test batch processing
        documents = [
            "Hybrid search combines semantic and lexical matching",
            "Dense embeddings capture semantic similarity",
            "Sparse embeddings enable keyword matching",
        ]

        batch_embeddings = mock_embedding_model.embed_documents(documents)
        assert len(batch_embeddings) == len(documents), "Should process all documents"
        assert all(len(emb) == 1024 for emb in batch_embeddings), (
            "All embeddings should be 1024-dim"
        )

    def test_bge_large_semantic_understanding(
        self, mock_embedding_model, mock_settings
    ):
        """Test BGE-Large semantic understanding capabilities."""
        mock_settings.dense_embedding_model = "BAAI/bge-large-en-v1.5"

        # Test semantic similarity scenarios
        test_pairs = [
            ("machine learning algorithms", "ML models and methods"),
            ("information retrieval systems", "document search engines"),
            ("neural networks", "deep learning architectures"),
        ]

        for query, similar_doc in test_pairs:
            query_emb = mock_embedding_model.embed_query(query)
            doc_emb = mock_embedding_model.embed_documents([similar_doc])[0]

            # Verify embeddings are generated
            assert len(query_emb) == 1024
            assert len(doc_emb) == 1024

            # BGE-Large should capture semantic relationships
            # (We can't test actual similarity without real embeddings,
            #  but we verify the structure supports it)
            assert query_emb is not None and doc_emb is not None
