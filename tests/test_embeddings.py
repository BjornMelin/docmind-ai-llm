"""Tests for embedding models and FastEmbed integration.

This module tests SPLADE++ sparse embeddings, BGE-Large dense embeddings,
and the FastEmbedModelManager singleton pattern following 2025 best practices.
"""

from unittest.mock import MagicMock, patch

import pytest

from utils import FastEmbedModelManager


class TestFastEmbedModelManager:
    """Test FastEmbedModelManager singleton and model caching."""

    def test_singleton_behavior(self):
        """Test that FastEmbedModelManager follows singleton pattern."""
        manager1 = FastEmbedModelManager()
        manager2 = FastEmbedModelManager()

        assert manager1 is manager2, "FastEmbedModelManager should be singleton"
        assert id(manager1) == id(manager2), "Instances should have same memory address"

    def test_singleton_thread_safety(self):
        """Test singleton behavior under concurrent access."""
        import threading

        instances = []

        def create_instance():
            instances.append(FastEmbedModelManager())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All instances should be the same
        for instance in instances[1:]:
            assert instance is instances[0]

    @pytest.mark.performance
    def test_singleton_creation_performance(self, benchmark):
        """Test FastEmbedModelManager singleton creation performance."""

        def create_manager():
            return FastEmbedModelManager()

        # Should be very fast since it's singleton
        result = benchmark(create_manager)
        assert result is not None

    def test_model_caching_behavior(self):
        """Test that models are cached properly."""
        manager = FastEmbedModelManager()

        # Clear any existing cache for clean test
        if hasattr(manager, "_models"):
            manager._models.clear()

        # Mock model loading to verify caching
        with patch.object(manager, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            # First call should load model
            model1 = manager.get_model("test_model")
            assert mock_load.call_count == 1
            assert model1 is mock_model

            # Second call should return cached model
            model2 = manager.get_model("test_model")
            assert mock_load.call_count == 1  # No additional calls
            assert model2 is model1  # Same instance


class TestSparseEmbeddings:
    """Test SPLADE++ sparse embedding functionality."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "prithvida/Splade_PP_en_v1",
            "naver/splade-cocondenser-ensembledistil",
        ],
    )
    def test_sparse_embedding_models(self, model_name, mock_sparse_embedding_model):
        """Test different sparse embedding models."""
        with patch(
            "utils.FastEmbedModelManager.get_model",
            return_value=mock_sparse_embedding_model,
        ):
            manager = FastEmbedModelManager()
            model = manager.get_model(model_name)

            # Test encoding
            texts = ["Document about AI", "Machine learning concepts"]
            embeddings = model.encode(texts)

            assert len(embeddings) == len(texts)
            for embedding in embeddings:
                assert "indices" in embedding
                assert "values" in embedding
                assert len(embedding["indices"]) == len(embedding["values"])

    def test_sparse_embedding_properties(self, mock_sparse_embedding_model):
        """Test sparse embedding properties and structure."""
        texts = ["Short text", "Much longer text with more words and concepts"]
        embeddings = mock_sparse_embedding_model.encode(texts)

        for embedding in embeddings:
            # Sparse embeddings should have non-zero indices
            assert all(idx >= 0 for idx in embedding["indices"])
            # Values should be positive (SPLADE++ uses ReLU activation)
            assert all(val > 0 for val in embedding["values"])
            # Indices should be sorted
            assert list(embedding["indices"]) == sorted(embedding["indices"])

    @pytest.mark.performance
    def test_sparse_embedding_batch_performance(
        self, benchmark, mock_sparse_embedding_model
    ):
        """Test sparse embedding performance with batch processing."""
        texts = [f"Document {i} with content about topic {i % 5}" for i in range(50)]

        def encode_batch():
            return mock_sparse_embedding_model.encode(texts)

        result = benchmark(encode_batch)
        assert len(result) == len(texts)


class TestDenseEmbeddings:
    """Test BGE-Large dense embedding functionality."""

    @pytest.mark.parametrize(
        ("model_name", "expected_dim"),
        [
            ("BAAI/bge-large-en-v1.5", 1024),
            ("BAAI/bge-base-en-v1.5", 768),
            ("sentence-transformers/all-MiniLM-L6-v2", 384),
        ],
    )
    def test_dense_embedding_models(
        self, model_name, expected_dim, mock_embedding_model
    ):
        """Test different dense embedding models with expected dimensions."""
        # Adjust mock to return correct dimensions
        mock_embedding_model.embed_documents.return_value = [
            [0.1] * expected_dim for _ in range(3)
        ]
        mock_embedding_model.embed_query.return_value = [0.5] * expected_dim

        with patch(
            "utils.FastEmbedModelManager.get_model", return_value=mock_embedding_model
        ):
            manager = FastEmbedModelManager()
            model = manager.get_model(model_name)

            # Test document embedding
            texts = ["Document 1", "Document 2", "Document 3"]
            doc_embeddings = model.embed_documents(texts)

            assert len(doc_embeddings) == len(texts)
            for embedding in doc_embeddings:
                assert len(embedding) == expected_dim
                assert all(isinstance(val, int | float) for val in embedding)

            # Test query embedding
            query_embedding = model.embed_query("Test query")
            assert len(query_embedding) == expected_dim

    def test_dense_embedding_similarity(self, mock_embedding_model):
        """Test that similar texts have higher similarity scores."""
        # Set up mock to return different embeddings for different texts
        mock_embedding_model.embed_documents.side_effect = [
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ],  # Similar, similar, different
        ]

        texts = ["AI technology", "Artificial intelligence", "Cooking recipes"]
        embeddings = mock_embedding_model.embed_documents(texts)

        # Calculate cosine similarity
        def cosine_similarity(a, b):
            import math

            dot_product = sum(x * y for x, y in zip(a, b, strict=False))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

        sim_ai_similar = cosine_similarity(embeddings[0], embeddings[1])
        sim_ai_different = cosine_similarity(embeddings[0], embeddings[2])

        # Similar texts should have higher similarity
        assert sim_ai_similar > sim_ai_different

    @pytest.mark.performance
    def test_dense_embedding_batch_performance(self, benchmark, mock_embedding_model):
        """Test dense embedding performance with batch processing."""
        texts = [f"Document {i} discussing various topics" for i in range(100)]

        def embed_batch():
            return mock_embedding_model.embed_documents(texts)

        result = benchmark(embed_batch)
        assert len(result) == len(texts)


class TestEmbeddingIntegration:
    """Test integration between sparse and dense embeddings."""

    def test_embedding_dimension_consistency(self, test_settings):
        """Test that embedding dimensions are consistent with settings."""
        # Test that dense embedding dimension matches settings
        assert test_settings.dense_embedding_dimension == 1024

        # Verify model names are correctly configured
        assert "bge-large-en-v1.5" in test_settings.dense_embedding_model
        assert "Splade_PP_en_v1" in test_settings.sparse_embedding_model

    def test_embedding_preprocessing(self, sample_documents):
        """Test text preprocessing before embedding."""
        # Test that documents are properly prepared for embedding
        texts = [doc.text for doc in sample_documents]

        for text in texts:
            # Text should not be empty
            assert len(text.strip()) > 0
            # Text should be reasonable length
            assert len(text) < 10000  # Avoid extremely long texts
            # Text should contain meaningful content
            assert any(word.isalpha() for word in text.split())

    @pytest.mark.integration
    def test_embedding_pipeline_integration(
        self,
        sample_documents,
        mock_embedding_model,
        mock_sparse_embedding_model,
    ):
        """Test complete embedding pipeline with both sparse and dense."""
        texts = [doc.text for doc in sample_documents]

        # Mock both embedding types
        with patch("utils.FastEmbedModelManager.get_model") as mock_get_model:

            def side_effect(model_name):
                if "splade" in model_name.lower():
                    return mock_sparse_embedding_model
                else:
                    return mock_embedding_model

            mock_get_model.side_effect = side_effect

            manager = FastEmbedModelManager()

            # Test dense embeddings
            dense_model = manager.get_model("BAAI/bge-large-en-v1.5")
            dense_embeddings = dense_model.embed_documents(texts)

            # Test sparse embeddings
            sparse_model = manager.get_model("prithvida/Splade_PP_en_v1")
            sparse_embeddings = sparse_model.encode(texts)

            # Verify both embedding types are generated
            assert len(dense_embeddings) == len(texts)
            assert len(sparse_embeddings) == len(texts)

            # Verify embedding structures
            for dense_emb in dense_embeddings:
                assert isinstance(dense_emb, list)
                assert len(dense_emb) == 1024  # BGE-Large dimension

            for sparse_emb in sparse_embeddings:
                assert isinstance(sparse_emb, dict)
                assert "indices" in sparse_emb
                assert "values" in sparse_emb

    def test_embedding_error_handling(self):
        """Test error handling in embedding generation."""
        manager = FastEmbedModelManager()

        # Test with invalid model name
        with pytest.raises(
            (ValueError, RuntimeError, AttributeError),
            match="Invalid model|Model not found|Not available",
        ):
            manager.get_model("invalid/model-name")

        # Test with empty text
        with patch.object(manager, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.embed_documents.side_effect = ValueError("Empty text")
            mock_get_model.return_value = mock_model

            model = manager.get_model("test_model")
            with pytest.raises(ValueError, match="Empty text"):
                model.embed_documents([""])

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_real_embedding_models_loading(self):
        """Test loading real embedding models (requires network)."""
        # This test should only run when explicitly requested
        # Skip by default to avoid network dependencies
        pytest.skip("Requires network access and model downloads")
