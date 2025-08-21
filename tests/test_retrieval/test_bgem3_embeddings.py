"""Unit tests for BGE-M3 unified embeddings (FEAT-002).

Tests the complete architectural replacement of BGE-large + SPLADE++
with BGE-M3 unified dense/sparse embeddings per ADR-002.

Test Coverage:
- BGEM3Embedding class initialization and configuration
- Unified dense/sparse/colbert embedding generation
- 8K context window support vs 512 in BGE-large
- FP16 acceleration for RTX 4090 optimization
- LlamaIndex BaseEmbedding integration
- Factory functions and Settings configuration
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval.embeddings.bge_m3_manager import (
    BGEM3Embedding,
    configure_bgem3_settings,
    create_bgem3_embedding,
)


class TestBGEM3Embedding:
    """Unit tests for BGEM3Embedding class."""

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_init_with_defaults(self, mock_flag_model, mock_bgem3_flag_model):
        """Test BGEM3Embedding initialization with default parameters."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()

        # Verify default parameters
        assert embedding.model_name == "BAAI/bge-m3"
        assert embedding.max_length == 8192  # 8K context vs 512 in BGE-large
        assert embedding.use_fp16 is True  # RTX 4090 optimization
        assert embedding.batch_size == 12  # RTX 4090 optimized
        assert embedding.device == "cuda"
        assert embedding.normalize_embeddings is True

        # Verify model initialization
        mock_flag_model.assert_called_once_with(
            "BAAI/bge-m3", use_fp16=True, device="cuda"
        )

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_init_with_custom_config(self, mock_flag_model, mock_bgem3_flag_model):
        """Test BGEM3Embedding initialization with custom configuration."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding(
            model_name="custom-bge-m3",
            max_length=4096,
            use_fp16=False,
            batch_size=8,
            device="cpu",
        )

        assert embedding.model_name == "custom-bge-m3"
        assert embedding.max_length == 4096
        assert embedding.use_fp16 is False
        assert embedding.batch_size == 8
        assert embedding.device == "cpu"

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel", None)
    def test_init_missing_flagembedding(self):
        """Test error handling when FlagEmbedding is not available."""
        with pytest.raises(ImportError, match="FlagEmbedding not available"):
            BGEM3Embedding()

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_init_model_loading_error(self, mock_flag_model):
        """Test error handling when BGE-M3 model fails to load."""
        mock_flag_model.side_effect = RuntimeError("Model loading failed")

        with pytest.raises(RuntimeError, match="Model loading failed"):
            BGEM3Embedding()

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_get_query_embedding(self, mock_flag_model, mock_bgem3_flag_model):
        """Test single query embedding generation."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        # Mock dense embedding output
        mock_bgem3_flag_model.encode.return_value = {
            "dense_vecs": np.random.rand(1, 1024).astype(np.float32)
        }

        embedding = BGEM3Embedding()
        query = "test query for embedding"

        result = embedding._get_query_embedding(query)

        # Verify embedding dimensions and type
        assert isinstance(result, list)
        assert len(result) == 1024  # BGE-M3 dense dimension
        assert all(isinstance(x, float) for x in result)

        # Verify model call
        mock_bgem3_flag_model.encode.assert_called_once_with(
            [query],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_get_text_embedding(self, mock_flag_model, mock_bgem3_flag_model):
        """Test single text embedding generation."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        mock_bgem3_flag_model.encode.return_value = {
            "dense_vecs": np.random.rand(1, 1024).astype(np.float32)
        }

        embedding = BGEM3Embedding()
        text = "test document text for embedding"

        result = embedding._get_text_embedding(text)

        assert isinstance(result, list)
        assert len(result) == 1024

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    async def test_async_query_embedding(self, mock_flag_model, mock_bgem3_flag_model):
        """Test async query embedding generation."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        mock_bgem3_flag_model.encode.return_value = {
            "dense_vecs": np.random.rand(1, 1024).astype(np.float32)
        }

        embedding = BGEM3Embedding()
        query = "async test query"

        result = await embedding._aget_query_embedding(query)

        assert isinstance(result, list)
        assert len(result) == 1024

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_get_unified_embeddings_dense_only(
        self, mock_flag_model, mock_bgem3_flag_model
    ):
        """Test unified embeddings with dense vectors only."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        mock_bgem3_flag_model.encode.return_value = {
            "dense_vecs": np.random.rand(2, 1024).astype(np.float32)
        }

        embedding = BGEM3Embedding()
        texts = ["text 1", "text 2"]

        result = embedding.get_unified_embeddings(
            texts, return_dense=True, return_sparse=False, return_colbert=False
        )

        # Verify output structure
        assert "dense" in result
        assert "sparse" not in result
        assert "colbert" not in result
        assert result["dense"].shape == (2, 1024)

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_get_unified_embeddings_full(
        self, mock_flag_model, sample_bgem3_embeddings
    ):
        """Test unified embeddings with all embedding types."""
        mock_flag_model.return_value = MagicMock()
        mock_flag_model.return_value.encode.return_value = {
            "dense_vecs": sample_bgem3_embeddings["dense"],
            "lexical_weights": sample_bgem3_embeddings["sparse"],
            "colbert_vecs": sample_bgem3_embeddings["colbert"],
        }

        embedding = BGEM3Embedding()
        texts = ["text 1", "text 2", "text 3"]

        result = embedding.get_unified_embeddings(
            texts, return_dense=True, return_sparse=True, return_colbert=True
        )

        # Verify all embedding types are present
        assert "dense" in result
        assert "sparse" in result
        assert "colbert" in result

        # Verify dense embeddings
        assert result["dense"].shape == (3, 1024)

        # Verify sparse embeddings (token weights)
        assert len(result["sparse"]) == 3
        assert isinstance(result["sparse"][0], dict)
        assert all(
            isinstance(k, int) and isinstance(v, float)
            for k, v in result["sparse"][0].items()
        )

        # Verify ColBERT embeddings
        assert len(result["colbert"]) == 3
        assert all(isinstance(cv, np.ndarray) for cv in result["colbert"])

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_get_unified_embeddings_error_handling(self, mock_flag_model):
        """Test error handling in unified embeddings."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("Encoding failed")
        mock_flag_model.return_value = mock_model

        embedding = BGEM3Embedding()

        with pytest.raises(RuntimeError, match="Encoding failed"):
            embedding.get_unified_embeddings(["test text"])

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_get_sparse_embedding(self, mock_flag_model, mock_bgem3_flag_model):
        """Test sparse embedding extraction for single text."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        # Mock sparse embedding output
        mock_sparse_weights = {1: 0.8, 5: 0.6, 10: 0.4, 23: 0.9}
        mock_bgem3_flag_model.encode.return_value = {
            "lexical_weights": [mock_sparse_weights]
        }

        embedding = BGEM3Embedding()
        text = "test text for sparse embedding"

        result = embedding.get_sparse_embedding(text)

        # Verify sparse embedding structure
        assert isinstance(result, dict)
        assert result == mock_sparse_weights
        assert all(
            isinstance(k, int) and isinstance(v, float) for k, v in result.items()
        )

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_get_sparse_embedding_empty(self, mock_flag_model, mock_bgem3_flag_model):
        """Test sparse embedding when no weights returned."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        # Mock empty sparse output
        mock_bgem3_flag_model.encode.return_value = {"lexical_weights": []}

        embedding = BGEM3Embedding()
        result = embedding.get_sparse_embedding("test text")

        assert result == {}

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_embed_dim_property(self, mock_flag_model, mock_bgem3_flag_model):
        """Test embed_dim property returns correct dimension."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()

        assert embedding.embed_dim == 1024  # BGE-M3 dimension

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_8k_context_window(self, mock_flag_model, mock_bgem3_flag_model):
        """Test 8K context window support vs 512 in BGE-large."""
        mock_flag_model.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode.return_value = {
            "dense_vecs": np.random.rand(1, 1024).astype(np.float32)
        }

        embedding = BGEM3Embedding(max_length=8192)

        # Generate long text (simulate large document chunk)
        long_text = "word " * 2000  # Approximately 4K tokens

        embedding._get_query_embedding(long_text)

        # Verify model called with 8K max_length
        mock_bgem3_flag_model.encode.assert_called_with(
            [long_text],
            batch_size=1,
            max_length=8192,  # 8K context vs 512 in BGE-large
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )


class TestBGEM3Factory:
    """Test factory functions and Settings integration."""

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_create_bgem3_embedding_defaults(
        self, mock_flag_model, mock_bgem3_flag_model
    ):
        """Test factory function with default parameters."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        embedding = create_bgem3_embedding()

        # Verify RTX 4090 optimized defaults
        assert embedding.model_name == "BAAI/bge-m3"
        assert embedding.use_fp16 is True
        assert embedding.device == "cuda"
        assert embedding.max_length == 8192
        assert embedding.batch_size == 12  # RTX 4090 optimized

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_create_bgem3_embedding_custom(
        self, mock_flag_model, mock_bgem3_flag_model
    ):
        """Test factory function with custom parameters."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        embedding = create_bgem3_embedding(
            model_name="custom-bge-m3", use_fp16=False, device="cpu", max_length=4096
        )

        assert embedding.model_name == "custom-bge-m3"
        assert embedding.use_fp16 is False
        assert embedding.device == "cpu"
        assert embedding.max_length == 4096
        assert embedding.batch_size == 4  # CPU optimized

    @patch("src.retrieval.embeddings.bge_m3_manager.create_bgem3_embedding")
    @patch("src.retrieval.embeddings.bge_m3_manager.Settings")
    def test_configure_bgem3_settings_success(self, mock_settings, mock_create):
        """Test successful Settings configuration."""
        mock_embedding = MagicMock()
        mock_create.return_value = mock_embedding

        configure_bgem3_settings()

        # Verify embedding created and assigned to Settings
        mock_create.assert_called_once()
        assert mock_settings.embed_model == mock_embedding

    @patch("src.retrieval.embeddings.bge_m3_manager.create_bgem3_embedding")
    def test_configure_bgem3_settings_error(self, mock_create):
        """Test Settings configuration error handling."""
        mock_create.side_effect = RuntimeError("Configuration failed")

        with pytest.raises(RuntimeError, match="Configuration failed"):
            configure_bgem3_settings()


class TestBGEM3Performance:
    """Performance and optimization tests."""

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_batch_processing_optimization(
        self, mock_flag_model, mock_bgem3_flag_model
    ):
        """Test batch processing for RTX 4090 optimization."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        # Mock batch embedding output
        mock_bgem3_flag_model.encode.return_value = {
            "dense_vecs": np.random.rand(12, 1024).astype(np.float32)
        }

        embedding = BGEM3Embedding(batch_size=12)
        texts = [f"text {i}" for i in range(12)]

        result = embedding.get_unified_embeddings(texts, return_dense=True)

        # Verify batch processing
        mock_bgem3_flag_model.encode.assert_called_once_with(
            texts,
            batch_size=12,  # RTX 4090 optimized batch size
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        assert result["dense"].shape == (12, 1024)

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_fp16_acceleration(self, mock_flag_model, mock_bgem3_flag_model):
        """Test FP16 acceleration for RTX 4090."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        _ = BGEM3Embedding(use_fp16=True, device="cuda")

        # Verify FP16 enabled in model initialization
        mock_flag_model.assert_called_once_with(
            "BAAI/bge-m3", use_fp16=True, device="cuda"
        )

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_memory_efficient_processing(self, mock_flag_model, mock_bgem3_flag_model):
        """Test memory-efficient processing for large documents."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        # Mock processing of large batch
        mock_bgem3_flag_model.encode.return_value = {
            "dense_vecs": np.random.rand(100, 1024).astype(np.float32)
        }

        embedding = BGEM3Embedding()

        # Process large document set
        large_texts = [f"document {i} with substantial content" for i in range(100)]

        result = embedding.get_unified_embeddings(large_texts)

        # Verify successful processing
        assert result["dense"].shape == (100, 1024)


@pytest.mark.integration
class TestBGEM3Integration:
    """Integration tests with LlamaIndex ecosystem."""

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_llamaindex_base_embedding_interface(
        self, mock_flag_model, mock_bgem3_flag_model
    ):
        """Test integration with LlamaIndex BaseEmbedding interface."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        mock_bgem3_flag_model.encode.return_value = {
            "dense_vecs": np.random.rand(1, 1024).astype(np.float32)
        }

        embedding = BGEM3Embedding()

        # Verify BaseEmbedding interface methods
        assert hasattr(embedding, "_get_query_embedding")
        assert hasattr(embedding, "_get_text_embedding")
        assert hasattr(embedding, "_aget_query_embedding")
        assert hasattr(embedding, "embed_dim")

        # Test interface methods
        query_result = embedding._get_query_embedding("test query")
        text_result = embedding._get_text_embedding("test text")

        assert len(query_result) == 1024
        assert len(text_result) == 1024

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_vector_store_integration(self, mock_flag_model, mock_bgem3_flag_model):
        """Test integration with vector stores (dimensions and types)."""
        mock_flag_model.return_value = mock_bgem3_flag_model

        mock_bgem3_flag_model.encode.return_value = {
            "dense_vecs": np.random.rand(3, 1024).astype(np.float32)
        }

        embedding = BGEM3Embedding()

        # Test embeddings suitable for vector store indexing
        texts = ["doc1", "doc2", "doc3"]
        result = embedding.get_unified_embeddings(texts)

        # Verify vector store compatibility
        assert result["dense"].dtype == np.float32  # Efficient storage
        assert result["dense"].shape[1] == 1024  # Consistent dimensions
        assert not np.any(np.isnan(result["dense"]))  # No NaN values
        assert np.all(np.isfinite(result["dense"]))  # Finite values
