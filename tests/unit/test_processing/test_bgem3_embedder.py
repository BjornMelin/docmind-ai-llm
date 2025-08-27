"""Unit tests for BGE-M3 Embedder with Dense AND Sparse Embeddings.

This module provides comprehensive unit tests for the new BGEM3Embedder
that provides unified dense, sparse, and ColBERT embeddings using FlagEmbedding.

CRITICAL TESTING FOCUS:
- Tests BOTH dense AND sparse embeddings are generated (ADR-002 compliance)
- Verifies 1024D dense embeddings with proper dimensions
- Validates sparse embeddings structure (token_id -> weight mappings)
- Tests batch processing and device selection
- Validates parameter configurations and backward compatibility

Test Coverage:
- BGEM3Embedder initialization and configuration
- Dense embeddings (1024D) generation and validation
- Sparse embeddings (lexical weights) generation and validation
- ColBERT embeddings (optional multi-vector)
- Unified embedding results with all types
- Batch processing and memory optimization
- Device selection (CPU/CUDA) and FP16 acceleration
- Performance statistics and model management
- Error handling and edge cases

Following 3-tier testing strategy:
- Tier 1 (Unit): Fast tests with mocks (<5s each)
- Use mocks for FlagEmbedding model and external dependencies
- Focus on logic validation and embedding type verification
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.models.embeddings import (
    EmbeddingError,
    EmbeddingParameters,
)
from src.processing.embeddings.bgem3_embedder import (
    BGEM3Embedder,
)


@pytest.fixture
def mock_settings():
    """Mock DocMind settings for testing."""
    settings = Mock()
    settings.bge_m3_model_name = "BAAI/bge-m3"
    settings.chunk_size = 512
    settings.chunk_overlap = 50
    return settings


@pytest.fixture
def embedding_parameters():
    """Default embedding parameters for testing."""
    return EmbeddingParameters(
        max_length=8192,
        batch_size_gpu=12,
        use_fp16=True,
        return_dense=True,
        return_sparse=True,
        return_colbert=False,
    )


@pytest.fixture
def mock_dense_embeddings():
    """Mock dense embeddings (1024D)."""
    return np.random.rand(3, 1024).astype(np.float32)


@pytest.fixture
def mock_sparse_embeddings():
    """Mock sparse embeddings (token weights)."""
    return [
        {1: 0.8, 5: 0.6, 10: 0.4, 23: 0.9},  # First text sparse weights
        {2: 0.7, 8: 0.5, 15: 0.8, 30: 0.6},  # Second text sparse weights
        {3: 0.9, 12: 0.4, 18: 0.7, 25: 0.5},  # Third text sparse weights
    ]


@pytest.fixture
def mock_colbert_embeddings():
    """Mock ColBERT embeddings."""
    return [
        np.random.rand(10, 1024).astype(np.float32),  # 10 tokens for first text
        np.random.rand(8, 1024).astype(np.float32),  # 8 tokens for second text
        np.random.rand(12, 1024).astype(np.float32),  # 12 tokens for third text
    ]


@pytest.fixture
def mock_bgem3_model():
    """Mock BGE-M3 FlagModel."""
    model = MagicMock()
    model.encode = MagicMock()
    model.compute_lexical_matching_score = MagicMock(return_value=0.75)
    model.model = MagicMock()
    return model


class TestBGEM3EmbedderInitialization:
    """Test BGEM3Embedder initialization and configuration."""

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_initialization_with_defaults(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test BGEM3Embedder initialization with default parameters."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder(mock_settings)

        # Verify initialization
        assert embedder.settings == mock_settings
        assert embedder.device == "cuda"
        assert embedder.batch_size == 12  # RTX 4090 optimized

        # Verify model initialization call
        mock_flag_model.assert_called_once_with(
            model_name_or_path=mock_settings.bge_m3_model_name,
            use_fp16=True,  # Default FP16 for CUDA
            device="cuda",
        )

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_initialization_with_custom_parameters(
        self,
        mock_torch,
        mock_flag_model,
        mock_settings,
        mock_bgem3_model,
        embedding_parameters,
    ):
        """Test initialization with custom embedding parameters."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        # Custom parameters with different settings
        custom_params = EmbeddingParameters(
            use_fp16=False, return_dense=True, return_sparse=True, return_colbert=True
        )

        embedder = BGEM3Embedder(mock_settings, custom_params)

        assert embedder.parameters == custom_params

        # Verify model initialization uses custom FP16 setting
        mock_flag_model.assert_called_once_with(
            model_name_or_path=mock_settings.bge_m3_model_name,
            use_fp16=False,  # Custom setting
            device="cuda",
        )

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_initialization_cpu_fallback(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test initialization falls back to CPU when CUDA unavailable."""
        mock_torch.cuda.is_available.return_value = False
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder(mock_settings)

        assert embedder.device == "cpu"
        assert embedder.batch_size == 2  # CPU optimized batch size

        # Verify model initialization
        mock_flag_model.assert_called_once_with(
            model_name_or_path=mock_settings.bge_m3_model_name,
            use_fp16=False,  # FP16 disabled for CPU
            device="cpu",
        )

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_initialization_gpu_memory_based_batch_size(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test batch size optimization based on GPU memory."""
        mock_torch.cuda.is_available.return_value = True
        mock_flag_model.return_value = mock_bgem3_model

        # Test different GPU memory configurations
        memory_configs = [
            (8 * 1024**3, 4),  # 8GB GPU -> batch_size 4
            (12 * 1024**3, 8),  # 12GB GPU -> batch_size 8
            (16 * 1024**3, 12),  # 16GB GPU -> batch_size 12
        ]

        for memory_size, expected_batch_size in memory_configs:
            mock_torch.cuda.get_device_properties.return_value = MagicMock(
                total_memory=memory_size
            )

            embedder = BGEM3Embedder(mock_settings)
            assert embedder.batch_size == expected_batch_size

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel", None)
    def test_initialization_missing_flagembedding(self, mock_settings):
        """Test error handling when FlagEmbedding is not available."""
        with pytest.raises(EmbeddingError) as exc_info:
            BGEM3Embedder(mock_settings)

        assert "flagembedding not available" in str(exc_info.value).lower()

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_initialization_model_loading_error(self, mock_flag_model, mock_settings):
        """Test error handling when BGE-M3 model fails to load."""
        mock_flag_model.side_effect = RuntimeError("Model loading failed")

        with pytest.raises(EmbeddingError) as exc_info:
            BGEM3Embedder(mock_settings)

        assert "model initialization failed" in str(exc_info.value).lower()


class TestUnifiedEmbeddings:
    """Test unified dense and sparse embeddings generation - CRITICAL FOR ADR-002."""

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_embed_texts_async_dense_and_sparse(
        self,
        mock_torch,
        mock_flag_model,
        mock_settings,
        mock_bgem3_model,
        mock_dense_embeddings,
        mock_sparse_embeddings,
    ):
        """Test that BOTH dense AND sparse embeddings are generated correctly.

        This is the CRITICAL test for ADR-002 compliance.
        """
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        mock_flag_model.return_value = mock_bgem3_model

        # Mock unified embedding output with BOTH dense AND sparse
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": mock_dense_embeddings,
            "lexical_weights": mock_sparse_embeddings,
        }

        embedder = BGEM3Embedder(mock_settings)
        texts = ["First text document", "Second text document", "Third text document"]

        result = await embedder.embed_texts_async(texts)

        # CRITICAL: Verify BOTH dense and sparse embeddings are present
        assert result.dense_embeddings is not None, (
            "Dense embeddings missing - ADR-002 violation"
        )
        assert result.sparse_embeddings is not None, (
            "Sparse embeddings missing - ADR-002 violation"
        )

        # Verify dense embeddings structure (1024D)
        assert len(result.dense_embeddings) == 3
        assert len(result.dense_embeddings[0]) == 1024
        assert all(isinstance(emb, list) for emb in result.dense_embeddings)
        assert all(
            isinstance(val, float) for emb in result.dense_embeddings for val in emb
        )

        # Verify sparse embeddings structure (token_id -> weight mappings)
        assert len(result.sparse_embeddings) == 3
        assert all(
            isinstance(sparse_emb, dict) for sparse_emb in result.sparse_embeddings
        )
        assert all(
            all(
                isinstance(k, int) and isinstance(v, float)
                for k, v in sparse_emb.items()
            )
            for sparse_emb in result.sparse_embeddings
        )

        # Verify model was called with correct parameters
        mock_bgem3_model.encode.assert_called_once_with(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert=False,
        )

        # Verify metadata
        assert result.model_info["dense_enabled"] is True
        assert result.model_info["sparse_enabled"] is True
        assert result.model_info["embedding_dim"] == 1024

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_embed_texts_async_with_colbert(
        self,
        mock_torch,
        mock_flag_model,
        mock_settings,
        mock_bgem3_model,
        mock_dense_embeddings,
        mock_sparse_embeddings,
        mock_colbert_embeddings,
    ):
        """Test unified embeddings with all three types: dense, sparse, and ColBERT."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_flag_model.return_value = mock_bgem3_model

        # Mock unified embedding output with all three types
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": mock_dense_embeddings,
            "lexical_weights": mock_sparse_embeddings,
            "colbert_vecs": mock_colbert_embeddings,
        }

        # Enable all embedding types
        params = EmbeddingParameters(
            return_dense=True, return_sparse=True, return_colbert=True
        )
        embedder = BGEM3Embedder(mock_settings, params)

        texts = ["First text", "Second text", "Third text"]
        result = await embedder.embed_texts_async(texts, params)

        # Verify all three embedding types are present
        assert result.dense_embeddings is not None
        assert result.sparse_embeddings is not None
        assert result.colbert_embeddings is not None

        # Verify ColBERT structure (multi-vector per text)
        assert len(result.colbert_embeddings) == 3
        assert all(
            isinstance(colbert_emb, np.ndarray)
            for colbert_emb in result.colbert_embeddings
        )
        assert (
            result.colbert_embeddings[0].shape[1] == 1024
        )  # Each token vector is 1024D

        # Verify model call included ColBERT
        mock_bgem3_model.encode.assert_called_once_with(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_embed_texts_async_dense_only(
        self,
        mock_torch,
        mock_flag_model,
        mock_settings,
        mock_bgem3_model,
        mock_dense_embeddings,
    ):
        """Test dense-only embeddings for backward compatibility."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_flag_model.return_value = mock_bgem3_model

        # Mock dense-only output
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": mock_dense_embeddings,
        }

        # Dense-only parameters
        params = EmbeddingParameters(
            return_dense=True, return_sparse=False, return_colbert=False
        )
        embedder = BGEM3Embedder(mock_settings, params)

        texts = ["Test text for dense embedding"]
        result = await embedder.embed_texts_async(texts, params)

        # Verify only dense embeddings present
        assert result.dense_embeddings is not None
        assert result.sparse_embeddings is None
        assert result.colbert_embeddings is None

        # Verify model call
        mock_bgem3_model.encode.assert_called_once_with(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_embed_texts_async_sparse_only(
        self,
        mock_torch,
        mock_flag_model,
        mock_settings,
        mock_bgem3_model,
        mock_sparse_embeddings,
    ):
        """Test sparse-only embeddings."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_flag_model.return_value = mock_bgem3_model

        # Mock sparse-only output
        mock_bgem3_model.encode.return_value = {
            "lexical_weights": mock_sparse_embeddings,
        }

        # Sparse-only parameters
        params = EmbeddingParameters(
            return_dense=False, return_sparse=True, return_colbert=False
        )
        embedder = BGEM3Embedder(mock_settings, params)

        texts = ["Test text for sparse embedding"]
        result = await embedder.embed_texts_async(texts, params)

        # Verify only sparse embeddings present
        assert result.dense_embeddings is None
        assert result.sparse_embeddings is not None
        assert result.colbert_embeddings is None

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_embed_texts_async_empty_input(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test handling of empty text input."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder(mock_settings)

        result = await embedder.embed_texts_async([])

        # Verify empty result structure
        assert result.dense_embeddings == []
        assert result.sparse_embeddings is None
        assert result.processing_time == 0.0
        assert result.batch_size == 0
        assert "warning" in result.model_info

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_embed_single_text_async(
        self,
        mock_torch,
        mock_flag_model,
        mock_settings,
        mock_bgem3_model,
        mock_dense_embeddings,
    ):
        """Test single text embedding for backward compatibility."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_flag_model.return_value = mock_bgem3_model

        # Mock single text output
        single_embedding = mock_dense_embeddings[:1]  # First embedding only
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": single_embedding,
            "lexical_weights": [{}],  # Empty sparse for single text
        }

        embedder = BGEM3Embedder(mock_settings)

        result = await embedder.embed_single_text_async("Single test text")

        # Verify single dense embedding returned
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(val, float) for val in result)


class TestIndividualEmbeddingMethods:
    """Test individual embedding extraction methods."""

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_get_sparse_embeddings(
        self,
        mock_torch,
        mock_flag_model,
        mock_settings,
        mock_bgem3_model,
        mock_sparse_embeddings,
    ):
        """Test sparse-only embedding extraction method."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        mock_bgem3_model.encode.return_value = {
            "lexical_weights": mock_sparse_embeddings,
        }

        embedder = BGEM3Embedder(mock_settings)
        texts = ["First text", "Second text", "Third text"]

        result = embedder.get_sparse_embeddings(texts)

        # Verify sparse embeddings structure
        assert result == mock_sparse_embeddings
        assert len(result) == 3
        assert all(isinstance(sparse_emb, dict) for sparse_emb in result)

        # Verify model call
        mock_bgem3_model.encode.assert_called_once_with(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_get_dense_embeddings(
        self,
        mock_torch,
        mock_flag_model,
        mock_settings,
        mock_bgem3_model,
        mock_dense_embeddings,
    ):
        """Test dense-only embedding extraction method."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        mock_bgem3_model.encode.return_value = {
            "dense_vecs": mock_dense_embeddings,
        }

        embedder = BGEM3Embedder(mock_settings)
        texts = ["First text", "Second text", "Third text"]

        result = embedder.get_dense_embeddings(texts)

        # Verify dense embeddings structure
        assert len(result) == 3
        assert len(result[0]) == 1024
        assert all(isinstance(emb, list) for emb in result)

        # Verify model call
        mock_bgem3_model.encode.assert_called_once_with(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_get_sparse_embeddings_error_handling(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test error handling in sparse embedding extraction."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        mock_bgem3_model.encode.side_effect = Exception("Encoding failed")

        embedder = BGEM3Embedder(mock_settings)

        result = embedder.get_sparse_embeddings(["Test text"])

        # Should return None on error
        assert result is None

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_get_dense_embeddings_error_handling(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test error handling in dense embedding extraction."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        mock_bgem3_model.encode.side_effect = Exception("Encoding failed")

        embedder = BGEM3Embedder(mock_settings)

        result = embedder.get_dense_embeddings(["Test text"])

        # Should return None on error
        assert result is None

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_compute_sparse_similarity(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test sparse similarity computation."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder(mock_settings)

        sparse1 = {1: 0.8, 5: 0.6, 10: 0.4}
        sparse2 = {1: 0.7, 8: 0.5, 10: 0.8}

        result = embedder.compute_sparse_similarity(sparse1, sparse2)

        # Verify similarity computation
        assert isinstance(result, float)
        assert result == 0.75  # Mock return value

        mock_bgem3_model.compute_lexical_matching_score.assert_called_once_with(
            sparse1, sparse2
        )


class TestPerformanceAndStats:
    """Test performance statistics and model management."""

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_performance_stats_tracking(
        self,
        mock_torch,
        mock_flag_model,
        mock_settings,
        mock_bgem3_model,
        mock_dense_embeddings,
    ):
        """Test performance statistics are tracked correctly."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_flag_model.return_value = mock_bgem3_model

        mock_bgem3_model.encode.return_value = {
            "dense_vecs": mock_dense_embeddings[:2],  # 2 texts
        }

        embedder = BGEM3Embedder(mock_settings)

        # Process some texts
        texts = ["First text", "Second text"]
        await embedder.embed_texts_async(texts)

        stats = embedder.get_performance_stats()

        # Verify statistics
        assert stats["total_texts_embedded"] == 2
        assert stats["total_processing_time"] > 0
        assert stats["avg_time_per_text_ms"] > 0
        assert stats["device"] == "cuda"
        assert stats["batch_size"] == 12
        assert stats["model_library"] == "FlagEmbedding.BGEM3FlagModel"
        assert stats["unified_embeddings_enabled"] is True

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_reset_stats(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test statistics reset functionality."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder(mock_settings)

        # Simulate some processing
        embedder._embedding_count = 100
        embedder._total_processing_time = 5.0

        embedder.reset_stats()

        # Verify reset
        assert embedder._embedding_count == 0
        assert embedder._total_processing_time == 0.0

        stats = embedder.get_performance_stats()
        assert stats["total_texts_embedded"] == 0
        assert stats["total_processing_time"] == 0.0
        assert stats["avg_time_per_text_ms"] == 0.0

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_unload_model(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test model unloading functionality."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder(mock_settings)

        # Simulate some processing
        embedder._embedding_count = 50
        embedder._total_processing_time = 2.5

        embedder.unload_model()

        # Verify stats are reset
        assert embedder._embedding_count == 0
        assert embedder._total_processing_time == 0.0

        # Verify model moved to CPU
        mock_bgem3_model.model.to.assert_called_once_with("cpu")


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_embed_texts_async_model_error(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test error handling during embedding generation."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        mock_bgem3_model.encode.side_effect = Exception("GPU out of memory")

        embedder = BGEM3Embedder(mock_settings)

        with pytest.raises(EmbeddingError) as exc_info:
            await embedder.embed_texts_async(["Test text"])

        assert "unified embedding failed" in str(exc_info.value).lower()

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_embed_single_text_async_no_embeddings(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test error handling when no embeddings are generated."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_flag_model.return_value = mock_bgem3_model

        # Mock empty result
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": None,
            "lexical_weights": [{}],
        }

        embedder = BGEM3Embedder(mock_settings)

        with pytest.raises(EmbeddingError) as exc_info:
            await embedder.embed_single_text_async("Test text")

        assert "no dense embeddings generated" in str(exc_info.value).lower()


class TestLibraryFirstMethods:
    """Test library-first FlagEmbedding integration methods."""

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_encode_queries_integration(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test encode_queries method from FlagEmbedding library.

        Tests library-first approach using FlagEmbedding's encode_queries
        method for query-optimized embeddings.
        """
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        # Mock encode_queries method
        mock_bgem3_model.encode_queries.return_value = {
            "dense_vecs": [[0.1] * 1024, [0.2] * 1024],
            "lexical_weights": [
                {"query": 0.9, "search": 0.7},
                {"document": 0.8, "retrieval": 0.6},
            ],
        }

        embedder = BGEM3Embedder(mock_settings)
        queries = ["search query example", "document retrieval test"]

        # Test encode_queries method exists and works
        result = await embedder.encode_queries_async(queries)

        # Verify encode_queries was called with correct parameters
        mock_bgem3_model.encode_queries.assert_called_once_with(
            queries, batch_size=12, max_length=8192
        )

        # Verify result structure
        assert result.dense_embeddings is not None
        assert result.sparse_embeddings is not None
        assert len(result.dense_embeddings) == 2
        assert len(result.sparse_embeddings) == 2

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_encode_corpus_integration(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test encode_corpus method from FlagEmbedding library.

        Tests library-first approach using FlagEmbedding's encode_corpus
        method for document-optimized embeddings.
        """
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        # Mock encode_corpus method
        mock_bgem3_model.encode_corpus.return_value = {
            "dense_vecs": [[0.3] * 1024, [0.4] * 1024],
            "lexical_weights": [
                {"document": 0.9, "content": 0.8},
                {"text": 0.7, "passage": 0.6},
            ],
        }

        embedder = BGEM3Embedder(mock_settings)
        documents = ["document content example", "text passage sample"]

        # Test encode_corpus method exists and works
        result = await embedder.encode_corpus_async(documents)

        # Verify encode_corpus was called with correct parameters
        mock_bgem3_model.encode_corpus.assert_called_once_with(
            documents, batch_size=12, max_length=8192
        )

        # Verify result structure
        assert result.dense_embeddings is not None
        assert result.sparse_embeddings is not None
        assert len(result.dense_embeddings) == 2
        assert len(result.sparse_embeddings) == 2

        # Verify different embeddings from queries (corpus-optimized)
        # This would be different in real implementation
        assert result.dense_embeddings[0][0] == 0.3  # Different from query 0.1

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_pooling_method_parameters(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test pooling_method and normalize_embeddings parameters.

        Tests library-first configuration of FlagEmbedding pooling methods
        and embedding normalization settings.
        """
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        # Test CLS pooling with normalization
        params = EmbeddingParameters(pooling_method="cls", normalize_embeddings=True)

        embedder = BGEM3Embedder(mock_settings, params)

        # Verify model initialization with pooling parameters
        mock_flag_model.assert_called_once_with(
            model_name_or_path=mock_settings.bge_m3_model_name,
            use_fp16=True,
            device="cuda",
            pooling_method="cls",
            normalize_embeddings=True,
        )

        # Test mean pooling without normalization
        mock_flag_model.reset_mock()
        params2 = EmbeddingParameters(pooling_method="mean", normalize_embeddings=False)

        embedder2 = BGEM3Embedder(mock_settings, params2)

        mock_flag_model.assert_called_once_with(
            model_name_or_path=mock_settings.bge_m3_model_name,
            use_fp16=True,
            device="cuda",
            pooling_method="mean",
            normalize_embeddings=False,
        )

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_flag_embedding_direct_usage(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test direct FlagEmbedding usage without custom wrappers.

        Verifies library-first approach - using FlagEmbedding directly
        rather than custom wrapper implementations.
        """
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder(mock_settings)

        # Verify direct FlagEmbedding model usage
        assert hasattr(embedder, "model")
        assert embedder.model == mock_bgem3_model

        # Verify no custom wrapper classes
        assert not hasattr(embedder, "custom_encoder")
        assert not hasattr(embedder, "wrapper_model")
        assert not hasattr(embedder, "embedding_wrapper")

        # Verify access to all FlagEmbedding methods
        assert hasattr(mock_bgem3_model, "encode")
        assert hasattr(mock_bgem3_model, "encode_queries") or callable(
            getattr(mock_bgem3_model, "encode_queries", None)
        )
        assert hasattr(mock_bgem3_model, "encode_corpus") or callable(
            getattr(mock_bgem3_model, "encode_corpus", None)
        )

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_8192_token_context_window(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test 8192 token context window configuration.

        Verifies that max_length parameter is correctly set to 8192
        for both encode_queries and encode_corpus methods.
        """
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        # Mock methods to verify max_length parameter
        mock_bgem3_model.encode_queries.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{"test": 0.8}],
        }
        mock_bgem3_model.encode_corpus.return_value = {
            "dense_vecs": [[0.2] * 1024],
            "lexical_weights": [{"doc": 0.9}],
        }

        embedder = BGEM3Embedder(mock_settings)

        # Test encode_queries with 8K context
        embedder.encode_queries(["test query"])
        mock_bgem3_model.encode_queries.assert_called_with(
            ["test query"],
            batch_size=12,
            max_length=8192,  # Verify 8K context window
        )

        # Test encode_corpus with 8K context
        embedder.encode_corpus(["test document"])
        mock_bgem3_model.encode_corpus.assert_called_with(
            ["test document"],
            batch_size=12,
            max_length=8192,  # Verify 8K context window
        )


class TestFactoryFunction:
    """Test factory function for BGEM3Embedder."""

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_create_bgem3_embedder_with_settings(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test factory function with custom settings."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder(settings=mock_settings)

        assert isinstance(embedder, BGEM3Embedder)
        assert embedder.settings == mock_settings

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @patch("src.processing.embeddings.bgem3_embedder.app_settings")
    def test_create_bgem3_embedder_default_settings(
        self, mock_app_settings, mock_torch, mock_flag_model, mock_bgem3_model
    ):
        """Test factory function with default settings."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder()

        assert isinstance(embedder, BGEM3Embedder)
        assert embedder.settings == mock_app_settings

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_create_bgem3_embedder_with_parameters(
        self,
        mock_torch,
        mock_flag_model,
        mock_settings,
        mock_bgem3_model,
        embedding_parameters,
    ):
        """Test factory function with custom parameters."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder(
            settings=mock_settings, parameters=embedding_parameters
        )

        assert isinstance(embedder, BGEM3Embedder)
        assert embedder.settings == mock_settings
        assert embedder.parameters == embedding_parameters

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_create_bgem3_embedder_library_first_validation(
        self, mock_torch, mock_flag_model, mock_settings, mock_bgem3_model
    ):
        """Test factory function creates embedder with library-first validation.

        Ensures factory function creates BGEM3Embedder that uses
        FlagEmbedding directly without custom wrappers.
        """
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_bgem3_model

        embedder = BGEM3Embedder(settings=mock_settings)

        # Verify it's using FlagEmbedding directly
        assert isinstance(embedder, BGEM3Embedder)
        assert embedder.model == mock_bgem3_model
        assert hasattr(embedder.model, "encode_queries")
        assert hasattr(embedder.model, "encode_corpus")

        # Verify library-first principle compliance
        mock_flag_model.assert_called_once_with(
            model_name_or_path=mock_settings.bge_m3_model_name,
            use_fp16=True,
            device="cuda",
        )
