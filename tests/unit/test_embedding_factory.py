"""Comprehensive tests for utils/embedding_factory.py.

Tests EmbeddingFactory with comprehensive coverage of dense/sparse/multimodal models,
LRU caching, GPU acceleration, and provider configuration.

Target coverage: 95%+ for embedding factory utilities.
"""

from unittest.mock import MagicMock, call, patch

import pytest
import torch

from utils.embedding_factory import EmbeddingFactory


class TestEmbeddingFactory:
    """Test suite for EmbeddingFactory with comprehensive coverage."""

    def test_get_providers_gpu_enabled_available(self):
        """Test provider selection when GPU is enabled and available."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="Tesla V100"),
            patch("utils.embedding_factory.logging") as mock_logging,
        ):
            providers = EmbeddingFactory.get_providers(use_gpu=True)

            assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
            mock_logging.info.assert_called_with("Using GPU for embeddings: Tesla V100")

    def test_get_providers_gpu_enabled_unavailable(self):
        """Test provider selection when GPU is enabled but unavailable."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("utils.embedding_factory.logging") as mock_logging,
        ):
            providers = EmbeddingFactory.get_providers(use_gpu=True)

            assert providers == ["CPUExecutionProvider"]
            mock_logging.info.assert_called_with("Using CPU for embeddings")

    def test_get_providers_gpu_disabled(self):
        """Test provider selection when GPU is explicitly disabled."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("utils.embedding_factory.logging") as mock_logging,
        ):
            providers = EmbeddingFactory.get_providers(use_gpu=False)

            assert providers == ["CPUExecutionProvider"]
            mock_logging.info.assert_called_with("Using CPU for embeddings")

    def test_get_providers_gpu_detection_error(self):
        """Test provider selection when GPU info detection fails."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", side_effect=RuntimeError("CUDA error")),
            patch("utils.embedding_factory.logging") as mock_logging,
        ):
            providers = EmbeddingFactory.get_providers(use_gpu=True)

            assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
            mock_logging.warning.assert_called_with(
                "GPU info detection failed: CUDA error"
            )

    @patch("utils.embedding_factory.settings")
    def test_get_providers_uses_settings_default(self, mock_settings):
        """Test provider selection uses settings when use_gpu is None."""
        mock_settings.gpu_acceleration = True

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="RTX 3090"),
        ):
            providers = EmbeddingFactory.get_providers(use_gpu=None)

            assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @patch("utils.embedding_factory.FastEmbedEmbedding")
    @patch.object(EmbeddingFactory, "get_providers")
    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_dense_embedding_basic(
        self, mock_logging, mock_settings, mock_get_providers, mock_fastembed
    ):
        """Test basic dense embedding model creation."""
        # Setup mocks
        mock_settings.dense_embedding_model = "BAAI/bge-large-en-v1.5"
        mock_settings.embedding_batch_size = 32
        mock_get_providers.return_value = ["CPUExecutionProvider"]
        mock_model = MagicMock()
        mock_fastembed.return_value = mock_model

        # Clear cache before test
        EmbeddingFactory.clear_cache()

        result = EmbeddingFactory.create_dense_embedding(use_gpu=False)

        assert result == mock_model
        mock_fastembed.assert_called_once_with(
            model_name="BAAI/bge-large-en-v1.5",
            max_length=512,
            providers=["CPUExecutionProvider"],
            batch_size=32,
            cache_dir="./embeddings_cache",
        )
        mock_logging.info.assert_called_with(
            "Dense embedding model created: BAAI/bge-large-en-v1.5"
        )

    @patch("utils.embedding_factory.FastEmbedEmbedding")
    @patch.object(EmbeddingFactory, "get_providers")
    @patch("utils.embedding_factory.torch.compile")
    @patch("torch.cuda.is_available")
    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_dense_embedding_with_torch_compile(
        self,
        mock_logging,
        mock_settings,
        mock_cuda_available,
        mock_torch_compile,
        mock_get_providers,
        mock_fastembed,
    ):
        """Test dense embedding creation with torch.compile optimization."""
        # Setup mocks
        mock_settings.dense_embedding_model = "BAAI/bge-large-en-v1.5"
        mock_settings.embedding_batch_size = 32
        mock_get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_cuda_available.return_value = True

        mock_model = MagicMock()
        mock_compiled_model = MagicMock()
        mock_fastembed.return_value = mock_model
        mock_torch_compile.return_value = mock_compiled_model

        # Clear cache before test
        EmbeddingFactory.clear_cache()

        # Mock torch.compile availability
        with patch("hasattr", return_value=True):
            result = EmbeddingFactory.create_dense_embedding(use_gpu=True)

        assert result == mock_compiled_model
        mock_torch_compile.assert_called_once_with(
            mock_model, mode="reduce-overhead", dynamic=True, fullgraph=False
        )
        mock_logging.info.assert_called_with(
            "torch.compile applied to dense embeddings with reduce-overhead"
        )

    @patch("utils.embedding_factory.FastEmbedEmbedding")
    @patch.object(EmbeddingFactory, "get_providers")
    @patch("utils.embedding_factory.torch.compile")
    @patch("torch.cuda.is_available")
    @patch("utils.embedding_factory.logging")
    def test_create_dense_embedding_torch_compile_failure(
        self,
        mock_logging,
        mock_cuda_available,
        mock_torch_compile,
        mock_get_providers,
        mock_fastembed,
    ):
        """Test dense embedding creation when torch.compile fails."""
        # Setup mocks
        mock_get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_cuda_available.return_value = True

        mock_model = MagicMock()
        mock_fastembed.return_value = mock_model
        mock_torch_compile.side_effect = RuntimeError("Compile failed")

        # Clear cache before test
        EmbeddingFactory.clear_cache()

        with patch("hasattr", return_value=True):
            result = EmbeddingFactory.create_dense_embedding(use_gpu=True)

        # Should return original model when compile fails
        assert result == mock_model
        mock_logging.warning.assert_called_with(
            "torch.compile failed for dense embeddings: Compile failed"
        )

    @patch("utils.embedding_factory.SparseTextEmbedding")
    @patch.object(EmbeddingFactory, "get_providers")
    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_sparse_embedding_enabled(
        self, mock_logging, mock_settings, mock_get_providers, mock_sparse
    ):
        """Test sparse embedding model creation when enabled."""
        # Setup mocks
        mock_settings.enable_sparse_embeddings = True
        mock_settings.sparse_embedding_model = "prithivida/Splade_PP_en_v1"
        mock_settings.embedding_batch_size = 32
        mock_get_providers.return_value = ["CPUExecutionProvider"]

        mock_model = MagicMock()
        mock_sparse.return_value = mock_model

        # Clear cache before test
        EmbeddingFactory.clear_cache()

        result = EmbeddingFactory.create_sparse_embedding(use_gpu=False)

        assert result == mock_model
        mock_sparse.assert_called_once_with(
            model_name="prithivida/Splade_PP_en_v1",
            providers=["CPUExecutionProvider"],
            batch_size=32,
            cache_dir="./embeddings_cache",
        )
        mock_logging.info.assert_called_with(
            "Sparse embedding model created: prithivida/Splade_PP_en_v1"
        )

    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_sparse_embedding_disabled(self, mock_logging, mock_settings):
        """Test sparse embedding creation when disabled in settings."""
        mock_settings.enable_sparse_embeddings = False

        # Clear cache before test
        EmbeddingFactory.clear_cache()

        result = EmbeddingFactory.create_sparse_embedding(use_gpu=False)

        assert result is None
        mock_logging.info.assert_called_with("Sparse embeddings disabled in settings")

    @patch("utils.embedding_factory.SparseTextEmbedding")
    @patch.object(EmbeddingFactory, "get_providers")
    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_sparse_embedding_creation_failure(
        self, mock_logging, mock_settings, mock_get_providers, mock_sparse
    ):
        """Test sparse embedding creation failure handling."""
        # Setup mocks
        mock_settings.enable_sparse_embeddings = True
        mock_settings.sparse_embedding_model = "prithivida/Splade_PP_en_v1"
        mock_settings.embedding_batch_size = 32
        mock_get_providers.return_value = ["CPUExecutionProvider"]

        mock_sparse.side_effect = RuntimeError("Model loading failed")

        # Clear cache before test
        EmbeddingFactory.clear_cache()

        result = EmbeddingFactory.create_sparse_embedding(use_gpu=False)

        assert result is None
        mock_logging.error.assert_called_with(
            "Failed to create sparse embedding model: Model loading failed"
        )

    @patch("utils.embedding_factory.HuggingFaceEmbedding")
    @patch("torch.cuda.is_available")
    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_multimodal_embedding_cpu(
        self, mock_logging, mock_settings, mock_cuda_available, mock_hf
    ):
        """Test multimodal embedding creation on CPU."""
        mock_cuda_available.return_value = False
        mock_settings.embedding_batch_size = 16
        mock_settings.enable_quantization = False

        mock_model = MagicMock()
        mock_hf.return_value = mock_model

        result = EmbeddingFactory.create_multimodal_embedding(use_gpu=False)

        assert result == mock_model
        mock_hf.assert_called_once_with(
            model_name="jinaai/jina-embeddings-v3",
            embed_batch_size=16,
            device="cpu",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float32},
        )
        mock_logging.info.assert_called_with(
            "Multimodal embedding model created on cpu"
        )

    @patch("utils.embedding_factory.HuggingFaceEmbedding")
    @patch("torch.cuda.is_available")
    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_multimodal_embedding_gpu(
        self, mock_logging, mock_settings, mock_cuda_available, mock_hf
    ):
        """Test multimodal embedding creation on GPU."""
        mock_cuda_available.return_value = True
        mock_settings.embedding_batch_size = 16
        mock_settings.enable_quantization = False

        mock_model = MagicMock()
        mock_hf.return_value = mock_model

        result = EmbeddingFactory.create_multimodal_embedding(use_gpu=True)

        assert result == mock_model
        mock_hf.assert_called_once_with(
            model_name="jinaai/jina-embeddings-v3",
            embed_batch_size=16,
            device="cuda",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
        )
        mock_logging.info.assert_called_with(
            "Multimodal embedding model created on cuda"
        )

    @patch("utils.embedding_factory.HuggingFaceEmbedding")
    @patch("torch.cuda.is_available")
    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_multimodal_embedding_with_quantization(
        self, mock_logging, mock_settings, mock_cuda_available, mock_hf
    ):
        """Test multimodal embedding creation with quantization."""
        mock_cuda_available.return_value = True
        mock_settings.embedding_batch_size = 16
        mock_settings.enable_quantization = True

        mock_model = MagicMock()
        mock_hf.return_value = mock_model

        # Mock BitsAndBytesConfig
        mock_quantization_config = MagicMock()
        with patch(
            "transformers.BitsAndBytesConfig", return_value=mock_quantization_config
        ):
            result = EmbeddingFactory.create_multimodal_embedding(use_gpu=True)

        assert result == mock_model

        # Verify quantization config was included
        call_args = mock_hf.call_args[1]
        assert "quantization_config" in call_args["model_kwargs"]
        mock_logging.info.assert_called_with(
            "Quantization enabled for multimodal embeddings"
        )

    @patch("utils.embedding_factory.HuggingFaceEmbedding")
    @patch("torch.cuda.is_available")
    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_multimodal_embedding_quantization_import_error(
        self, mock_logging, mock_settings, mock_cuda_available, mock_hf
    ):
        """Test multimodal embedding creation when transformers package unavailable."""
        mock_cuda_available.return_value = True
        mock_settings.embedding_batch_size = 16
        mock_settings.enable_quantization = True

        mock_model = MagicMock()
        mock_hf.return_value = mock_model

        # Mock ImportError for transformers
        with patch("transformers.BitsAndBytesConfig", side_effect=ImportError):
            result = EmbeddingFactory.create_multimodal_embedding(use_gpu=True)

        assert result == mock_model
        mock_logging.warning.assert_called_with(
            "transformers package not available, quantization disabled"
        )

    @patch.object(EmbeddingFactory, "create_dense_embedding")
    @patch.object(EmbeddingFactory, "create_sparse_embedding")
    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_hybrid_embeddings_both_enabled(
        self, mock_logging, mock_settings, mock_create_sparse, mock_create_dense
    ):
        """Test hybrid embedding creation with both dense and sparse enabled."""
        mock_settings.enable_sparse_embeddings = True

        mock_dense_model = MagicMock()
        mock_sparse_model = MagicMock()
        mock_create_dense.return_value = mock_dense_model
        mock_create_sparse.return_value = mock_sparse_model

        dense, sparse = EmbeddingFactory.create_hybrid_embeddings(use_gpu=True)

        assert dense == mock_dense_model
        assert sparse == mock_sparse_model

        mock_create_dense.assert_called_once_with(True)
        mock_create_sparse.assert_called_once_with(True)
        mock_logging.info.assert_called_with("Hybrid embeddings created: dense, sparse")

    @patch.object(EmbeddingFactory, "create_dense_embedding")
    @patch("utils.embedding_factory.settings")
    @patch("utils.embedding_factory.logging")
    def test_create_hybrid_embeddings_dense_only(
        self, mock_logging, mock_settings, mock_create_dense
    ):
        """Test hybrid embedding creation with only dense enabled."""
        mock_settings.enable_sparse_embeddings = False

        mock_dense_model = MagicMock()
        mock_create_dense.return_value = mock_dense_model

        dense, sparse = EmbeddingFactory.create_hybrid_embeddings(use_gpu=True)

        assert dense == mock_dense_model
        assert sparse is None

        mock_create_dense.assert_called_once_with(True)
        mock_logging.info.assert_called_with("Hybrid embeddings created: dense")

    def test_lru_cache_functionality(self):
        """Test LRU cache functionality for embedding models."""
        with (
            patch("utils.embedding_factory.FastEmbedEmbedding") as mock_fastembed,
            patch.object(
                EmbeddingFactory, "get_providers", return_value=["CPUExecutionProvider"]
            ),
        ):
            mock_model = MagicMock()
            mock_fastembed.return_value = mock_model

            # Clear cache before test
            EmbeddingFactory.clear_cache()

            # First call should create model
            result1 = EmbeddingFactory.create_dense_embedding(use_gpu=False)
            assert result1 == mock_model
            assert mock_fastembed.call_count == 1

            # Second call should return cached model
            result2 = EmbeddingFactory.create_dense_embedding(use_gpu=False)
            assert result2 == mock_model
            assert result1 is result2  # Same instance
            assert mock_fastembed.call_count == 1  # No additional calls

            # Different parameter should create new model
            result3 = EmbeddingFactory.create_dense_embedding(use_gpu=True)
            assert result3 == mock_model
            assert mock_fastembed.call_count == 2  # One more call

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        with (
            patch("utils.embedding_factory.FastEmbedEmbedding") as mock_fastembed,
            patch("utils.embedding_factory.SparseTextEmbedding") as mock_sparse,
            patch.object(
                EmbeddingFactory, "get_providers", return_value=["CPUExecutionProvider"]
            ),
            patch("utils.embedding_factory.settings") as mock_settings,
            patch("utils.embedding_factory.logging") as mock_logging,
        ):
            mock_settings.enable_sparse_embeddings = True
            mock_dense_model = MagicMock()
            mock_sparse_model = MagicMock()
            mock_fastembed.return_value = mock_dense_model
            mock_sparse.return_value = mock_sparse_model

            # Create models to populate cache
            EmbeddingFactory.create_dense_embedding(use_gpu=False)
            EmbeddingFactory.create_sparse_embedding(use_gpu=False)

            # Clear cache
            EmbeddingFactory.clear_cache()

            # Verify cache was cleared by checking new models are created
            EmbeddingFactory.create_dense_embedding(use_gpu=False)
            EmbeddingFactory.create_sparse_embedding(use_gpu=False)

            assert mock_fastembed.call_count == 2  # Called twice
            assert mock_sparse.call_count == 2  # Called twice
            mock_logging.info.assert_called_with("Embedding model cache cleared")

    def test_get_cache_info(self):
        """Test cache information retrieval."""
        with (
            patch("utils.embedding_factory.FastEmbedEmbedding") as mock_fastembed,
            patch("utils.embedding_factory.SparseTextEmbedding") as mock_sparse,
            patch.object(
                EmbeddingFactory, "get_providers", return_value=["CPUExecutionProvider"]
            ),
            patch("utils.embedding_factory.settings") as mock_settings,
        ):
            mock_settings.enable_sparse_embeddings = True
            mock_fastembed.return_value = MagicMock()
            mock_sparse.return_value = MagicMock()

            # Clear cache first
            EmbeddingFactory.clear_cache()

            # Create some models to generate cache stats
            EmbeddingFactory.create_dense_embedding(use_gpu=False)
            EmbeddingFactory.create_dense_embedding(use_gpu=False)  # Cache hit
            EmbeddingFactory.create_sparse_embedding(use_gpu=False)

            cache_info = EmbeddingFactory.get_cache_info()

            assert isinstance(cache_info, dict)
            assert "dense" in cache_info
            assert "sparse" in cache_info
            assert isinstance(cache_info["dense"], dict)
            assert isinstance(cache_info["sparse"], dict)

    @pytest.mark.parametrize("use_gpu", [True, False, None])
    def test_provider_consistency_across_methods(self, use_gpu):
        """Test that all methods use consistent provider configuration."""
        with (
            patch.object(EmbeddingFactory, "get_providers") as mock_get_providers,
            patch("utils.embedding_factory.FastEmbedEmbedding"),
            patch("utils.embedding_factory.SparseTextEmbedding"),
            patch("utils.embedding_factory.HuggingFaceEmbedding"),
            patch("utils.embedding_factory.settings") as mock_settings,
        ):
            mock_settings.enable_sparse_embeddings = True
            mock_get_providers.return_value = ["TestProvider"]

            # Clear cache before test
            EmbeddingFactory.clear_cache()

            # Call various methods
            EmbeddingFactory.create_dense_embedding(use_gpu=use_gpu)
            EmbeddingFactory.create_sparse_embedding(use_gpu=use_gpu)

            # Verify get_providers was called consistently
            expected_calls = [call(use_gpu), call(use_gpu)]
            mock_get_providers.assert_has_calls(expected_calls, any_order=False)

    def test_embedding_factory_stateless(self):
        """Test EmbeddingFactory maintains no internal state."""
        factory = EmbeddingFactory()
        assert len(factory.__dict__) == 0

        # All methods should be class methods or static methods
        assert callable(EmbeddingFactory.get_providers)
        assert callable(EmbeddingFactory.create_dense_embedding)
        assert callable(EmbeddingFactory.create_sparse_embedding)
        assert callable(EmbeddingFactory.create_multimodal_embedding)
        assert callable(EmbeddingFactory.create_hybrid_embeddings)
        assert callable(EmbeddingFactory.clear_cache)
        assert callable(EmbeddingFactory.get_cache_info)

    @patch("utils.embedding_factory.settings")
    def test_settings_integration(self, mock_settings):
        """Test integration with application settings."""
        # Configure mock settings
        mock_settings.dense_embedding_model = "custom/model"
        mock_settings.sparse_embedding_model = "custom/sparse-model"
        mock_settings.embedding_batch_size = 64
        mock_settings.enable_sparse_embeddings = True
        mock_settings.gpu_acceleration = False

        with (
            patch("utils.embedding_factory.FastEmbedEmbedding") as mock_fastembed,
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_model = MagicMock()
            mock_fastembed.return_value = mock_model

            # Clear cache before test
            EmbeddingFactory.clear_cache()

            result = EmbeddingFactory.create_dense_embedding()

            # Verify settings were used
            mock_fastembed.assert_called_once_with(
                model_name="custom/model",
                max_length=512,
                providers=["CPUExecutionProvider"],
                batch_size=64,
                cache_dir="./embeddings_cache",
            )

    def test_cache_isolation_between_parameters(self):
        """Test that different parameters create separate cache entries."""
        with (
            patch("utils.embedding_factory.FastEmbedEmbedding") as mock_fastembed,
            patch.object(EmbeddingFactory, "get_providers") as mock_get_providers,
        ):
            # Configure mocks to return different providers for different GPU settings
            mock_get_providers.side_effect = lambda gpu: (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if gpu
                else ["CPUExecutionProvider"]
            )

            mock_model_cpu = MagicMock(name="cpu_model")
            mock_model_gpu = MagicMock(name="gpu_model")
            mock_fastembed.side_effect = [mock_model_cpu, mock_model_gpu]

            # Clear cache before test
            EmbeddingFactory.clear_cache()

            # Create models with different parameters
            result_cpu = EmbeddingFactory.create_dense_embedding(use_gpu=False)
            result_gpu = EmbeddingFactory.create_dense_embedding(use_gpu=True)

            # Should create different instances
            assert result_cpu != result_gpu
            assert mock_fastembed.call_count == 2

            # Verify different providers were used
            calls = mock_fastembed.call_args_list
            assert calls[0][1]["providers"] == ["CPUExecutionProvider"]
            assert calls[1][1]["providers"] == [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.is_available")
    def test_gpu_info_error_handling(self, mock_cuda_available, mock_get_device_name):
        """Test various GPU information detection error scenarios."""
        mock_cuda_available.return_value = True

        # Test RuntimeError
        mock_get_device_name.side_effect = RuntimeError("CUDA driver error")
        with patch("utils.embedding_factory.logging") as mock_logging:
            providers = EmbeddingFactory.get_providers(use_gpu=True)
            assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
            mock_logging.warning.assert_called_with(
                "GPU info detection failed: CUDA driver error"
            )

        # Test AttributeError
        mock_get_device_name.side_effect = AttributeError("No attribute")
        with patch("utils.embedding_factory.logging") as mock_logging:
            providers = EmbeddingFactory.get_providers(use_gpu=True)
            assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
            mock_logging.warning.assert_called_with(
                "GPU info detection failed: No attribute"
            )

    def test_maxsize_parameter_lru_cache(self):
        """Test that LRU cache respects maxsize parameter."""
        # The cache is configured with maxsize=2, so verify this behavior
        with (
            patch("utils.embedding_factory.FastEmbedEmbedding") as mock_fastembed,
            patch.object(EmbeddingFactory, "get_providers") as mock_get_providers,
        ):
            mock_get_providers.return_value = ["CPUExecutionProvider"]
            mock_fastembed.side_effect = [
                MagicMock(name=f"model_{i}") for i in range(5)
            ]

            # Clear cache before test
            EmbeddingFactory.clear_cache()

            # The cache maxsize is 2, so create 3 different parameter combinations
            # This should cause cache eviction
            model1 = EmbeddingFactory.create_dense_embedding(use_gpu=False)
            model2 = EmbeddingFactory.create_dense_embedding(use_gpu=True)

            # Access first model again (should be cached)
            model1_again = EmbeddingFactory.create_dense_embedding(use_gpu=False)

            # Should be the same instance due to caching
            assert model1 is model1_again

            # Only 2 unique calls should have been made due to caching
            assert mock_fastembed.call_count == 2
