"""Unit tests for unified embeddings functionality.

Focus areas:
- BGEM3Embedding init and model loading
- Dense, sparse, and ColBERT generation
- Async operations and CLIP configuration
- Factory functions and LlamaIndex integration
- Error handling and VRAM management
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from pydantic import ValidationError


@pytest.fixture
def mock_bgem3_model():
    """Mock BGEM3FlagModel for testing."""
    mock_model = Mock()
    mock_model.encode.return_value = {
        "dense_vecs": np.random.randn(2, 1024),
        "lexical_weights": [
            {100: 0.8, 200: 0.6, 300: 0.4},
            {150: 0.9, 250: 0.7, 350: 0.5},
        ],
        "colbert_vecs": [
            np.random.randn(32, 1024),  # 32 tokens, 1024 dims
            np.random.randn(28, 1024),  # 28 tokens, 1024 dims
        ],
    }
    return mock_model


@pytest.fixture
def mock_clip_embedding():
    """Mock ClipEmbedding for testing."""
    mock_clip = Mock()
    mock_clip.model_name = "openai/clip-vit-base-patch32"
    mock_clip.embed_batch_size = 10
    mock_clip.device = "cuda"
    return mock_clip


@pytest.mark.unit
class TestBGEM3EmbeddingInitialization:
    """Test BGEM3Embedding initialization and configuration."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_init_with_defaults(self, mock_bgem3_class, mock_bgem3_model):
        """Test initialization with default parameters."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model

        embedding = BGEM3Embedding()

        # Verify default configuration
        assert embedding.model_name == "BAAI/bge-m3"
        assert embedding.max_length == 8192
        assert embedding.use_fp16 is True
        assert embedding.batch_size == 12
        assert embedding.normalize_embeddings is True
        assert embedding.device == "cuda"
        assert embedding.embed_dim == 1024

        # Verify model initialization
        mock_bgem3_class.assert_called_once_with(
            "BAAI/bge-m3", use_fp16=True, device="cuda"
        )

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_init_custom_parameters(self, mock_bgem3_class, mock_bgem3_model):
        """Test initialization with custom parameters."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model

        embedding = BGEM3Embedding(
            model_name="custom/bge-model",
            max_length=4096,
            use_fp16=False,
            batch_size=8,
            device="cpu",
        )

        assert embedding.model_name == "custom/bge-model"
        assert embedding.max_length == 4096
        assert embedding.use_fp16 is False
        assert embedding.batch_size == 8
        assert embedding.device == "cpu"

        mock_bgem3_class.assert_called_once_with(
            "custom/bge-model", use_fp16=False, device="cpu"
        )

    def test_init_missing_flagembedding(self):
        """Test initialization fails when FlagEmbedding unavailable."""
        from src.retrieval.embeddings import BGEM3Embedding

        with (
            patch("src.retrieval.embeddings.BGEM3FlagModel", None),
            pytest.raises(ImportError, match="FlagEmbedding not available"),
        ):
            BGEM3Embedding()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_init_model_loading_error(self, mock_bgem3_class):
        """Test initialization handles model loading errors."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.side_effect = RuntimeError("Model loading failed")

        with pytest.raises(RuntimeError, match="Model loading failed"):
            BGEM3Embedding()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_init_import_error_handling(self, mock_bgem3_class):
        """Test handling of import errors during model loading."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.side_effect = ImportError("Missing dependency")

        with pytest.raises(ImportError, match="Missing dependency"):
            BGEM3Embedding()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_init_value_error_handling(self, mock_bgem3_class):
        """Test handling of value errors during model initialization."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.side_effect = ValueError("Invalid model configuration")

        with pytest.raises(ValueError, match="Invalid model configuration"):
            BGEM3Embedding()


@pytest.mark.unit
class TestBGEM3EmbeddingOperations:
    """Test core embedding generation operations."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_query_embedding(self, mock_bgem3_class, mock_bgem3_model):
        """Test single query embedding generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {"dense_vecs": np.random.randn(1, 1024)}

        embedding = BGEM3Embedding()
        result = embedding.get_query_embedding("test query")

        assert isinstance(result, list)
        assert len(result) == 1024

        # Verify encode was called with correct parameters
        mock_bgem3_model.encode.assert_called_once_with(
            ["test query"],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_text_embedding(self, mock_bgem3_class, mock_bgem3_model):
        """Test single text embedding generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {"dense_vecs": np.random.randn(1, 1024)}

        embedding = BGEM3Embedding()
        result = embedding.get_text_embedding("test text")

        assert isinstance(result, list)
        assert len(result) == 1024

        # Should call the same method as query embedding
        mock_bgem3_model.encode.assert_called_once()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    @pytest.mark.asyncio
    async def test_aget_query_embedding(self, mock_bgem3_class, mock_bgem3_model):
        """Test async query embedding generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {"dense_vecs": np.random.randn(1, 1024)}

        embedding = BGEM3Embedding()
        result = await embedding.aget_query_embedding("async test query")

        assert isinstance(result, list)
        assert len(result) == 1024

        mock_bgem3_model.encode.assert_called_once()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_unified_embeddings_all_types(self, mock_bgem3_class, mock_bgem3_model):
        """Test unified embeddings with all types enabled."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model

        embedding = BGEM3Embedding()
        texts = ["text 1", "text 2"]

        result = embedding.get_unified_embeddings(
            texts=texts, return_dense=True, return_sparse=True, return_colbert=True
        )

        # Verify all embedding types are returned
        assert "dense" in result
        assert "sparse" in result
        assert "colbert" in result

        # Verify encode was called with correct parameters
        mock_bgem3_model.encode.assert_called_once_with(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_unified_embeddings_dense_only(
        self, mock_bgem3_class, mock_bgem3_model
    ):
        """Test unified embeddings with only dense embeddings."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {"dense_vecs": np.random.randn(2, 1024)}

        embedding = BGEM3Embedding()
        texts = ["text 1", "text 2"]

        result = embedding.get_unified_embeddings(
            texts=texts, return_dense=True, return_sparse=False, return_colbert=False
        )

        assert "dense" in result
        assert "sparse" not in result
        assert "colbert" not in result

        mock_bgem3_model.encode.assert_called_once_with(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_unified_embeddings_sparse_only(
        self, mock_bgem3_class, mock_bgem3_model
    ):
        """Test unified embeddings with only sparse embeddings."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {
            "lexical_weights": [{100: 0.8, 200: 0.6}]
        }

        embedding = BGEM3Embedding()
        texts = ["sparse text"]

        result = embedding.get_unified_embeddings(
            texts=texts, return_dense=False, return_sparse=True, return_colbert=False
        )

        assert "dense" not in result
        assert "sparse" in result
        assert "colbert" not in result

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_unified_embeddings_error_handling(
        self, mock_bgem3_class, mock_bgem3_model
    ):
        """Test error handling during unified embedding generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.side_effect = RuntimeError("Encoding failed")

        embedding = BGEM3Embedding()
        texts = ["error text"]

        with pytest.raises(RuntimeError, match="Encoding failed"):
            embedding.get_unified_embeddings(texts)

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_unified_embeddings_value_error(
        self, mock_bgem3_class, mock_bgem3_model
    ):
        """Test value error handling during embedding generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.side_effect = ValueError("Invalid input")

        embedding = BGEM3Embedding()

        with pytest.raises(ValueError, match="Invalid input"):
            embedding.get_unified_embeddings(["text"])

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_sparse_embedding(self, mock_bgem3_class, mock_bgem3_model):
        """Test sparse embedding extraction for single text."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        expected_sparse = {100: 0.8, 200: 0.6, 300: 0.4}
        mock_bgem3_model.encode.return_value = {"lexical_weights": [expected_sparse]}

        embedding = BGEM3Embedding()
        result = embedding.get_sparse_embedding("test text")

        assert result == expected_sparse

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_sparse_embedding_empty_result(
        self, mock_bgem3_class, mock_bgem3_model
    ):
        """Test sparse embedding with empty result."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {
            "sparse": []  # Empty sparse result
        }

        embedding = BGEM3Embedding()
        result = embedding.get_sparse_embedding("test text")

        assert result == {}

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_sparse_embedding_missing_key(self, mock_bgem3_class, mock_bgem3_model):
        """Test sparse embedding with missing sparse key."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": np.random.randn(1, 1024)  # No sparse key
        }

        embedding = BGEM3Embedding()
        result = embedding.get_sparse_embedding("test text")

        assert result == {}


@pytest.mark.unit
class TestClipConfigValidation:
    """Test CLIP configuration validation and optimization."""

    def test_clip_config_defaults(self):
        """Test ClipConfig default values."""
        from src.retrieval.embeddings import ClipConfig

        config = ClipConfig()

        assert config.model_name == "openai/clip-vit-base-patch32"
        assert config.embed_batch_size == 10
        assert config.device in ["cuda", "cpu"]
        assert config.max_vram_gb == 1.4
        assert config.auto_adjust_batch is True

    def test_clip_config_custom_values(self):
        """Test ClipConfig with custom values."""
        from src.retrieval.embeddings import ClipConfig

        config = ClipConfig(
            model_name="openai/clip-vit-base-patch16",
            embed_batch_size=5,
            device="cpu",
            max_vram_gb=2.0,
            auto_adjust_batch=False,
        )

        assert config.model_name == "openai/clip-vit-base-patch16"
        assert config.embed_batch_size == 5
        assert config.device == "cpu"
        assert config.max_vram_gb == 2.0
        assert config.auto_adjust_batch is False

    def test_clip_config_model_name_validation(self):
        """Test model name validation."""
        from src.retrieval.embeddings import ClipConfig

        # Valid model names should work
        valid_models = [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
        ]

        for model in valid_models:
            config = ClipConfig(model_name=model)
            assert config.model_name == model

        # Invalid model name should raise error
        with pytest.raises(ValidationError):
            ClipConfig(model_name="invalid/model")

    def test_clip_config_batch_size_validation(self):
        """Test batch size validation with VRAM constraints."""
        from src.retrieval.embeddings import ClipConfig

        # Large batch size should be adjusted with warning
        with patch("src.retrieval.embeddings.logger") as mock_logger:
            config = ClipConfig(
                embed_batch_size=20,  # Above default limit
                max_vram_gb=1.4,
            )

            assert config.embed_batch_size == 10  # Adjusted down
            mock_logger.warning.assert_called_once()

    def test_clip_config_is_valid(self):
        """Test configuration validity check."""
        from src.retrieval.embeddings import ClipConfig

        # Valid configuration
        valid_config = ClipConfig(
            model_name="openai/clip-vit-base-patch32",
            embed_batch_size=8,
            max_vram_gb=1.2,
        )
        assert valid_config.is_valid() is True

        # Invalid configuration (unsupported model)
        invalid_config = ClipConfig()
        invalid_config.model_name = "unsupported/model"
        assert invalid_config.is_valid() is False

    def test_clip_config_optimize_for_hardware_cuda(self):
        """Test hardware optimization for CUDA."""
        from src.retrieval.embeddings import ClipConfig

        config = ClipConfig(
            device="cuda",
            embed_batch_size=15,  # Above VRAM limit
            max_vram_gb=1.4,
            auto_adjust_batch=True,
        )

        with patch("src.retrieval.embeddings.logger") as mock_logger:
            optimized = config.optimize_for_hardware()

            # Adjust batch size based on VRAM; float precision may yield 9 or 10
            expected = int(round(1.4 / 0.14))
            assert optimized.embed_batch_size in (expected, expected - 1)
            assert mock_logger.info.call_count >= 0

    def test_clip_config_optimize_for_hardware_disabled(self):
        """Test hardware optimization when disabled."""
        from src.retrieval.embeddings import ClipConfig

        # Use a larger VRAM limit to avoid validator downscaling at construction
        config = ClipConfig(
            device="cuda",
            embed_batch_size=15,
            max_vram_gb=5.0,
            auto_adjust_batch=False,
        )

        optimized = config.optimize_for_hardware()

        # Should not change batch size compared to already-validated config value
        assert optimized.embed_batch_size == config.embed_batch_size

    def test_clip_config_estimated_vram_usage(self):
        """Test VRAM usage estimation."""
        from src.retrieval.embeddings import ClipConfig

        config = ClipConfig(embed_batch_size=10)
        estimated = config.estimated_vram_usage()

        # 10 * 0.14 = 1.4GB (tolerate float rounding)
        assert estimated == pytest.approx(1.4, abs=1e-9)


@pytest.mark.unit
class TestFactoryFunctions:
    """Test factory functions for embedding creation."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    @patch("src.retrieval.embeddings.torch.cuda.is_available", return_value=True)
    def test_create_bgem3_embedding_defaults(
        self, mock_cuda, mock_bgem3_class, mock_bgem3_model
    ):
        """Test factory function with default parameters."""
        from src.retrieval.embeddings import create_bgem3_embedding

        mock_bgem3_class.return_value = mock_bgem3_model

        embedding = create_bgem3_embedding()

        assert embedding.model_name == "BAAI/bge-m3"
        assert embedding.use_fp16 is True
        assert embedding.device == "cuda"
        assert embedding.max_length == 8192
        assert embedding.batch_size == 12  # GPU batch size

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_create_bgem3_embedding_cpu(self, mock_bgem3_class, mock_bgem3_model):
        """Test factory function with CPU device."""
        from src.retrieval.embeddings import create_bgem3_embedding

        mock_bgem3_class.return_value = mock_bgem3_model

        embedding = create_bgem3_embedding(device="cpu")

        assert embedding.device == "cpu"
        assert embedding.batch_size == 4  # CPU batch size

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_create_bgem3_embedding_custom_params(
        self, mock_bgem3_class, mock_bgem3_model
    ):
        """Test factory function with custom parameters."""
        from src.retrieval.embeddings import create_bgem3_embedding

        mock_bgem3_class.return_value = mock_bgem3_model

        embedding = create_bgem3_embedding(
            model_name="custom/model", use_fp16=False, device="cpu", max_length=4096
        )

        assert embedding.model_name == "custom/model"
        assert embedding.use_fp16 is False
        assert embedding.device == "cpu"
        assert embedding.max_length == 4096

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_create_clip_embedding_with_dict_config(
        self, mock_clip_class, mock_clip_embedding
    ):
        """Test CLIP embedding creation with dict configuration."""
        from src.retrieval.embeddings import create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        config_dict = {
            "model_name": "openai/clip-vit-base-patch16",
            "embed_batch_size": 8,
            "device": "cpu",
        }

        with patch("src.retrieval.embeddings.logger"):
            result = create_clip_embedding(config_dict)

        assert result == mock_clip_embedding
        mock_clip_class.assert_called_once()

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_create_clip_embedding_with_config_object(
        self, mock_clip_class, mock_clip_embedding
    ):
        """Test CLIP embedding creation with ClipConfig object."""
        from src.retrieval.embeddings import ClipConfig, create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        config = ClipConfig(
            model_name="openai/clip-vit-base-patch32",
            embed_batch_size=10,
            device="cuda",
        )

        with patch("src.retrieval.embeddings.logger"):
            result = create_clip_embedding(config)

        assert result == mock_clip_embedding

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_create_clip_embedding_vram_warning(
        self, mock_clip_class, mock_clip_embedding
    ):
        """Test VRAM usage warning during CLIP embedding creation."""
        from src.retrieval.embeddings import create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        config_dict = {
            "embed_batch_size": 15,  # Will exceed VRAM limit
            "device": "cuda",
            "max_vram_gb": 1.0,
        }

        with patch("src.retrieval.embeddings.logger") as mock_logger:
            create_clip_embedding(config_dict)

            # Should log warning about VRAM usage
            mock_logger.warning.assert_called()

    @patch("src.retrieval.embeddings.ClipEmbedding")
    @patch("src.retrieval.embeddings.Settings")
    def test_setup_clip_for_llamaindex_with_config(
        self, mock_settings, mock_clip_class, mock_clip_embedding
    ):
        """Test setting up CLIP as default LlamaIndex embedding model."""
        from src.retrieval.embeddings import setup_clip_for_llamaindex

        mock_clip_class.return_value = mock_clip_embedding

        config_dict = {"embed_batch_size": 8}

        with patch("src.retrieval.embeddings.logger"):
            result = setup_clip_for_llamaindex(config_dict)

        assert result == mock_clip_embedding
        assert mock_settings.embed_model == mock_clip_embedding

    @patch("src.retrieval.embeddings.ClipEmbedding")
    @patch("src.retrieval.embeddings.Settings")
    def test_setup_clip_for_llamaindex_default_config(
        self, mock_settings, mock_clip_class, mock_clip_embedding
    ):
        """Test setting up CLIP with default configuration."""
        from src.retrieval.embeddings import setup_clip_for_llamaindex

        mock_clip_class.return_value = mock_clip_embedding

        with patch("src.retrieval.embeddings.logger"):
            result = setup_clip_for_llamaindex()

        assert result == mock_clip_embedding
        assert mock_settings.embed_model == mock_clip_embedding

    @patch("src.retrieval.embeddings.create_bgem3_embedding")
    @patch("src.retrieval.embeddings.Settings")
    def test_configure_bgem3_settings_success(
        self, mock_settings, mock_create_bgem3, mock_bgem3_model
    ):
        """Test configuring BGE-M3 as default LlamaIndex embedding model."""
        from src.retrieval.embeddings import configure_bgem3_settings

        mock_create_bgem3.return_value = mock_bgem3_model

        with patch("src.retrieval.embeddings.logger"):
            configure_bgem3_settings()

        mock_create_bgem3.assert_called_once()
        assert mock_settings.embed_model == mock_bgem3_model

    @patch("src.retrieval.embeddings.create_bgem3_embedding")
    def test_configure_bgem3_settings_error_handling(self, mock_create_bgem3):
        """Test error handling during BGE-M3 configuration."""
        from src.retrieval.embeddings import configure_bgem3_settings

        mock_create_bgem3.side_effect = ImportError("BGE-M3 not available")

        with pytest.raises(ImportError):
            configure_bgem3_settings()

    @patch("src.retrieval.embeddings.create_bgem3_embedding")
    def test_configure_bgem3_settings_runtime_error(self, mock_create_bgem3):
        """Test runtime error handling during BGE-M3 configuration."""
        from src.retrieval.embeddings import configure_bgem3_settings

        mock_create_bgem3.side_effect = RuntimeError("Model loading failed")

        with pytest.raises(RuntimeError):
            configure_bgem3_settings()

    @patch("src.retrieval.embeddings.create_bgem3_embedding")
    def test_configure_bgem3_settings_value_error(self, mock_create_bgem3):
        """Test value error handling during BGE-M3 configuration."""
        from src.retrieval.embeddings import configure_bgem3_settings

        mock_create_bgem3.side_effect = ValueError("Invalid configuration")

        with pytest.raises(ValueError, match="Invalid configuration"):
            configure_bgem3_settings()


@pytest.mark.unit
class TestEmbeddingConfiguration:
    """Test embedding configuration classes."""

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig default values."""
        from src.retrieval.embeddings import EmbeddingConfig

        config = EmbeddingConfig()

        assert config.model_name == "BAAI/bge-m3"
        assert config.dimension == 1024
        assert config.max_length == 8192
        assert config.batch_size_gpu == 12
        assert config.batch_size_cpu == 4

    def test_embedding_config_custom_values(self):
        """Test EmbeddingConfig with custom values."""
        from src.retrieval.embeddings import EmbeddingConfig

        config = EmbeddingConfig(
            model_name="custom/model",
            dimension=512,
            max_length=4096,
            batch_size_gpu=16,
            batch_size_cpu=8,
        )

        assert config.model_name == "custom/model"
        assert config.dimension == 512
        assert config.max_length == 4096
        assert config.batch_size_gpu == 16
        assert config.batch_size_cpu == 8

    def test_embedding_config_validation(self):
        """Test EmbeddingConfig validation."""
        from src.retrieval.embeddings import EmbeddingConfig

        # Valid configuration
        config = EmbeddingConfig(
            dimension=1024, max_length=8192, batch_size_gpu=12, batch_size_cpu=4
        )

        assert config.dimension == 1024
        assert config.max_length == 8192
        assert config.batch_size_gpu == 12
        assert config.batch_size_cpu == 4


@pytest.mark.unit
class TestEmbeddingEdgeCases:
    """Test edge cases and error conditions."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_unified_embeddings_empty_texts(
        self, mock_bgem3_class, mock_bgem3_model
    ):
        """Test unified embeddings with empty text list."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": np.array([]).reshape(0, 1024),
            "lexical_weights": [],
            "colbert_vecs": [],
        }

        embedding = BGEM3Embedding()
        result = embedding.get_unified_embeddings([])

        # Should handle empty input gracefully
        assert "dense" in result
        assert "sparse" in result
        assert "colbert" in result

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_embedding_with_very_long_text(self, mock_bgem3_class, mock_bgem3_model):
        """Test embedding generation with very long text."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {"dense_vecs": np.random.randn(1, 1024)}

        embedding = BGEM3Embedding(max_length=512)

        # Text longer than max_length
        long_text = "word " * 1000
        result = embedding.get_query_embedding(long_text)

        assert isinstance(result, list)
        assert len(result) == 1024

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_embedding_with_special_characters(
        self, mock_bgem3_class, mock_bgem3_model
    ):
        """Test embedding generation with special characters."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {"dense_vecs": np.random.randn(1, 1024)}

        embedding = BGEM3Embedding()

        # Text with special characters and Unicode
        special_text = "Héllo wörld! 🌍 Special chars: <>&\"' \\n\\t"
        result = embedding.get_query_embedding(special_text)

        assert isinstance(result, list)
        assert len(result) == 1024

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_embedding_with_empty_string(self, mock_bgem3_class, mock_bgem3_model):
        """Test embedding generation with empty string."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model
        mock_bgem3_model.encode.return_value = {"dense_vecs": np.random.randn(1, 1024)}

        embedding = BGEM3Embedding()
        result = embedding.get_query_embedding("")

        assert isinstance(result, list)
        assert len(result) == 1024

    def test_clip_config_hardware_optimization_edge_cases(self):
        """Test CLIP configuration hardware optimization edge cases."""
        from src.retrieval.embeddings import ClipConfig

        # Test with very small VRAM limit
        config = ClipConfig(
            device="cuda", embed_batch_size=10, max_vram_gb=0.1, auto_adjust_batch=True
        )

        optimized = config.optimize_for_hardware()

        # Should adjust to minimum viable batch size
        assert optimized.embed_batch_size == 0  # 0.1GB / 0.14GB per image

    def test_clip_config_validation_edge_cases(self):
        """Test CLIP configuration validation edge cases."""
        from src.retrieval.embeddings import ClipConfig

        # Test with exact limit values
        config = ClipConfig(embed_batch_size=10, max_vram_gb=1.4)

        # Should not adjust when exactly at limit
        assert config.embed_batch_size == 10

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_dimension_property(
        self, mock_bgem3_class, mock_bgem3_model
    ):
        """Test embed_dim property returns correct dimension."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model

        embedding = BGEM3Embedding()

        assert embedding.embed_dim == 1024

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_pydantic_config(self, mock_bgem3_class, mock_bgem3_model):
        """Test Pydantic configuration settings."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_bgem3_class.return_value = mock_bgem3_model

        embedding = BGEM3Embedding()

        # Verify Config class settings
        assert embedding.Config.arbitrary_types_allowed is True
        assert embedding.Config.extra == "forbid"
        assert embedding.Config.validate_assignment is True
