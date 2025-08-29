"""Comprehensive unit tests for retrieval embeddings module.

Tests focus on covering the 138 uncovered statements in retrieval/embeddings.py
with emphasis on LlamaIndex integration, factory functions, configuration validation,
and CLIP multimodal embeddings with VRAM constraints.

Key areas:
- BGEM3Embedding LlamaIndex BaseEmbedding integration
- Factory functions and embedding creation utilities
- Configuration validation and hardware optimization
- CLIP embedding with VRAM constraint management
- Error handling and device management
- Performance optimization and batch processing
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from pydantic import ValidationError


@pytest.fixture
def mock_bgem3_flag_model():
    """Mock BGEM3FlagModel for retrieval embeddings testing."""
    model = Mock()

    # Mock encode methods for all variants
    def mock_encode(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)

        result = {}
        if kwargs.get("return_dense", True):
            result["dense_vecs"] = np.random.randn(batch_size, 1024).astype(np.float32)
        if kwargs.get("return_sparse", False):
            result["lexical_weights"] = [
                {i: float(np.random.random()) for i in range(5)}
                for _ in range(batch_size)
            ]
        if kwargs.get("return_colbert_vecs", False):
            result["colbert_vecs"] = [
                np.random.randn(np.random.randint(20, 100), 1024)
                for _ in range(batch_size)
            ]
        return result

    model.encode.side_effect = mock_encode
    return model


@pytest.fixture
def mock_clip_embedding():
    """Mock ClipEmbedding for testing."""
    clip = Mock(spec=ClipEmbedding)
    clip.embed_batch_size = 10
    clip.device = "cuda"
    clip.model_name = "openai/clip-vit-base-patch32"
    return clip


@pytest.mark.unit
class TestEmbeddingConfigDefaults:
    """Test EmbeddingConfig default values and validation."""

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
            batch_size_gpu=8,
            batch_size_cpu=2,
        )

        assert config.model_name == "custom/model"
        assert config.dimension == 512
        assert config.max_length == 4096
        assert config.batch_size_gpu == 8
        assert config.batch_size_cpu == 2

    def test_embedding_config_validation_limits(self):
        """Test EmbeddingConfig validation limits."""
        from src.retrieval.embeddings import EmbeddingConfig

        # Valid edge cases
        config = EmbeddingConfig(dimension=256, max_length=512, batch_size_gpu=1)
        assert config.dimension == 256
        assert config.max_length == 512
        assert config.batch_size_gpu == 1

        # Test boundaries
        with pytest.raises(ValidationError):
            EmbeddingConfig(dimension=200)  # Below minimum

        with pytest.raises(ValidationError):
            EmbeddingConfig(dimension=5000)  # Above maximum

        with pytest.raises(ValidationError):
            EmbeddingConfig(max_length=400)  # Below minimum

        with pytest.raises(ValidationError):
            EmbeddingConfig(max_length=20000)  # Above maximum


@pytest.mark.unit
class TestBGEM3EmbeddingInitialization:
    """Test BGEM3Embedding initialization and configuration."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_default_initialization(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGEM3Embedding initialization with defaults."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()

        # Verify default initialization
        assert embedding.model_name == "BAAI/bge-m3"
        assert embedding.max_length == 8192
        assert embedding.use_fp16 is True
        assert embedding.batch_size == 12  # GPU default
        assert embedding.normalize_embeddings is True
        assert embedding.device == "cuda"

        # Verify model was initialized
        mock_flag_model_class.assert_called_once_with(
            "BAAI/bge-m3", use_fp16=True, device="cuda"
        )

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_custom_initialization(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGEM3Embedding initialization with custom parameters."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding(
            model_name="custom/model",
            max_length=4096,
            use_fp16=False,
            batch_size=8,
            device="cpu",
        )

        # Verify custom initialization
        assert embedding.model_name == "custom/model"
        assert embedding.max_length == 4096
        assert embedding.use_fp16 is False
        assert embedding.batch_size == 8
        assert embedding.device == "cpu"

        # Verify model was initialized with custom params
        mock_flag_model_class.assert_called_once_with(
            "custom/model", use_fp16=False, device="cpu"
        )

    def test_bgem3_embedding_missing_flagembedding(self):
        """Test BGEM3Embedding handles missing FlagEmbedding."""
        from src.retrieval.embeddings import BGEM3Embedding

        with patch("src.retrieval.embeddings.BGEM3FlagModel", None):
            with pytest.raises(ImportError, match="FlagEmbedding not available"):
                BGEM3Embedding()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_model_loading_failure(self, mock_flag_model_class):
        """Test BGEM3Embedding handles model loading failure."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.side_effect = RuntimeError("Model loading failed")

        with pytest.raises(RuntimeError, match="Model loading failed"):
            BGEM3Embedding()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_pydantic_configuration(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGEM3Embedding Pydantic configuration."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()

        # Test Config class attributes
        config = embedding.Config
        assert config.arbitrary_types_allowed is True
        assert config.extra == "forbid"
        assert config.validate_assignment is True

        # Test embed_dim property
        assert embedding.embed_dim == 1024


@pytest.mark.unit
class TestBGEM3EmbeddingCoreOperations:
    """Test BGEM3Embedding core embedding operations."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_query_embedding(self, mock_flag_model_class, mock_bgem3_flag_model):
        """Test single query embedding generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()
        query = "What is machine learning?"

        result = embedding._get_query_embedding(query)

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)

        # Verify model was called correctly
        mock_bgem3_flag_model.encode.assert_called_once_with(
            [query],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_text_embedding(self, mock_flag_model_class, mock_bgem3_flag_model):
        """Test single text embedding generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()
        text = "Document content for embedding."

        result = embedding._get_text_embedding(text)

        # Should delegate to _get_query_embedding
        assert isinstance(result, list)
        assert len(result) == 1024

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    async def test_aget_query_embedding_async(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test async query embedding generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()
        query = "Async query test"

        result = await embedding._aget_query_embedding(query)

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_unified_embeddings_dense_only(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test unified embeddings with dense only."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()
        texts = ["Text 1", "Text 2", "Text 3"]

        result = embedding.get_unified_embeddings(
            texts, return_dense=True, return_sparse=False, return_colbert=False
        )

        # Verify result structure
        assert "dense" in result
        assert "sparse" not in result
        assert "colbert" not in result

        assert result["dense"].shape == (3, 1024)

        # Verify model call
        mock_bgem3_flag_model.encode.assert_called_once_with(
            texts,
            batch_size=12,  # Default GPU batch size
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_unified_embeddings_all_types(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test unified embeddings with all embedding types."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()
        texts = ["Text 1", "Text 2"]

        result = embedding.get_unified_embeddings(
            texts, return_dense=True, return_sparse=True, return_colbert=True
        )

        # Verify all embedding types are present
        assert "dense" in result
        assert "sparse" in result
        assert "colbert" in result

        # Verify dense embeddings
        assert result["dense"].shape[0] == 2
        assert result["dense"].shape[1] == 1024

        # Verify sparse embeddings
        assert len(result["sparse"]) == 2
        assert all(isinstance(sparse, dict) for sparse in result["sparse"])

        # Verify ColBERT embeddings
        assert len(result["colbert"]) == 2

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_unified_embeddings_error_handling(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test unified embeddings error handling."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode.side_effect = RuntimeError("Model error")

        embedding = BGEM3Embedding()
        texts = ["Test text"]

        with pytest.raises(RuntimeError, match="Model error"):
            embedding.get_unified_embeddings(texts)

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_sparse_embedding_single(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test single text sparse embedding extraction."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()
        text = "Single text for sparse embedding"

        result = embedding.get_sparse_embedding(text)

        assert isinstance(result, dict)
        assert all(
            isinstance(k, int) and isinstance(v, float) for k, v in result.items()
        )

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_sparse_embedding_empty_result(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test sparse embedding when no sparse results returned."""
        from src.retrieval.embeddings import BGEM3Embedding

        # Mock empty sparse result
        def mock_encode_empty_sparse(*args, **kwargs):
            return {"dense_vecs": np.random.randn(1, 1024)}  # No sparse data

        mock_bgem3_flag_model.encode.side_effect = mock_encode_empty_sparse
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()
        text = "Test text"

        result = embedding.get_sparse_embedding(text)

        assert result == {}  # Should return empty dict


@pytest.mark.unit
class TestBGEM3EmbeddingLlamaIndexIntegration:
    """Test BGEM3Embedding LlamaIndex BaseEmbedding compliance."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_llamaindex_base_embedding_interface(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGEM3Embedding implements BaseEmbedding interface."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()

        # Verify inheritance
        assert isinstance(embedding, BaseEmbedding)

        # Verify required methods exist
        assert hasattr(embedding, "_get_query_embedding")
        assert hasattr(embedding, "_get_text_embedding")
        assert hasattr(embedding, "_aget_query_embedding")
        assert hasattr(embedding, "embed_dim")

        # Verify properties
        assert embedding.embed_dim == 1024
        assert callable(embedding._get_query_embedding)
        assert callable(embedding._get_text_embedding)
        assert callable(embedding._aget_query_embedding)

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_llamaindex_embedding_dimensions(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test embedding dimension consistency with LlamaIndex."""
        from src.retrieval.embeddings import BGEM3Embedding, embedding_settings

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()

        # All dimension references should be consistent
        assert embedding.embed_dim == 1024
        assert embedding_settings.dimension == 1024

        # Test actual embedding dimensions
        result = embedding._get_query_embedding("test")
        assert len(result) == 1024


@pytest.mark.unit
class TestClipConfig:
    """Test ClipConfig validation and optimization."""

    @patch("src.retrieval.embeddings.torch")
    def test_clip_config_defaults(self, mock_torch):
        """Test ClipConfig default values."""
        from src.retrieval.embeddings import ClipConfig

        mock_torch.cuda.is_available.return_value = True

        config = ClipConfig()

        assert config.model_name == "openai/clip-vit-base-patch32"
        assert config.embed_batch_size == 10
        assert config.device == "cuda"  # Since CUDA available
        assert config.max_vram_gb == 1.4
        assert config.auto_adjust_batch is True

    @patch("src.retrieval.embeddings.torch")
    def test_clip_config_cpu_fallback(self, mock_torch):
        """Test ClipConfig CPU device detection."""
        from src.retrieval.embeddings import ClipConfig

        mock_torch.cuda.is_available.return_value = False

        config = ClipConfig()

        assert config.device == "cpu"

    def test_clip_config_model_name_validation(self):
        """Test ClipConfig model name validation."""
        from src.retrieval.embeddings import ClipConfig

        # Valid models should work
        valid_models = [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
        ]

        for model in valid_models:
            config = ClipConfig(model_name=model)
            assert config.model_name == model

        # Invalid model should raise error
        with pytest.raises(ValidationError, match="Unsupported model"):
            ClipConfig(model_name="invalid/model")

    def test_clip_config_batch_size_validation(self):
        """Test ClipConfig batch size adjustment."""
        from src.retrieval.embeddings import ClipConfig

        # Test batch size adjustment for VRAM constraint
        with patch("src.retrieval.embeddings.logger") as mock_logger:
            config = ClipConfig(embed_batch_size=20, max_vram_gb=1.4)

            # Should be adjusted down to default (10)
            assert config.embed_batch_size == 10

            # Should log warning
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Batch size 20 may exceed" in warning_msg

    def test_clip_config_is_valid(self):
        """Test ClipConfig validation method."""
        from src.retrieval.embeddings import ClipConfig

        # Valid configuration
        config = ClipConfig(
            model_name="openai/clip-vit-base-patch32",
            embed_batch_size=8,
            max_vram_gb=1.2,
        )
        assert config.is_valid() is True

        # Invalid configuration (unsupported model)
        config_invalid = ClipConfig(
            model_name="openai/clip-vit-large-patch14",  # Large model
            embed_batch_size=15,  # High batch size
            max_vram_gb=1.4,
        )
        assert config_invalid.is_valid() is False

    def test_clip_config_hardware_optimization(self):
        """Test ClipConfig hardware optimization."""
        from src.retrieval.embeddings import ClipConfig

        config = ClipConfig(
            embed_batch_size=20, max_vram_gb=2.0, auto_adjust_batch=True, device="cuda"
        )

        optimized = config.optimize_for_hardware()

        # Should optimize batch size for VRAM (2.0 / 0.14 ≈ 14)
        expected_max_batch = int(2.0 / 0.14)
        assert optimized.embed_batch_size <= expected_max_batch

    def test_clip_config_hardware_optimization_disabled(self):
        """Test ClipConfig with optimization disabled."""
        from src.retrieval.embeddings import ClipConfig

        original_batch_size = 20
        config = ClipConfig(
            embed_batch_size=original_batch_size,
            max_vram_gb=1.0,
            auto_adjust_batch=False,  # Disabled
        )

        optimized = config.optimize_for_hardware()

        # Should not change batch size when optimization disabled
        assert optimized.embed_batch_size == original_batch_size

    def test_clip_config_vram_estimation(self):
        """Test ClipConfig VRAM usage estimation."""
        from src.retrieval.embeddings import ClipConfig

        config = ClipConfig(embed_batch_size=10)
        estimated = config.estimated_vram_usage()

        # 10 images * 0.14GB per image = 1.4GB
        assert estimated == 1.4

        # Test different batch size
        config = ClipConfig(embed_batch_size=5)
        estimated = config.estimated_vram_usage()
        assert estimated == 0.7


@pytest.mark.unit
class TestFactoryFunctions:
    """Test factory functions for embedding creation."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_create_bgem3_embedding_defaults(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGE-M3 embedding creation with defaults."""
        from src.retrieval.embeddings import create_bgem3_embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = create_bgem3_embedding()

        # Should use default settings
        assert embedding.model_name == "BAAI/bge-m3"
        assert embedding.use_fp16 is True
        assert embedding.device == "cuda"
        assert embedding.max_length == 8192
        assert embedding.batch_size == 12  # GPU batch size

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_create_bgem3_embedding_custom_params(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGE-M3 embedding creation with custom parameters."""
        from src.retrieval.embeddings import create_bgem3_embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = create_bgem3_embedding(
            model_name="custom/model", use_fp16=False, device="cpu", max_length=4096
        )

        # Should use custom parameters
        assert embedding.model_name == "custom/model"
        assert embedding.use_fp16 is False
        assert embedding.device == "cpu"
        assert embedding.max_length == 4096
        assert embedding.batch_size == 4  # CPU batch size

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_create_clip_embedding_dict_config(
        self, mock_clip_class, mock_clip_embedding
    ):
        """Test CLIP embedding creation with dict config."""
        from src.retrieval.embeddings import create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        config_dict = {
            "model_name": "openai/clip-vit-base-patch32",
            "embed_batch_size": 8,
            "device": "cuda",
            "max_vram_gb": 1.2,
        }

        embedding = create_clip_embedding(config_dict)

        assert embedding == mock_clip_embedding

        # Verify ClipEmbedding was called with correct parameters
        mock_clip_class.assert_called_once()
        call_kwargs = mock_clip_class.call_args[1]
        assert call_kwargs["model_name"] == "openai/clip-vit-base-patch32"
        assert call_kwargs["device"] == "cuda"

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_create_clip_embedding_config_object(
        self, mock_clip_class, mock_clip_embedding
    ):
        """Test CLIP embedding creation with ClipConfig object."""
        from src.retrieval.embeddings import ClipConfig, create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        config = ClipConfig(
            model_name="openai/clip-vit-base-patch16", embed_batch_size=6, device="cpu"
        )

        embedding = create_clip_embedding(config)

        assert embedding == mock_clip_embedding
        mock_clip_class.assert_called_once()

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_create_clip_embedding_hardware_optimization(
        self, mock_clip_class, mock_clip_embedding
    ):
        """Test CLIP embedding creation applies hardware optimization."""
        from src.retrieval.embeddings import create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        # Config that needs optimization
        config_dict = {
            "embed_batch_size": 20,  # Too high for VRAM
            "max_vram_gb": 1.4,
            "auto_adjust_batch": True,
            "device": "cuda",
        }

        create_clip_embedding(config_dict)

        # Should be optimized down
        call_kwargs = mock_clip_class.call_args[1]
        assert call_kwargs["embed_batch_size"] <= 10  # Adjusted

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_create_clip_embedding_vram_warning(
        self, mock_clip_class, mock_clip_embedding
    ):
        """Test CLIP embedding creation logs VRAM warning."""
        from src.retrieval.embeddings import create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        # Config that exceeds VRAM limit
        config_dict = {
            "embed_batch_size": 15,  # High batch size
            "max_vram_gb": 1.0,  # Low VRAM limit
            "auto_adjust_batch": False,  # No adjustment
            "device": "cuda",
        }

        with patch("src.retrieval.embeddings.logger") as mock_logger:
            create_clip_embedding(config_dict)

            # Should log warning
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "VRAM usage" in warning_msg
            assert "exceeds limit" in warning_msg


@pytest.mark.unit
class TestLlamaIndexIntegration:
    """Test LlamaIndex Settings integration."""

    @patch("src.retrieval.embeddings.Settings")
    @patch("src.retrieval.embeddings.create_clip_embedding")
    def test_setup_clip_for_llamaindex_default(
        self, mock_create_clip, mock_settings, mock_clip_embedding
    ):
        """Test CLIP setup for LlamaIndex with defaults."""
        from src.retrieval.embeddings import setup_clip_for_llamaindex

        mock_create_clip.return_value = mock_clip_embedding

        result = setup_clip_for_llamaindex()

        # Should create CLIP with default config
        mock_create_clip.assert_called_once()

        # Should set as default embedding model
        assert mock_settings.embed_model == mock_clip_embedding

        # Should return the embedding
        assert result == mock_clip_embedding

    @patch("src.retrieval.embeddings.Settings")
    @patch("src.retrieval.embeddings.create_clip_embedding")
    def test_setup_clip_for_llamaindex_custom_config(
        self, mock_create_clip, mock_settings, mock_clip_embedding
    ):
        """Test CLIP setup with custom configuration."""
        from src.retrieval.embeddings import setup_clip_for_llamaindex

        mock_create_clip.return_value = mock_clip_embedding

        custom_config = {"model_name": "openai/clip-vit-large-patch14", "device": "cpu"}

        result = setup_clip_for_llamaindex(custom_config)

        # Should pass config to create_clip_embedding
        mock_create_clip.assert_called_once()

        # Should set as default
        assert mock_settings.embed_model == mock_clip_embedding
        assert result == mock_clip_embedding

    @patch("src.retrieval.embeddings.Settings")
    @patch("src.retrieval.embeddings.create_bgem3_embedding")
    def test_configure_bgem3_settings_success(
        self, mock_create_bgem3, mock_settings, mock_bgem3_flag_model
    ):
        """Test BGE-M3 LlamaIndex settings configuration."""
        from src.retrieval.embeddings import BGEM3Embedding, configure_bgem3_settings

        mock_embedding = Mock(spec=BGEM3Embedding)
        mock_create_bgem3.return_value = mock_embedding

        configure_bgem3_settings()

        # Should create BGE-M3 embedding
        mock_create_bgem3.assert_called_once()

        # Should set as default embedding model
        assert mock_settings.embed_model == mock_embedding

    @patch("src.retrieval.embeddings.create_bgem3_embedding")
    def test_configure_bgem3_settings_error(self, mock_create_bgem3):
        """Test BGE-M3 settings configuration error handling."""
        from src.retrieval.embeddings import configure_bgem3_settings

        mock_create_bgem3.side_effect = ImportError("FlagEmbedding not available")

        with pytest.raises(ImportError, match="FlagEmbedding not available"):
            configure_bgem3_settings()


@pytest.mark.unit
class TestEmbeddingConstants:
    """Test embedding constants and global settings."""

    def test_embedding_settings_singleton(self):
        """Test embedding settings global instance."""
        from src.retrieval.embeddings import embedding_settings

        # Should be consistent
        assert embedding_settings.model_name == "BAAI/bge-m3"
        assert embedding_settings.dimension == 1024
        assert embedding_settings.max_length == 8192

    def test_clip_constants(self):
        """Test CLIP configuration constants."""
        from src.retrieval.embeddings import (
            DEFAULT_CLIP_BATCH_SIZE,
            MAX_VRAM_GB_LIMIT,
            VRAM_PER_IMAGE_GB,
        )

        # Verify constants are reasonable
        assert VRAM_PER_IMAGE_GB == 0.14
        assert DEFAULT_CLIP_BATCH_SIZE == 10
        assert MAX_VRAM_GB_LIMIT == 1.4

        # Test calculation consistency
        max_images = int(MAX_VRAM_GB_LIMIT / VRAM_PER_IMAGE_GB)
        assert max_images == DEFAULT_CLIP_BATCH_SIZE


@pytest.mark.unit
class TestErrorHandlingAndDeviceManagement:
    """Test error handling and device management."""

    def test_missing_flagembedding_import_error(self):
        """Test handling of missing FlagEmbedding import."""
        from src.retrieval.embeddings import BGEM3Embedding

        with patch("src.retrieval.embeddings.BGEM3FlagModel", None):
            with pytest.raises(ImportError, match="FlagEmbedding not available"):
                BGEM3Embedding()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_model_initialization_errors(self, mock_flag_model_class):
        """Test various model initialization errors."""
        from src.retrieval.embeddings import BGEM3Embedding

        # ImportError
        mock_flag_model_class.side_effect = ImportError("Model not found")
        with pytest.raises(ImportError):
            BGEM3Embedding()

        # RuntimeError
        mock_flag_model_class.side_effect = RuntimeError("CUDA out of memory")
        with pytest.raises(RuntimeError):
            BGEM3Embedding()

        # ValueError
        mock_flag_model_class.side_effect = ValueError("Invalid model path")
        with pytest.raises(ValueError):
            BGEM3Embedding()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_device_management_cuda_available(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test device management when CUDA is available."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        with patch("src.retrieval.embeddings.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            embedding = BGEM3Embedding()

            # Should default to CUDA
            assert embedding.device == "cuda"

            # Model should be initialized with CUDA
            mock_flag_model_class.assert_called_once()
            init_call = mock_flag_model_class.call_args[1]
            assert init_call["device"] == "cuda"

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_device_management_cuda_not_available(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test device management when CUDA is not available."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        with patch("src.retrieval.embeddings.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            embedding = BGEM3Embedding()

            # Should default to CPU
            assert embedding.device == "cpu"


@pytest.mark.unit
class TestPerformanceOptimization:
    """Test performance optimization features."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_batch_size_optimization_gpu(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test GPU batch size optimization."""
        from src.retrieval.embeddings import create_bgem3_embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = create_bgem3_embedding(device="cuda")

        # Should use GPU-optimized batch size
        assert embedding.batch_size == 12

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_batch_size_optimization_cpu(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test CPU batch size optimization."""
        from src.retrieval.embeddings import create_bgem3_embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = create_bgem3_embedding(device="cpu")

        # Should use CPU-optimized batch size
        assert embedding.batch_size == 4

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_fp16_optimization_enabled(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test FP16 optimization when enabled."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding(use_fp16=True, device="cuda")

        # Verify FP16 is enabled
        assert embedding.use_fp16 is True

        # Verify model initialization used FP16
        mock_flag_model_class.assert_called_once()
        init_call = mock_flag_model_class.call_args[1]
        assert init_call["use_fp16"] is True

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_fp16_optimization_disabled(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test FP16 optimization when disabled."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding(use_fp16=False)

        # Verify FP16 is disabled
        assert embedding.use_fp16 is False

        # Verify model initialization disabled FP16
        mock_flag_model_class.assert_called_once()
        init_call = mock_flag_model_class.call_args[1]
        assert init_call["use_fp16"] is False


@pytest.mark.unit
class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_embedding_dimension_consistency(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test embedding dimension consistency across methods."""
        from src.retrieval.embeddings import BGEM3Embedding, embedding_settings

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()

        # All should report consistent dimensions
        assert embedding.embed_dim == 1024
        assert embedding_settings.dimension == 1024

        # Actual embeddings should match
        result = embedding._get_query_embedding("test")
        assert len(result) == 1024

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_batch_processing_large_inputs(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test batch processing with large input sets."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding(batch_size=4)

        # Large input set
        texts = [f"Document {i} content" for i in range(50)]

        result = embedding.get_unified_embeddings(texts)

        # Should handle all inputs
        assert result["dense"].shape[0] == 50
        assert len(result["sparse"]) == 50

        # Should use specified batch size
        mock_bgem3_flag_model.encode.assert_called_once()
        call_kwargs = mock_bgem3_flag_model.encode.call_args[1]
        assert call_kwargs["batch_size"] == 4

    def test_clip_vram_constraint_edge_cases(self):
        """Test CLIP VRAM constraint edge cases."""
        from src.retrieval.embeddings import ClipConfig

        # Minimal VRAM
        config = ClipConfig(max_vram_gb=0.14, auto_adjust_batch=True)
        optimized = config.optimize_for_hardware()

        # Should allow at least 1 image
        assert optimized.embed_batch_size >= 1

        # High VRAM
        config = ClipConfig(max_vram_gb=10.0, auto_adjust_batch=True)
        optimized = config.optimize_for_hardware()

        # Should optimize to reasonable batch size
        expected_max = int(10.0 / 0.14)
        assert optimized.embed_batch_size <= expected_max
