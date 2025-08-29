"""Comprehensive unit tests for retrieval embeddings module.

Tests focus on LlamaIndex integration, factory functions, configuration validation,
and CLIP embedding functionality while mocking heavy ML operations.

Key testing areas:
- BGEM3Embedding (LlamaIndex BaseEmbedding integration)
- Factory functions for embedding creation
- Configuration validation and optimization
- CLIP multimodal embeddings with VRAM constraints
- Error handling and device management
- Performance optimization and batch processing
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_bgem3_flag_model():
    """Mock BGEM3FlagModel for testing."""
    model = Mock()

    def mock_encode(texts, **kwargs):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {
            "dense_vecs": np.random.randn(batch_size, 1024).astype(np.float32),
            "lexical_weights": [
                {i: np.random.random() for i in range(5)} for _ in range(batch_size)
            ],
        }

    model.encode.side_effect = mock_encode
    return model


@pytest.fixture
def mock_clip_embedding():
    """Mock ClipEmbedding for testing."""
    clip_model = Mock()
    clip_model.embed_batch_size = 10
    clip_model.device = "cuda"

    # Mock embedding methods
    clip_model._get_query_embedding.return_value = [
        0.1
    ] * 512  # CLIP ViT-B/32 dimension
    clip_model._get_text_embedding.return_value = [0.1] * 512

    return clip_model


@pytest.mark.unit
class TestEmbeddingConfig:
    """Test embedding configuration validation and defaults."""

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig default values."""
        from src.retrieval.embeddings import EmbeddingConfig

        config = EmbeddingConfig()

        assert config.model_name == "BAAI/bge-m3"
        assert config.dimension == 1024
        assert config.max_length == 8192
        assert config.batch_size_gpu == 12
        assert config.batch_size_cpu == 4

    def test_embedding_config_validation(self):
        """Test EmbeddingConfig field validation."""
        from src.retrieval.embeddings import EmbeddingConfig

        # Valid configuration
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
        from pydantic import ValidationError

        from src.retrieval.embeddings import EmbeddingConfig

        # Test dimension limits
        with pytest.raises(ValidationError):
            EmbeddingConfig(dimension=100)  # Below minimum

        with pytest.raises(ValidationError):
            EmbeddingConfig(dimension=5000)  # Above maximum

        # Test max_length limits
        with pytest.raises(ValidationError):
            EmbeddingConfig(max_length=256)  # Below minimum

        with pytest.raises(ValidationError):
            EmbeddingConfig(max_length=20000)  # Above maximum


@pytest.mark.unit
class TestClipConfig:
    """Test CLIP configuration validation and optimization."""

    def test_clip_config_defaults(self):
        """Test ClipConfig default values."""
        from src.retrieval.embeddings import ClipConfig

        config = ClipConfig()

        assert config.model_name == "openai/clip-vit-base-patch32"
        assert config.embed_batch_size == 10
        assert config.max_vram_gb == 1.4
        assert config.auto_adjust_batch is True

    @patch("src.retrieval.embeddings.torch")
    def test_clip_config_device_detection(self, mock_torch):
        """Test CLIP config device detection."""
        from src.retrieval.embeddings import ClipConfig

        # Test CUDA available
        mock_torch.cuda.is_available.return_value = True
        config = ClipConfig()
        assert config.device == "cuda"

        # Test CUDA not available - need fresh config instance
        with patch("src.retrieval.embeddings.torch") as mock_torch_cpu:
            mock_torch_cpu.cuda.is_available.return_value = False
            config_cpu = ClipConfig()
            assert config_cpu.device == "cpu"

    def test_clip_config_model_validation(self):
        """Test CLIP model name validation."""
        from pydantic import ValidationError

        from src.retrieval.embeddings import ClipConfig

        # Valid models
        valid_models = [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
        ]

        for model in valid_models:
            config = ClipConfig(model_name=model)
            assert config.model_name == model

        # Invalid model
        with pytest.raises(ValidationError, match="Unsupported model"):
            ClipConfig(model_name="invalid/model")

    def test_clip_config_batch_size_adjustment(self):
        """Test automatic batch size adjustment."""
        from src.retrieval.embeddings import ClipConfig

        # Test adjustment for VRAM constraint
        config = ClipConfig(embed_batch_size=20, max_vram_gb=1.4)

        # Should be adjusted down to default (10) due to VRAM limit
        assert config.embed_batch_size == 10

    def test_clip_config_hardware_optimization(self):
        """Test hardware optimization."""
        from src.retrieval.embeddings import ClipConfig

        config = ClipConfig(
            embed_batch_size=20, max_vram_gb=2.0, auto_adjust_batch=True
        )
        optimized = config.optimize_for_hardware()

        # Should optimize batch size based on VRAM
        expected_batch = int(2.0 / 0.14)  # VRAM_PER_IMAGE_GB = 0.14
        assert optimized.embed_batch_size <= expected_batch

    def test_clip_config_vram_estimation(self):
        """Test VRAM usage estimation."""
        from src.retrieval.embeddings import ClipConfig

        config = ClipConfig(embed_batch_size=10)
        estimated = config.estimated_vram_usage()

        # 10 images * 0.14GB per image
        assert estimated == 1.4

    def test_clip_config_validation(self):
        """Test configuration validation."""
        from src.retrieval.embeddings import ClipConfig

        # Valid configuration
        config = ClipConfig(
            model_name="openai/clip-vit-base-patch32",
            embed_batch_size=8,
            max_vram_gb=1.2,
        )
        assert config.is_valid() is True

        # Invalid configuration (over VRAM limit)
        config = ClipConfig(
            model_name="openai/clip-vit-large-patch14",  # Larger model
            embed_batch_size=15,
            max_vram_gb=1.4,
        )
        assert config.is_valid() is False


@pytest.mark.unit
class TestBGEM3Embedding:
    """Test BGEM3Embedding LlamaIndex integration."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_initialization(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGEM3Embedding initialization."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding(
            model_name="BAAI/bge-m3", max_length=8192, use_fp16=True, device="cuda"
        )

        assert embedding.model_name == "BAAI/bge-m3"
        assert embedding.max_length == 8192
        assert embedding.use_fp16 is True
        assert embedding.device == "cuda"
        assert embedding.embed_dim == 1024

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_missing_flagembedding(self, mock_flag_model_class):
        """Test BGEM3Embedding handles missing FlagEmbedding."""
        from src.retrieval.embeddings import BGEM3Embedding

        with patch("src.retrieval.embeddings.BGEM3FlagModel", None):
            with pytest.raises(ImportError, match="FlagEmbedding not available"):
                BGEM3Embedding()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_model_load_failure(self, mock_flag_model_class):
        """Test BGEM3Embedding handles model loading failure."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.side_effect = RuntimeError("Model loading failed")

        with pytest.raises(RuntimeError):
            BGEM3Embedding()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_query_embedding(self, mock_flag_model_class, mock_bgem3_flag_model):
        """Test query embedding generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()
        query = "What is machine learning?"

        result = embedding._get_query_embedding(query)

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)

        # Verify model called with correct parameters
        mock_bgem3_flag_model.encode.assert_called_with(
            [query],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_text_embedding(self, mock_flag_model_class, mock_bgem3_flag_model):
        """Test text embedding generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()
        text = "Document text for embedding generation."

        result = embedding._get_text_embedding(text)

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)

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
    def test_get_unified_embeddings(self, mock_flag_model_class, mock_bgem3_flag_model):
        """Test unified embeddings generation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()
        texts = ["Text 1", "Text 2", "Text 3"]

        result = embedding.get_unified_embeddings(
            texts, return_dense=True, return_sparse=True, return_colbert=False
        )

        assert "dense" in result
        assert "sparse" in result
        assert "colbert" not in result

        # Verify dense embeddings
        assert result["dense"].shape == (3, 1024)

        # Verify sparse embeddings
        assert len(result["sparse"]) == 3
        for sparse_emb in result["sparse"]:
            assert isinstance(sparse_emb, dict)

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_get_sparse_embedding(self, mock_flag_model_class, mock_bgem3_flag_model):
        """Test single text sparse embedding."""
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
    def test_get_unified_embeddings_error_handling(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test unified embeddings error handling."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode.side_effect = RuntimeError("Model error")

        embedding = BGEM3Embedding()
        texts = ["Test text"]

        with pytest.raises(RuntimeError):
            embedding.get_unified_embeddings(texts)


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

        assert embedding.model_name == "BAAI/bge-m3"
        assert embedding.max_length == 8192
        assert embedding.use_fp16 is True
        assert embedding.device == "cuda"
        assert embedding.batch_size == 12  # GPU batch size

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_create_bgem3_embedding_custom_params(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGE-M3 embedding creation with custom parameters."""
        from src.retrieval.embeddings import create_bgem3_embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = create_bgem3_embedding(
            model_name="custom/model",
            use_fp16=False,
            device="cpu",
            max_length=4096,
        )

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

        config = {
            "model_name": "openai/clip-vit-base-patch32",
            "embed_batch_size": 8,
            "device": "cpu",
            "max_vram_gb": 1.0,
        }

        embedding = create_clip_embedding(config)

        assert embedding is not None
        mock_clip_class.assert_called_once()

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_create_clip_embedding_config_object(
        self, mock_clip_class, mock_clip_embedding
    ):
        """Test CLIP embedding creation with ClipConfig object."""
        from src.retrieval.embeddings import ClipConfig, create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        config = ClipConfig(
            model_name="openai/clip-vit-base-patch16",
            embed_batch_size=6,
            device="cuda",
        )

        embedding = create_clip_embedding(config)

        assert embedding is not None
        mock_clip_class.assert_called_once()

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_create_clip_embedding_hardware_optimization(
        self, mock_clip_class, mock_clip_embedding
    ):
        """Test CLIP embedding creation with hardware optimization."""
        from src.retrieval.embeddings import ClipConfig, create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        # Config that needs optimization
        config = ClipConfig(
            embed_batch_size=20,  # Too high
            max_vram_gb=1.4,
            auto_adjust_batch=True,
            device="cuda",
        )

        create_clip_embedding(config)

        # Should be optimized
        call_kwargs = mock_clip_class.call_args[1]
        assert call_kwargs["embed_batch_size"] <= 10  # Adjusted down

    @patch("src.retrieval.embeddings.Settings")
    @patch("src.retrieval.embeddings.create_clip_embedding")
    def test_setup_clip_for_llamaindex(
        self, mock_create_clip, mock_settings, mock_clip_embedding
    ):
        """Test CLIP setup for LlamaIndex."""
        from src.retrieval.embeddings import setup_clip_for_llamaindex

        mock_create_clip.return_value = mock_clip_embedding

        result = setup_clip_for_llamaindex()

        assert result == mock_clip_embedding
        assert mock_settings.embed_model == mock_clip_embedding

    @patch("src.retrieval.embeddings.Settings")
    @patch("src.retrieval.embeddings.create_clip_embedding")
    def test_setup_clip_for_llamaindex_custom_config(
        self, mock_create_clip, mock_settings, mock_clip_embedding
    ):
        """Test CLIP setup with custom configuration."""
        from src.retrieval.embeddings import setup_clip_for_llamaindex

        mock_create_clip.return_value = mock_clip_embedding

        config = {"model_name": "openai/clip-vit-large-patch14", "device": "cpu"}

        result = setup_clip_for_llamaindex(config)

        assert result == mock_clip_embedding
        mock_create_clip.assert_called_once()

    @patch("src.retrieval.embeddings.Settings")
    @patch("src.retrieval.embeddings.create_bgem3_embedding")
    def test_configure_bgem3_settings(
        self, mock_create_bgem3, mock_settings, mock_bgem3_flag_model
    ):
        """Test BGE-M3 LlamaIndex settings configuration."""
        from src.retrieval.embeddings import BGEM3Embedding, configure_bgem3_settings

        mock_embedding = Mock(spec=BGEM3Embedding)
        mock_create_bgem3.return_value = mock_embedding

        configure_bgem3_settings()

        assert mock_settings.embed_model == mock_embedding
        mock_create_bgem3.assert_called_once()

    @patch("src.retrieval.embeddings.create_bgem3_embedding")
    def test_configure_bgem3_settings_error_handling(self, mock_create_bgem3):
        """Test BGE-M3 settings configuration error handling."""
        from src.retrieval.embeddings import configure_bgem3_settings

        mock_create_bgem3.side_effect = ImportError("FlagEmbedding not available")

        with pytest.raises(ImportError):
            configure_bgem3_settings()


@pytest.mark.unit
class TestEmbeddingIntegration:
    """Test embedding integration with LlamaIndex and performance."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_llamaindex_compatibility(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGE-M3 embedding LlamaIndex compatibility."""
        from llama_index.core.base.embeddings.base import BaseEmbedding

        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()

        # Should be instance of BaseEmbedding
        assert isinstance(embedding, BaseEmbedding)

        # Should have required properties
        assert hasattr(embedding, "embed_dim")
        assert hasattr(embedding, "_get_query_embedding")
        assert hasattr(embedding, "_get_text_embedding")
        assert hasattr(embedding, "_aget_query_embedding")

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_batch_processing(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGE-M3 embedding batch processing."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding(batch_size=8)

        texts = [f"Document {i} for batch processing test." for i in range(20)]

        result = embedding.get_unified_embeddings(texts)

        # Should handle all texts in batch
        assert result["dense"].shape[0] == 20
        assert len(result["sparse"]) == 20

        # Verify correct batch size used
        call_kwargs = mock_bgem3_flag_model.encode.call_args[1]
        assert call_kwargs["batch_size"] == 8

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_dimension_validation(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGE-M3 embedding dimension validation."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()

        # Test single embedding
        result = embedding._get_query_embedding("Test query")
        assert len(result) == 1024

        # Test batch embeddings
        batch_result = embedding.get_unified_embeddings(["Text 1", "Text 2"])
        assert batch_result["dense"].shape[1] == 1024

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_bgem3_embedding_device_management(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test BGE-M3 embedding device management."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        # Test CUDA device
        embedding_cuda = BGEM3Embedding(device="cuda")
        assert embedding_cuda.device == "cuda"

        # Test CPU device
        embedding_cpu = BGEM3Embedding(device="cpu")
        assert embedding_cpu.device == "cpu"

        # Verify model initialization with correct device
        cuda_call = mock_flag_model_class.call_args_list[-2]
        cpu_call = mock_flag_model_class.call_args_list[-1]

        assert cuda_call[1]["device"] == "cuda"
        assert cpu_call[1]["device"] == "cpu"


@pytest.mark.unit
class TestPerformanceOptimization:
    """Test performance optimization features."""

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_fp16_acceleration(self, mock_flag_model_class, mock_bgem3_flag_model):
        """Test FP16 acceleration configuration."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        # Test FP16 enabled
        embedding_fp16 = BGEM3Embedding(use_fp16=True, device="cuda")
        assert embedding_fp16.use_fp16 is True

        # Test FP16 disabled
        embedding_fp32 = BGEM3Embedding(use_fp16=False)
        assert embedding_fp32.use_fp16 is False

        # Verify model initialization with FP16 settings
        fp16_call = mock_flag_model_class.call_args_list[-2]
        fp32_call = mock_flag_model_class.call_args_list[-1]

        assert fp16_call[1]["use_fp16"] is True
        assert fp32_call[1]["use_fp16"] is False

    def test_batch_size_optimization(self):
        """Test batch size optimization for different devices."""
        from src.retrieval.embeddings import create_bgem3_embedding

        with patch("src.retrieval.embeddings.BGEM3FlagModel") as mock_flag_model_class:
            mock_model = Mock()
            mock_flag_model_class.return_value = mock_model

            # Test GPU batch size
            gpu_embedding = create_bgem3_embedding(device="cuda")
            assert gpu_embedding.batch_size == 12

            # Test CPU batch size
            cpu_embedding = create_bgem3_embedding(device="cpu")
            assert cpu_embedding.batch_size == 4

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_clip_vram_optimization(self, mock_clip_class, mock_clip_embedding):
        """Test CLIP VRAM optimization."""
        from src.retrieval.embeddings import ClipConfig, create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        # Test automatic batch size adjustment
        config = ClipConfig(
            embed_batch_size=15,  # Over limit
            max_vram_gb=1.4,
            auto_adjust_batch=True,
        )

        create_clip_embedding(config)

        # Should be adjusted down
        call_kwargs = mock_clip_class.call_args[1]
        assert call_kwargs["embed_batch_size"] <= 10


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling and recovery scenarios."""

    def test_missing_flagembedding_import(self):
        """Test handling of missing FlagEmbedding import."""
        from src.retrieval.embeddings import BGEM3Embedding

        with patch("src.retrieval.embeddings.BGEM3FlagModel", None):
            with pytest.raises(ImportError, match="FlagEmbedding not available"):
                BGEM3Embedding()

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_model_loading_error(self, mock_flag_model_class):
        """Test model loading error handling."""
        from src.retrieval.embeddings import BGEM3Embedding

        mock_flag_model_class.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError):
            BGEM3Embedding()

    def test_invalid_clip_model_name(self):
        """Test invalid CLIP model name handling."""
        from pydantic import ValidationError

        from src.retrieval.embeddings import ClipConfig

        with pytest.raises(ValidationError, match="Unsupported model"):
            ClipConfig(model_name="invalid/model-name")

    @patch("src.retrieval.embeddings.ClipEmbedding")
    def test_clip_vram_warning(self, mock_clip_class, mock_clip_embedding):
        """Test CLIP VRAM usage warning."""
        from src.retrieval.embeddings import ClipConfig, create_clip_embedding

        mock_clip_class.return_value = mock_clip_embedding

        # Config that exceeds VRAM limit
        config = ClipConfig(
            embed_batch_size=20,  # High batch size
            max_vram_gb=1.0,  # Low VRAM limit
            auto_adjust_batch=False,  # Disable auto adjustment
        )

        with patch("src.retrieval.embeddings.logger") as mock_logger:
            create_clip_embedding(config)

            # Should log warning about VRAM usage
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "VRAM usage" in warning_call
            assert "exceeds limit" in warning_call


@pytest.mark.unit
class TestConfigurationConsistency:
    """Test configuration consistency and validation."""

    def test_embedding_settings_singleton(self):
        """Test embedding settings singleton behavior."""
        from src.retrieval.embeddings import embedding_settings

        # Should be consistent across imports
        assert embedding_settings.model_name == "BAAI/bge-m3"
        assert embedding_settings.dimension == 1024
        assert embedding_settings.max_length == 8192

    def test_clip_config_consistency(self):
        """Test CLIP configuration consistency."""
        from src.retrieval.embeddings import (
            DEFAULT_CLIP_BATCH_SIZE,
            MAX_VRAM_GB_LIMIT,
            VRAM_PER_IMAGE_GB,
            ClipConfig,
        )

        config = ClipConfig()

        # Test constants consistency
        assert config.embed_batch_size == DEFAULT_CLIP_BATCH_SIZE
        assert config.max_vram_gb == MAX_VRAM_GB_LIMIT

        # Test VRAM calculation consistency
        expected_vram = config.embed_batch_size * VRAM_PER_IMAGE_GB
        actual_vram = config.estimated_vram_usage()
        assert actual_vram == expected_vram

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_dimension_consistency_across_methods(
        self, mock_flag_model_class, mock_bgem3_flag_model
    ):
        """Test dimension consistency across all methods."""
        from src.retrieval.embeddings import BGEM3Embedding, embedding_settings

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedding = BGEM3Embedding()

        # All should report 1024 dimensions
        assert embedding.embed_dim == 1024
        assert embedding_settings.dimension == 1024

        # Verify actual embedding dimensions
        result = embedding._get_query_embedding("Test")
        assert len(result) == 1024
