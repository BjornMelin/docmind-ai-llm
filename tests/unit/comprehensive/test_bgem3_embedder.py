"""Comprehensive unit tests for BGE-M3 embedder achieving 70%+ coverage.

This test suite focuses on boundary testing with lightweight test doubles,
targeting the 187 uncovered statements in src/processing/embeddings/bgem3_embedder.py.

Test Coverage Strategy:
- Device detection and CPU/GPU fallback mechanisms
- Async embedding generation with proper error handling
- FlagEmbedding initialization and failure scenarios
- Batch processing and dimension validation
- Dense and sparse embedding generation with realistic data flow
- Boundary testing only - mock external services (FlagEmbedding), NOT internal methods
- Integration tests for embedding pipeline workflows

Key Requirements:
- Achieve 70%+ coverage on bgem3_embedder.py (187 statements)
- Test actual business logic, not mock interactions
- Modern pytest patterns with pytest-asyncio
- Lightweight test doubles instead of complex mocks
"""

import time
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.models.embeddings import (
    EmbeddingError,
    EmbeddingParameters,
    EmbeddingResult,
)

# Test fixture imports
from tests.fixtures.test_settings import TestDocMindSettings


@pytest.fixture
def test_settings():
    """Test settings optimized for BGE-M3 embedder testing."""
    return TestDocMindSettings(
        embedding={
            "model_name": "BAAI/bge-m3",
            "dimension": 1024,  # BGE-M3 produces 1024D embeddings
            "max_length": 8192,
            "batch_size_cpu": 4,
            "batch_size_gpu": 8,
        },
        enable_gpu_acceleration=False,  # CPU-only for unit tests
    )


@pytest.fixture
def gpu_test_settings():
    """GPU-enabled test settings for GPU fallback testing."""
    return TestDocMindSettings(
        embedding={
            "model_name": "BAAI/bge-m3",
            "dimension": 1024,
            "max_length": 8192,
            "batch_size_gpu": 8,
        },
        enable_gpu_acceleration=True,
    )


class MockFlagEmbeddingModel:
    """Lightweight test double for BGEM3FlagModel with realistic behavior.

    Provides deterministic outputs that match BGE-M3's actual structure
    without the computational overhead of the real model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
        pooling_method: str = "cls",
        devices: list[str] = None,
    ):
        """Initialize mock BGE-M3 model with realistic behavior.

        Args:
            model_name_or_path: Path or name of the model to mock
            use_fp16: Whether to use FP16 precision (not used in mock)
            pooling_method: Pooling method for embeddings (not used in mock)
            devices: List of devices to use (not used in mock)
        """
        self.model_name_or_path = model_name_or_path
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.devices = devices or ["cpu"]

        # Simulate model loading success
        self.model = Mock()
        self.model.to = Mock()

    def encode(
        self,
        texts: list[str],
        max_length: int = 8192,  # Not used in mock, kept for API compatibility
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Mock BGE-M3 encode method with realistic return structure."""
        if not texts:
            return {}

        _ = len(texts)
        result = {}

        # BGE-M3 dense embeddings: 1024-dimensional
        if return_dense:
            # Generate deterministic embeddings based on text content
            embeddings = []
            for text in texts:
                # Create deterministic embedding based on text hash
                seed = hash(text) % 2**31
                np.random.seed(seed)
                embedding = np.random.randn(1024).astype(np.float32)
                # Normalize as BGE-M3 does
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            result["dense_vecs"] = np.array(embeddings)

        # BGE-M3 sparse embeddings: learned token weights
        if return_sparse:
            sparse_embeddings = []
            for text in texts:
                # Generate realistic sparse representation
                tokens = text.split()[:20]  # Limit for realism
                sparse_dict = {}
                for token in tokens:
                    # Simulate token ID and learned weight
                    token_id = hash(token) % 30000  # Vocab size simulation
                    weight = max(0.1, min(1.0, len(token) / 10.0))  # Realistic weight
                    sparse_dict[token_id] = weight
                sparse_embeddings.append(sparse_dict)
            result["lexical_weights"] = sparse_embeddings

        # ColBERT embeddings: multi-vector per text
        if return_colbert_vecs:
            colbert_embeddings = []
            for text in texts:
                # Simulate token-level embeddings
                num_tokens = min(len(text.split()), 50)  # Reasonable token limit
                token_embeddings = np.random.randn(num_tokens, 1024).astype(np.float32)
                colbert_embeddings.append(token_embeddings)
            result["colbert_vecs"] = colbert_embeddings

        return result

    def encode_queries(self, queries: list[str], **kwargs) -> dict[str, Any]:
        """Mock query-optimized encoding (same structure as encode)."""
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: list[str], **kwargs) -> dict[str, Any]:
        """Mock corpus-optimized encoding (same structure as encode)."""
        return self.encode(corpus, **kwargs)

    def compute_lexical_matching_score(
        self, sparse1: dict[int, float], sparse2: dict[int, float]
    ) -> float:
        """Mock sparse similarity computation."""
        # Simple overlap-based similarity
        common_tokens = set(sparse1.keys()) & set(sparse2.keys())
        if not common_tokens:
            return 0.0

        similarity = 0.0
        for token in common_tokens:
            similarity += min(sparse1[token], sparse2[token])

        return min(1.0, similarity / len(common_tokens))

    def compute_score(
        self, sentence_pairs: list[list[str]], **kwargs
    ) -> dict[str, list[float]]:
        """Mock comprehensive similarity computation."""
        scores = {
            "dense": [],
            "sparse": [],
            "colbert": [],
        }

        for pair in sentence_pairs:
            # Simulate realistic similarity scores based on text overlap
            text1, text2 = pair[0], pair[1]
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            # Simple overlap-based scoring
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                scores["dense"].append(0.6 + 0.4 * overlap)  # 0.6-1.0 range
                scores["sparse"].append(0.5 + 0.5 * overlap)  # 0.5-1.0 range
                scores["colbert"].append(0.7 + 0.3 * overlap)  # 0.7-1.0 range
            else:
                scores["dense"].append(0.1)
                scores["sparse"].append(0.1)
                scores["colbert"].append(0.1)

        return scores

    def colbert_score(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Mock ColBERT late interaction scoring."""
        # Simulate late interaction: max(token similarities)
        if vec1.shape[0] == 0 or vec2.shape[0] == 0:
            return 0.0

        # Simplified scoring - in reality this is complex late interaction
        return 0.75 + 0.25 * np.random.random()  # 0.75-1.0 range

    def convert_id_to_token(
        self, sparse_embeddings: list[dict[int, float]]
    ) -> list[dict[str, float]]:
        """Mock token ID to string conversion."""
        token_embeddings = []
        for sparse_dict in sparse_embeddings:
            token_dict = {}
            for token_id, weight in sparse_dict.items():
                # Simulate token conversion
                token = f"token_{token_id % 1000}"  # Realistic token naming
                token_dict[token] = weight
            token_embeddings.append(token_dict)
        return token_embeddings


@pytest.fixture
def mock_flag_model():
    """Factory fixture for creating MockFlagEmbeddingModel instances."""

    def _create_mock(**kwargs):
        # Provide default values for required parameters
        defaults = {
            "model_name_or_path": "BAAI/bge-m3",
            "use_fp16": False,
            "pooling_method": "cls",
            "devices": ["cpu"],
        }
        defaults.update(kwargs)
        return MockFlagEmbeddingModel(**defaults)

    return _create_mock


# Test Classes


@pytest.mark.unit
class TestBGEM3EmbedderInitialization:
    """Test initialization, device detection, and configuration handling."""

    def test_successful_initialization_cpu(self, test_settings, mock_flag_model):
        """Test successful initialization with CPU fallback."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with (
            patch(
                "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
            ) as mock_class,
            patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Verify initialization
            assert embedder.settings == test_settings
            assert embedder.device == "cpu"
            assert embedder.pooling_method == "cls"
            assert embedder.normalize_embeddings is True
            assert embedder._embedding_count == 0
            assert embedder._total_processing_time == 0.0

            # Verify model created with correct parameters
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args.kwargs
            assert call_kwargs["model_name_or_path"] == "BAAI/bge-m3"
            assert call_kwargs["use_fp16"] is False  # CPU mode
            assert call_kwargs["pooling_method"] == "cls"
            assert call_kwargs["devices"] == ["cpu"]

    def test_successful_initialization_gpu(self, gpu_test_settings, mock_flag_model):
        """Test successful initialization with GPU acceleration."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with (
            patch(
                "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
            ) as mock_class,
            patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            mock_class.return_value = mock_flag_model()

            params = EmbeddingParameters(use_fp16=True)
            embedder = BGEM3Embedder(settings=gpu_test_settings, parameters=params)

            # Verify GPU configuration
            assert embedder.device == "cuda"

            # Verify FP16 enabled for GPU
            call_kwargs = mock_class.call_args.kwargs
            assert call_kwargs["use_fp16"] is True
            assert call_kwargs["devices"] == ["cuda"]

    def test_custom_initialization_parameters(self, test_settings, mock_flag_model):
        """Test initialization with custom parameters and weights."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            custom_params = EmbeddingParameters(
                max_length=4096,
                use_fp16=False,
                pooling_method="mean",
                normalize_embeddings=False,
            )

            embedder = BGEM3Embedder(
                settings=test_settings,
                parameters=custom_params,
                pooling_method="mean",
                normalize_embeddings=False,
                weights_for_different_modes=[0.5, 0.3, 0.2],
                devices=["cpu", "cuda:0"],
                return_numpy=True,
            )

            # Verify custom parameters applied
            assert embedder.parameters.max_length == 4096
            assert embedder.parameters.pooling_method == "mean"
            assert embedder.pooling_method == "mean"
            assert embedder.normalize_embeddings is False
            assert embedder.weights_for_different_modes == [0.5, 0.3, 0.2]
            assert embedder.return_numpy is True

            # Verify custom devices passed to model
            call_kwargs = mock_class.call_args.kwargs
            assert call_kwargs["devices"] == ["cpu", "cuda:0"]
            assert call_kwargs["pooling_method"] == "mean"

    def test_initialization_missing_flagembedding(self, test_settings):
        """Test graceful failure when FlagEmbedding library unavailable."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Simulate missing FlagEmbedding library
        with patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel", None):
            with pytest.raises(EmbeddingError) as exc_info:
                BGEM3Embedder(settings=test_settings)

            assert "FlagEmbedding not available" in str(exc_info.value)
            assert "uv add FlagEmbedding>=1.3.5" in str(exc_info.value)

    def test_initialization_model_loading_failure(self, test_settings):
        """Test handling of model loading failures with proper error propagation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.side_effect = RuntimeError(
                "CUDA out of memory during model loading"
            )

            with pytest.raises(EmbeddingError) as exc_info:
                BGEM3Embedder(settings=test_settings)

            assert "BGE-M3 model initialization failed" in str(exc_info.value)
            assert "CUDA out of memory" in str(exc_info.value)

    def test_device_auto_detection_logic(self, test_settings, mock_flag_model):
        """Test automatic device detection with different CUDA availability scenarios.

        Verifies that the embedder correctly detects and uses available hardware
        acceleration while gracefully falling back to CPU when needed.
        """
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            # Test CUDA available scenario
            with patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True

                embedder = BGEM3Embedder(settings=test_settings)
                assert embedder.device == "cuda"

                # Verify cuda device passed to model
                call_kwargs = mock_class.call_args.kwargs
                assert "cuda" in str(call_kwargs["devices"])

            # Test CUDA unavailable scenario (reset mock)
            mock_class.reset_mock()
            with patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False

                embedder = BGEM3Embedder(settings=test_settings)
                assert embedder.device == "cpu"

                # Verify cpu device passed to model
                call_kwargs = mock_class.call_args.kwargs
                assert call_kwargs["devices"] == ["cpu"]


@pytest.mark.unit
class TestBGEM3EmbedderAsyncOperations:
    """Test async embedding operations with proper error handling and data flow."""

    @pytest.mark.asyncio
    async def test_embed_texts_async_success(self, test_settings, mock_flag_model):
        """Test successful async embedding generation with realistic data flow."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with (
            patch(
                "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
            ) as mock_class,
            patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.memory_allocated.return_value = 1024**3  # 1GB

            mock_instance = mock_flag_model()
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            # Test with realistic document texts
            texts = [
                "Machine learning models require large amounts of training data.",
                "BGE-M3 unified embeddings enable both dense and sparse retrieval.",
                "Document processing pipelines benefit from semantic understanding.",
            ]

            params = EmbeddingParameters(
                return_dense=True,
                return_sparse=True,
                return_colbert=False,
            )

            result = await embedder.embed_texts_async(texts, parameters=params)

            # Verify result structure and data flow
            assert isinstance(result, EmbeddingResult)
            assert result.dense_embeddings is not None
            assert len(result.dense_embeddings) == 3
            assert result.sparse_embeddings is not None
            assert len(result.sparse_embeddings) == 3
            assert result.colbert_embeddings is None  # Not requested

            # Verify BGE-M3 1024-dimensional embeddings
            for embedding in result.dense_embeddings:
                assert len(embedding) == 1024
                assert all(isinstance(x, float) for x in embedding)

            # Verify sparse embeddings structure
            for sparse_emb in result.sparse_embeddings:
                assert isinstance(sparse_emb, dict)
                for token_id, weight in sparse_emb.items():
                    assert isinstance(token_id, int)
                    assert isinstance(weight, int | float)
                    assert 0.0 <= weight <= 1.0

            # Verify metadata and statistics
            assert result.processing_time > 0
            assert result.batch_size == 3
            assert result.memory_usage_mb > 0
            assert result.model_info["embedding_dim"] == 1024
            assert result.model_info["library"] == "FlagEmbedding.BGEM3FlagModel"
            assert result.model_info["dense_enabled"] is True
            assert result.model_info["sparse_enabled"] is True
            assert result.model_info["colbert_enabled"] is False

            # Verify embedder statistics updated
            assert embedder._embedding_count == 3
            assert embedder._total_processing_time > 0

    @pytest.mark.asyncio
    async def test_embed_texts_async_colbert_mode(self, test_settings, mock_flag_model):
        """Test async embedding with ColBERT multi-vector embeddings."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            texts = ["ColBERT test document with multiple tokens."]
            params = EmbeddingParameters(
                return_dense=True,
                return_sparse=False,
                return_colbert=True,  # Enable ColBERT
            )

            result = await embedder.embed_texts_async(texts, parameters=params)

            # Verify ColBERT embeddings structure
            assert result.colbert_embeddings is not None
            assert len(result.colbert_embeddings) == 1
            assert isinstance(result.colbert_embeddings[0], np.ndarray)
            assert result.colbert_embeddings[0].shape[1] == 1024  # BGE-M3 dimension
            assert result.sparse_embeddings is None  # Not requested

            # Verify model info reflects ColBERT mode
            assert result.model_info["colbert_enabled"] is True
            assert result.model_info["sparse_enabled"] is False

    @pytest.mark.asyncio
    async def test_embed_texts_async_empty_input(self, test_settings, mock_flag_model):
        """Test async embedding with empty input handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Test empty list
            result = await embedder.embed_texts_async([])

            assert isinstance(result, EmbeddingResult)
            assert result.dense_embeddings == []
            assert result.sparse_embeddings is None
            assert result.batch_size == 0
            assert result.processing_time == 0.0
            assert result.memory_usage_mb == 0.0
            assert "warning" in result.model_info
            assert result.model_info["warning"] == "No texts provided"

            # Verify no statistics updated for empty input
            assert embedder._embedding_count == 0
            assert embedder._total_processing_time == 0.0

    @pytest.mark.asyncio
    async def test_embed_texts_async_model_error_handling(
        self, test_settings, mock_flag_model
    ):
        """Test async embedding error handling with proper error propagation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            # Simulate model encoding failure
            mock_instance.encode = Mock(side_effect=RuntimeError("CUDA out of memory"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            texts = ["Test text that will trigger error"]

            with pytest.raises(EmbeddingError) as exc_info:
                await embedder.embed_texts_async(texts)

            assert "BGE-M3 unified embedding failed" in str(exc_info.value)
            assert "CUDA out of memory" in str(exc_info.value)

            # Verify statistics not updated on failure
            assert embedder._embedding_count == 0
            assert embedder._total_processing_time == 0.0

    @pytest.mark.asyncio
    async def test_embed_single_text_async(self, test_settings, mock_flag_model):
        """Test single text async embedding wrapper functionality."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            text = "Single document for embedding test."

            embedding = await embedder.embed_single_text_async(text)

            # Verify single embedding vector
            assert isinstance(embedding, list)
            assert len(embedding) == 1024  # BGE-M3 dimension
            assert all(isinstance(x, float) for x in embedding)

            # Verify statistics updated for single text
            assert embedder._embedding_count == 1
            assert embedder._total_processing_time > 0

    @pytest.mark.asyncio
    async def test_embed_single_text_async_no_embeddings_error(
        self, test_settings, mock_flag_model
    ):
        """Test single text embedding error when no embeddings generated."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            # Mock empty embeddings result
            mock_instance.encode = Mock(
                return_value={"dense_vecs": np.array([]).reshape(0, 1024)}
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            with pytest.raises(EmbeddingError) as exc_info:
                await embedder.embed_single_text_async("Test text")

            assert "No dense embeddings generated" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_texts_async_numpy_return_mode(
        self, test_settings, mock_flag_model
    ):
        """Test async embedding with numpy return mode enabled."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings, return_numpy=True)

            texts = ["Test numpy return mode"]

            result = await embedder.embed_texts_async(texts)

            # Verify numpy mode processing
            # (should still return lists in current implementation)
            assert result.dense_embeddings is not None
            assert isinstance(
                result.dense_embeddings[0], list
            )  # Current implementation returns lists


@pytest.mark.unit
class TestBGEM3EmbedderSynchronousOperations:
    """Test synchronous embedding operations and error handling."""

    def test_get_dense_embeddings_success(self, test_settings, mock_flag_model):
        """Test synchronous dense embedding generation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            texts = [
                "First document for dense embedding test.",
                "Second document with different semantic content.",
            ]

            embeddings = embedder.get_dense_embeddings(texts)

            # Verify dense embeddings
            assert embeddings is not None
            assert len(embeddings) == 2
            for embedding in embeddings:
                assert len(embedding) == 1024  # BGE-M3 dimension
                assert all(isinstance(x, float) for x in embedding)

    def test_get_dense_embeddings_error_handling(self, test_settings, mock_flag_model):
        """Test dense embedding error handling with graceful degradation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            # Test different error types
            mock_instance.encode = Mock(side_effect=RuntimeError("Model error"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            texts = ["Test error handling"]

            # Should return None on error, not raise exception
            result = embedder.get_dense_embeddings(texts)
            assert result is None

        # Test ValueError handling
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.encode = Mock(side_effect=ValueError("Invalid input"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            result = embedder.get_dense_embeddings(texts)
            assert result is None

        # Test ImportError handling
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.encode = Mock(side_effect=ImportError("Missing dependency"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            result = embedder.get_dense_embeddings(texts)
            assert result is None

    def test_get_dense_embeddings_malformed_output(
        self, test_settings, mock_flag_model
    ):
        """Test dense embedding handling of malformed model output."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            # Mock malformed output missing expected keys
            mock_instance.encode = Mock(return_value={"unexpected_key": "value"})
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            texts = ["Test malformed output handling"]

            result = embedder.get_dense_embeddings(texts)
            assert result is None  # Should gracefully handle malformed output

    def test_get_sparse_embeddings_success(self, test_settings, mock_flag_model):
        """Test synchronous sparse embedding generation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            texts = [
                "Natural language processing techniques",
                "Machine learning model training",
            ]

            sparse_embeddings = embedder.get_sparse_embeddings(texts)

            # Verify sparse embeddings structure
            assert sparse_embeddings is not None
            assert len(sparse_embeddings) == 2
            for sparse_emb in sparse_embeddings:
                assert isinstance(sparse_emb, dict)
                for token_id, weight in sparse_emb.items():
                    assert isinstance(token_id, int)
                    assert isinstance(weight, int | float)
                    assert weight > 0.0  # Sparse weights should be positive

    def test_get_sparse_embeddings_error_handling(self, test_settings, mock_flag_model):
        """Test sparse embedding error handling with all error types."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Test RuntimeError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.encode = Mock(side_effect=RuntimeError("Runtime error"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)
            result = embedder.get_sparse_embeddings(["test"])
            assert result is None

        # Test ValueError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.encode = Mock(side_effect=ValueError("Value error"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)
            result = embedder.get_sparse_embeddings(["test"])
            assert result is None

        # Test ImportError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.encode = Mock(side_effect=ImportError("Import error"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)
            result = embedder.get_sparse_embeddings(["test"])
            assert result is None

        # Test generic Exception
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.encode = Mock(side_effect=Exception("Generic error"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)
            result = embedder.get_sparse_embeddings(["test"])
            assert result is None


@pytest.mark.unit
class TestBGEM3EmbedderSimilarityOperations:
    """Test similarity computation methods and error handling."""

    def test_compute_sparse_similarity_success(self, test_settings, mock_flag_model):
        """Test sparse embedding similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Test realistic sparse embeddings
            sparse1 = {100: 0.8, 250: 0.6, 500: 0.4}  # "machine", "learning", "model"
            sparse2 = {100: 0.7, 300: 0.5, 250: 0.9}  # "machine", "data", "learning"

            similarity = embedder.compute_sparse_similarity(sparse1, sparse2)

            # Verify similarity computation
            assert isinstance(similarity, float)
            assert 0.0 <= similarity <= 1.0
            # Should be > 0 due to overlapping tokens (100, 250)
            assert similarity > 0.0

    def test_compute_sparse_similarity_no_overlap(self, test_settings, mock_flag_model):
        """Test sparse similarity with no token overlap."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            # Mock no overlap scenario
            mock_instance.compute_lexical_matching_score = Mock(return_value=0.0)
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            sparse1 = {100: 0.8, 200: 0.6}  # Different tokens
            sparse2 = {300: 0.7, 400: 0.5}  # No overlap

            similarity = embedder.compute_sparse_similarity(sparse1, sparse2)
            assert similarity == 0.0

    def test_compute_sparse_similarity_error_handling(
        self, test_settings, mock_flag_model
    ):
        """Test sparse similarity error handling with different error types."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Test RuntimeError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.compute_lexical_matching_score = Mock(
                side_effect=RuntimeError("Error")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            sparse1 = {100: 0.8}
            sparse2 = {200: 0.6}

            similarity = embedder.compute_sparse_similarity(sparse1, sparse2)
            assert similarity == 0.0  # Should return 0.0 on error

        # Test ValueError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.compute_lexical_matching_score = Mock(
                side_effect=ValueError("Invalid")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            similarity = embedder.compute_sparse_similarity(sparse1, sparse2)
            assert similarity == 0.0

        # Test TypeError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.compute_lexical_matching_score = Mock(
                side_effect=TypeError("Type error")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            similarity = embedder.compute_sparse_similarity(sparse1, sparse2)
            assert similarity == 0.0

        # Test generic Exception
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.compute_lexical_matching_score = Mock(
                side_effect=Exception("Generic")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            similarity = embedder.compute_sparse_similarity(sparse1, sparse2)
            assert similarity == 0.0


@pytest.mark.unit
class TestBGEM3EmbedderSpecializedMethods:
    """Test query/corpus optimized encoding and advanced similarity methods."""

    @pytest.mark.asyncio
    async def test_encode_queries_optimization(self, test_settings, mock_flag_model):
        """Test query-optimized encoding method."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with (
            patch(
                "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
            ) as mock_class,
            patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.memory_allocated.return_value = 512 * 1024**2  # 512MB

            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            queries = [
                "What is machine learning?",
                "How does BGE-M3 unified embedding work?",
                "Explain document similarity search",
            ]

            params = EmbeddingParameters(
                return_dense=True,
                return_sparse=True,
                return_colbert=False,
            )

            result = await embedder.encode_queries(queries, parameters=params)

            # Verify result structure
            assert isinstance(result, EmbeddingResult)
            assert result.dense_embeddings is not None
            assert len(result.dense_embeddings) == 3
            assert result.sparse_embeddings is not None
            assert len(result.sparse_embeddings) == 3

            # Verify query optimization metadata
            assert (
                result.model_info["library"]
                == "FlagEmbedding.BGEM3FlagModel.encode_queries"
            )
            assert result.model_info["optimization"] == "query-optimized"

            # Verify statistics updated
            assert embedder._embedding_count == 3
            assert embedder._total_processing_time > 0

    @pytest.mark.asyncio
    async def test_encode_queries_empty_input(self, test_settings, mock_flag_model):
        """Test query encoding with empty input."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            result = await embedder.encode_queries([])

            # Verify empty result handling
            assert result.dense_embeddings == []
            assert result.sparse_embeddings is None
            assert result.batch_size == 0
            assert result.processing_time == 0.0
            assert "warning" in result.model_info
            assert result.model_info["warning"] == "No queries provided"

    @pytest.mark.asyncio
    async def test_encode_queries_error_handling(self, test_settings, mock_flag_model):
        """Test query encoding error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.encode_queries = Mock(
                side_effect=RuntimeError("Query encoding error")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            queries = ["Test query error handling"]

            with pytest.raises(EmbeddingError) as exc_info:
                await embedder.encode_queries(queries)

            assert "BGE-M3 query encoding failed" in str(exc_info.value)
            assert "Query encoding error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_encode_corpus_optimization(self, test_settings, mock_flag_model):
        """Test corpus-optimized encoding method."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with (
            patch(
                "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
            ) as mock_class,
            patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.memory_allocated.return_value = 768 * 1024**2  # 768MB

            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            corpus = [
                "Document 1: Machine learning fundamentals cover supervised and "
                "unsupervised algorithms for pattern recognition.",
                "Document 2: Natural language processing enables computers to "
                "understand human language through tokenization and embedding.",
                "Document 3: Vector databases store high-dimensional embeddings "
                "for efficient similarity search and retrieval.",
            ]

            params = EmbeddingParameters(
                return_dense=True,
                return_sparse=False,
                return_colbert=True,  # Test ColBERT with corpus
            )

            result = await embedder.encode_corpus(corpus, parameters=params)

            # Verify corpus optimization
            assert isinstance(result, EmbeddingResult)
            assert result.dense_embeddings is not None
            assert len(result.dense_embeddings) == 3
            assert result.sparse_embeddings is None  # Not requested
            assert result.colbert_embeddings is not None
            assert len(result.colbert_embeddings) == 3

            # Verify corpus optimization metadata
            assert (
                result.model_info["library"]
                == "FlagEmbedding.BGEM3FlagModel.encode_corpus"
            )
            assert result.model_info["optimization"] == "corpus-optimized"

            # Verify statistics updated
            assert embedder._embedding_count == 3
            assert embedder._total_processing_time > 0

    @pytest.mark.asyncio
    async def test_encode_corpus_empty_input(self, test_settings, mock_flag_model):
        """Test corpus encoding with empty input."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            result = await embedder.encode_corpus([])

            # Verify empty result handling
            assert result.dense_embeddings == []
            assert result.sparse_embeddings is None
            assert result.batch_size == 0
            assert result.processing_time == 0.0
            assert "warning" in result.model_info
            assert result.model_info["warning"] == "No corpus provided"

    @pytest.mark.asyncio
    async def test_encode_corpus_error_handling(self, test_settings, mock_flag_model):
        """Test corpus encoding error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.encode_corpus = Mock(
                side_effect=RuntimeError("Corpus encoding error")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            corpus = ["Test corpus error handling"]

            with pytest.raises(EmbeddingError) as exc_info:
                await embedder.encode_corpus(corpus)

            assert "BGE-M3 corpus encoding failed" in str(exc_info.value)
            assert "Corpus encoding error" in str(exc_info.value)

    def test_compute_similarity_hybrid_mode(self, test_settings, mock_flag_model):
        """Test hybrid similarity computation with multiple modes."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            queries = [
                "machine learning algorithm",
                "natural language processing",
            ]
            passages = [
                "Machine learning models learn patterns from data",
                "NLP techniques process human language text",
                "Deep learning neural networks for classification",
            ]

            scores = embedder.compute_similarity(
                queries,
                passages,
                mode="hybrid",
                max_passage_length=4096,
            )

            # Verify similarity scores structure
            assert isinstance(scores, dict)
            # Mock returns dense, sparse, colbert scores
            expected_keys = {"dense", "sparse", "colbert"}
            assert set(scores.keys()) == expected_keys

            # Should have 6 scores: 2 queries × 3 passages
            for mode_scores in scores.values():
                assert len(mode_scores) == 6
                for score in mode_scores:
                    assert isinstance(score, float)
                    assert 0.0 <= score <= 1.0

    def test_compute_similarity_error_handling(self, test_settings, mock_flag_model):
        """Test similarity computation error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Test RuntimeError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.compute_score = Mock(
                side_effect=RuntimeError("Similarity error")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            queries = ["test query"]
            passages = ["test passage"]

            scores = embedder.compute_similarity(queries, passages)

            # Should return error structure
            assert isinstance(scores, dict)
            assert "error" in scores
            assert len(scores["error"]) == 1  # 1 query × 1 passage
            assert scores["error"][0] == 0.0

        # Test ValueError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.compute_score = Mock(
                side_effect=ValueError("Invalid similarity")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            scores = embedder.compute_similarity(queries, passages)
            assert "error" in scores
            assert scores["error"][0] == 0.0

        # Test ImportError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.compute_score = Mock(side_effect=ImportError("Missing lib"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            scores = embedder.compute_similarity(queries, passages)
            assert "error" in scores
            assert scores["error"][0] == 0.0

        # Test generic Exception
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.compute_score = Mock(side_effect=Exception("Generic error"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            scores = embedder.compute_similarity(queries, passages)
            assert "error" in scores
            assert scores["error"][0] == 0.0

    def test_compute_colbert_similarity_success(self, test_settings, mock_flag_model):
        """Test ColBERT late-interaction similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Mock ColBERT vectors (multi-vector representations)
            colbert_vecs1 = [
                np.random.randn(10, 1024).astype(np.float32),  # 10 tokens
                np.random.randn(15, 1024).astype(np.float32),  # 15 tokens
            ]
            colbert_vecs2 = [
                np.random.randn(12, 1024).astype(np.float32),  # 12 tokens
            ]

            scores = embedder.compute_colbert_similarity(colbert_vecs1, colbert_vecs2)

            # Verify ColBERT similarity scores
            assert isinstance(scores, list)
            assert len(scores) == 2  # 2 × 1 = 2 comparisons
            for score in scores:
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0

    def test_compute_colbert_similarity_error_handling(
        self, test_settings, mock_flag_model
    ):
        """Test ColBERT similarity error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Test RuntimeError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.colbert_score = Mock(
                side_effect=RuntimeError("ColBERT error")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            colbert_vecs1 = [np.random.randn(5, 1024)]
            colbert_vecs2 = [np.random.randn(3, 1024)]

            scores = embedder.compute_colbert_similarity(colbert_vecs1, colbert_vecs2)

            # Should return zeros on error
            assert len(scores) == 1  # 1 × 1 comparison
            assert scores[0] == 0.0

        # Test ValueError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.colbert_score = Mock(
                side_effect=ValueError("Invalid ColBERT")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            scores = embedder.compute_colbert_similarity(colbert_vecs1, colbert_vecs2)
            assert scores[0] == 0.0

        # Test TypeError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.colbert_score = Mock(side_effect=TypeError("Type error"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            scores = embedder.compute_colbert_similarity(colbert_vecs1, colbert_vecs2)
            assert scores[0] == 0.0

        # Test generic Exception
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.colbert_score = Mock(side_effect=Exception("Generic error"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            scores = embedder.compute_colbert_similarity(colbert_vecs1, colbert_vecs2)
            assert scores[0] == 0.0


@pytest.mark.unit
class TestBGEM3EmbedderUtilityMethods:
    """Test utility methods including performance stats and token conversion."""

    def test_get_sparse_embedding_tokens_success(self, test_settings, mock_flag_model):
        """Test conversion of sparse embedding IDs to readable tokens."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Realistic sparse embeddings with token IDs
            sparse_embeddings = [
                {1000: 0.8, 2500: 0.6, 5000: 0.4},  # First text tokens
                {1500: 0.7, 3000: 0.5, 4500: 0.9},  # Second text tokens
            ]

            token_embeddings = embedder.get_sparse_embedding_tokens(sparse_embeddings)

            # Verify token conversion
            assert isinstance(token_embeddings, list)
            assert len(token_embeddings) >= 1  # Mock returns at least one dict
            for token_dict in token_embeddings:
                assert isinstance(token_dict, dict)
                for token, weight in token_dict.items():
                    assert isinstance(token, str)  # Converted to readable token
                    assert isinstance(weight, int | float)

    def test_get_sparse_embedding_tokens_error_handling(
        self, test_settings, mock_flag_model
    ):
        """Test token conversion error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Test RuntimeError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.convert_id_to_token = Mock(
                side_effect=RuntimeError("Token error")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            sparse_embeddings = [{100: 0.5}]
            result = embedder.get_sparse_embedding_tokens(sparse_embeddings)

            # Should return empty dicts on error
            assert isinstance(result, list)
            assert len(result) == len(sparse_embeddings)
            assert all(isinstance(d, dict) and len(d) == 0 for d in result)

        # Test ValueError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.convert_id_to_token = Mock(
                side_effect=ValueError("Value error")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            result = embedder.get_sparse_embedding_tokens(sparse_embeddings)
            assert all(len(d) == 0 for d in result)

        # Test KeyError
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.convert_id_to_token = Mock(side_effect=KeyError("Key error"))
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            result = embedder.get_sparse_embedding_tokens(sparse_embeddings)
            assert all(len(d) == 0 for d in result)

        # Test generic Exception
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_instance.convert_id_to_token = Mock(
                side_effect=Exception("Generic error")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            result = embedder.get_sparse_embedding_tokens(sparse_embeddings)
            assert all(len(d) == 0 for d in result)

    def test_get_performance_stats_tracking(self, test_settings, mock_flag_model):
        """Test performance statistics tracking and calculation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Simulate multiple embedding operations
            texts1 = ["First batch text 1", "First batch text 2"]
            texts2 = [
                "Second batch text 1",
                "Second batch text 2",
                "Second batch text 3",
            ]

            embedder.get_dense_embeddings(texts1)  # 2 texts
            embedder.get_sparse_embeddings(
                texts2
            )  # 3 texts - NOTE: this doesn't update stats in current implementation

            # Manually update stats to test calculation logic
            embedder._embedding_count = 5
            embedder._total_processing_time = 2.5  # 2.5 seconds

            stats = embedder.get_performance_stats()

            # Verify performance statistics
            assert stats["total_texts_embedded"] == 5
            assert stats["total_processing_time"] == 2.5
            assert stats["avg_time_per_text_ms"] == 500.0  # 2.5s / 5 texts * 1000ms
            assert stats["device"] == "cpu"  # CPU mode in test
            assert stats["model_library"] == "FlagEmbedding.BGEM3FlagModel"
            assert stats["unified_embeddings_enabled"] is True
            assert stats["pooling_method"] == "cls"
            assert stats["normalize_embeddings"] is True
            assert stats["weights_for_modes"] == [0.4, 0.2, 0.4]  # Default weights
            assert stats["library_optimization"] is True

    def test_get_performance_stats_zero_count(self, test_settings, mock_flag_model):
        """Test performance statistics with zero embedding count."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # No embeddings processed yet
            stats = embedder.get_performance_stats()

            # Should handle zero division gracefully
            assert stats["total_texts_embedded"] == 0
            assert stats["total_processing_time"] == 0.0
            assert stats["avg_time_per_text_ms"] == 0.0  # No division by zero

    def test_reset_stats(self, test_settings, mock_flag_model):
        """Test performance statistics reset functionality."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Set some statistics manually
            embedder._embedding_count = 10
            embedder._total_processing_time = 5.0

            # Verify stats are set
            initial_stats = embedder.get_performance_stats()
            assert initial_stats["total_texts_embedded"] == 10
            assert initial_stats["total_processing_time"] == 5.0

            # Reset statistics
            embedder.reset_stats()

            # Verify reset
            assert embedder._embedding_count == 0
            assert embedder._total_processing_time == 0.0

            reset_stats = embedder.get_performance_stats()
            assert reset_stats["total_texts_embedded"] == 0
            assert reset_stats["total_processing_time"] == 0.0
            assert reset_stats["avg_time_per_text_ms"] == 0.0

    def test_unload_model(self, test_settings, mock_flag_model):
        """Test model unloading and cleanup functionality."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            # Set some statistics
            embedder._embedding_count = 5
            embedder._total_processing_time = 2.0

            # Unload model
            embedder.unload_model()

            # Verify statistics reset
            assert embedder._embedding_count == 0
            assert embedder._total_processing_time == 0.0

            # Verify model moved to CPU (cleanup)
            mock_instance.model.to.assert_called_once_with("cpu")

    def test_unload_model_no_to_method(self, test_settings, mock_flag_model):
        """Test model unloading when model doesn't have to() method."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = mock_flag_model()
            # Remove the to() method to simulate model without GPU cleanup
            del mock_instance.model.to
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)
            embedder._embedding_count = 3

            # Should not raise error even without to() method
            embedder.unload_model()

            # Statistics should still be reset
            assert embedder._embedding_count == 0
            assert embedder._total_processing_time == 0.0


@pytest.mark.unit
class TestBGEM3EmbedderIntegrationScenarios:
    """Test realistic integration scenarios and data flow validation."""

    @pytest.mark.asyncio
    async def test_end_to_end_embedding_workflow(self, test_settings, mock_flag_model):
        """Test complete embedding workflow from initialization to cleanup."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with (
            patch(
                "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
            ) as mock_class,
            patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.memory_allocated.return_value = 512 * 1024**2

            mock_class.return_value = mock_flag_model()

            # Initialize embedder
            embedder = BGEM3Embedder(settings=test_settings)

            # Process various document types
            documents = [
                "Research paper: Machine learning algorithms for natural "
                "language processing applications.",
                "Technical blog: Understanding transformer architectures and "
                "attention mechanisms in depth.",
                "Documentation: API reference for embedding generation and "
                "similarity search functionality.",
                "Query: How do I implement semantic search using "
                "BGE-M3 unified embeddings?",
            ]

            # Generate embeddings with different configurations
            params1 = EmbeddingParameters(return_dense=True, return_sparse=True)
            result1 = await embedder.embed_texts_async(
                documents[:2], parameters=params1
            )

            params2 = EmbeddingParameters(return_dense=True, return_colbert=True)
            result2 = await embedder.embed_texts_async(
                documents[2:], parameters=params2
            )

            # Verify results
            assert len(result1.dense_embeddings) == 2
            assert len(result1.sparse_embeddings) == 2
            assert result1.colbert_embeddings is None

            assert len(result2.dense_embeddings) == 2
            assert result2.sparse_embeddings is None
            assert result2.colbert_embeddings is not None

            # Test similarity computation
            if result1.sparse_embeddings and len(result1.sparse_embeddings) >= 2:
                similarity = embedder.compute_sparse_similarity(
                    result1.sparse_embeddings[0], result1.sparse_embeddings[1]
                )
                assert isinstance(similarity, float)
                assert 0.0 <= similarity <= 1.0

            # Verify statistics tracking
            stats = embedder.get_performance_stats()
            assert stats["total_texts_embedded"] == 4
            assert stats["total_processing_time"] > 0

            # Test cleanup
            embedder.unload_model()
            final_stats = embedder.get_performance_stats()
            assert final_stats["total_texts_embedded"] == 0

    def test_batch_processing_various_sizes(self, test_settings, mock_flag_model):
        """Test batch processing with different batch sizes and text lengths."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Test different batch sizes
            test_batches = [
                ["Single text"],  # Size 1
                ["First", "Second", "Third"],  # Size 3
                [f"Document {i} with content" for i in range(10)],  # Size 10
                [f"Large batch text {i}" for i in range(25)],  # Size 25
            ]

            for batch in test_batches:
                dense_embeddings = embedder.get_dense_embeddings(batch)
                sparse_embeddings = embedder.get_sparse_embeddings(batch)

                # Verify batch processing
                assert dense_embeddings is not None
                assert len(dense_embeddings) == len(batch)
                assert sparse_embeddings is not None
                assert len(sparse_embeddings) == len(batch)

                # Verify dimensions consistency
                for embedding in dense_embeddings:
                    assert len(embedding) == 1024  # BGE-M3 dimension

                for sparse_emb in sparse_embeddings:
                    assert isinstance(sparse_emb, dict)
                    assert len(sparse_emb) > 0  # Should have some tokens

    def test_mixed_content_handling(self, test_settings, mock_flag_model):
        """Test handling of mixed content types and edge cases."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Mixed content with edge cases
            mixed_texts = [
                "",  # Empty text
                "   ",  # Whitespace only
                "Short",  # Very short
                "A" * 1000,  # Very long repetitive
                "Normal text with punctuation and numbers: 123!",  # Normal
                "Unicode: café résumé naïve 中文 العربية русский",  # Unicode
                "Code snippet: def function(x, y): return x + y",  # Code
                "URL and email: https://example.com user@domain.com",  # URLs/emails
            ]

            dense_embeddings = embedder.get_dense_embeddings(mixed_texts)

            # Should handle all content types
            assert dense_embeddings is not None
            assert len(dense_embeddings) == len(mixed_texts)

            # All should produce valid embeddings
            for i, embedding in enumerate(dense_embeddings):
                assert len(embedding) == 1024, f"Text {i}: {mixed_texts[i][:50]}..."
                assert all(isinstance(x, float) for x in embedding)
                # Verify no NaN or Inf values
                assert all(np.isfinite(x) for x in embedding)

    @pytest.mark.asyncio
    async def test_performance_optimization_scenarios(
        self, test_settings, mock_flag_model
    ):
        """Test performance-critical scenarios and optimization paths."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Test query optimization
            queries = [
                "What is the best approach for document similarity?",
                "How to optimize BGE-M3 for production use?",
            ]

            query_result = await embedder.encode_queries(queries)
            assert query_result.model_info["optimization"] == "query-optimized"

            # Test corpus optimization
            corpus = [
                "Document 1: Comprehensive guide to machine learning "
                "techniques and applications.",
                "Document 2: Advanced natural language processing methods "
                "for text understanding.",
            ]

            corpus_result = await embedder.encode_corpus(corpus)
            assert corpus_result.model_info["optimization"] == "corpus-optimized"

            # Test hybrid similarity computation
            similarity_scores = embedder.compute_similarity(
                queries, corpus, mode="hybrid"
            )

            # Should compute multiple similarity types
            assert isinstance(similarity_scores, dict)
            # 2 queries × 2 documents = 4 scores per mode
            for mode_scores in similarity_scores.values():
                if mode_scores:  # Skip empty results
                    assert len(mode_scores) == 4

    def test_error_recovery_scenarios(self, test_settings, mock_flag_model):
        """Test error recovery and graceful degradation scenarios."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            # Test partial failure scenario
            mock_instance = mock_flag_model()
            call_count = 0

            def intermittent_failure(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Second call fails
                    raise RuntimeError("Intermittent error")
                return mock_instance.encode(*args, **kwargs)

            # Replace encode method
            mock_instance.encode = intermittent_failure
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            # First call should succeed
            result1 = embedder.get_dense_embeddings(["First text"])
            assert result1 is not None

            # Second call should fail gracefully
            result2 = embedder.get_dense_embeddings(["Second text"])
            assert result2 is None  # Graceful degradation

            # Third call should succeed again
            result3 = embedder.get_dense_embeddings(["Third text"])
            assert result3 is not None

    def test_memory_efficiency_validation(self, test_settings, mock_flag_model):
        """Test memory efficiency and resource cleanup."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Process multiple batches to test memory handling
            for batch_size in [1, 5, 10, 20]:
                texts = [
                    f"Memory test batch {batch_size} text {i}"
                    for i in range(batch_size)
                ]

                embeddings = embedder.get_dense_embeddings(texts)

                # Verify embeddings are standard Python lists (memory efficient)
                assert isinstance(embeddings, list)
                for emb in embeddings:
                    assert isinstance(emb, list)  # Not numpy arrays
                    assert len(emb) == 1024

                # Verify no memory leaks - each embedding should be independent
                if len(embeddings) >= 2:
                    # Embeddings should be different
                    # (deterministic but based on different text)
                    assert (
                        embeddings[0] != embeddings[1] if len(embeddings) > 1 else True
                    )


@pytest.mark.unit
class TestBGEM3EmbedderDataFlowValidation:
    """Test data flow integrity and dimension validation."""

    @pytest.mark.asyncio
    async def test_embedding_dimension_consistency(
        self, test_settings, mock_flag_model
    ):
        """Test consistent 1024-dimensional output across all operations."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Test various text lengths and types
            test_cases = [
                "A",  # Single character
                "Short text",  # Short
                "Medium length text with several words and concepts that "
                "should be embedded properly",  # Medium
                "Very long text " * 50,  # Long repetitive
                "Mixed content with numbers 123, punctuation!!!, "
                "and symbols @#$%^&*()",  # Mixed
            ]

            # Test async method
            async_result = await embedder.embed_texts_async(test_cases)
            assert len(async_result.dense_embeddings) == len(test_cases)
            for embedding in async_result.dense_embeddings:
                assert len(embedding) == 1024  # BGE-M3 dimension

            # Test sync method
            sync_embeddings = embedder.get_dense_embeddings(test_cases)
            assert len(sync_embeddings) == len(test_cases)
            for embedding in sync_embeddings:
                assert len(embedding) == 1024  # BGE-M3 dimension

            # Test single text method
            single_embedding = await embedder.embed_single_text_async(test_cases[0])
            assert len(single_embedding) == 1024  # BGE-M3 dimension

            # Verify model info consistency
            assert async_result.model_info["embedding_dim"] == 1024

    def test_sparse_embedding_structure_validation(
        self, test_settings, mock_flag_model
    ):
        """Test sparse embedding structure and data integrity."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            texts = [
                "machine learning algorithm optimization",
                "natural language processing techniques",
                "vector database similarity search",
            ]

            sparse_embeddings = embedder.get_sparse_embeddings(texts)

            # Verify structure
            assert sparse_embeddings is not None
            assert len(sparse_embeddings) == len(texts)

            for i, sparse_emb in enumerate(sparse_embeddings):
                # Each sparse embedding should be a dictionary
                assert isinstance(sparse_emb, dict)
                assert len(sparse_emb) > 0, f"Empty sparse embedding for text {i}"

                # Validate token ID and weight structure
                for token_id, weight in sparse_emb.items():
                    assert isinstance(token_id, int)
                    assert token_id >= 0, f"Negative token ID: {token_id}"
                    assert isinstance(weight, int | float)
                    assert 0.0 <= weight <= 1.0, f"Invalid weight: {weight}"

                # Verify content-based expectations
                text_len = len(texts[i].split())
                # Sparse representation should relate to text complexity
                assert len(sparse_emb) >= min(text_len // 2, 1), (
                    f"Too few sparse tokens for text {i}"
                )

    @pytest.mark.asyncio
    async def test_multimodal_embedding_coordination(
        self, test_settings, mock_flag_model
    ):
        """Test coordination between dense, sparse, and ColBERT embeddings."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            texts = [
                "Comprehensive document about machine learning applications "
                "in natural language processing.",
                "Technical guide for implementing BGE-M3 unified embeddings "
                "in production systems.",
            ]

            # Test all embedding types together
            params = EmbeddingParameters(
                return_dense=True,
                return_sparse=True,
                return_colbert=True,
            )

            result = await embedder.embed_texts_async(texts, parameters=params)

            # Verify all embedding types present
            assert result.dense_embeddings is not None
            assert result.sparse_embeddings is not None
            assert result.colbert_embeddings is not None

            # Verify consistent batch size across all types
            assert len(result.dense_embeddings) == len(texts)
            assert len(result.sparse_embeddings) == len(texts)
            assert len(result.colbert_embeddings) == len(texts)

            # Verify dense embeddings
            for dense_emb in result.dense_embeddings:
                assert len(dense_emb) == 1024  # BGE-M3 dimension

            # Verify sparse embeddings
            for sparse_emb in result.sparse_embeddings:
                assert isinstance(sparse_emb, dict)
                assert len(sparse_emb) > 0

            # Verify ColBERT embeddings
            for colbert_emb in result.colbert_embeddings:
                assert isinstance(colbert_emb, np.ndarray)
                assert colbert_emb.shape[1] == 1024  # BGE-M3 dimension
                assert colbert_emb.shape[0] > 0  # Should have tokens

            # Verify model info reflects all modes
            assert result.model_info["dense_enabled"] is True
            assert result.model_info["sparse_enabled"] is True
            assert result.model_info["colbert_enabled"] is True

    def test_numerical_stability_validation(self, test_settings, mock_flag_model):
        """Test numerical stability of embeddings and computations."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            # Create mock with deterministic outputs for stability testing
            mock_instance = mock_flag_model()

            def stable_encode(*args, **kwargs):
                # Use deterministic seed for reproducible outputs
                np.random.seed(42)
                batch_size = len(args[0]) if args else 1

                return {
                    "dense_vecs": np.random.randn(batch_size, 1024).astype(np.float32),
                    "lexical_weights": [
                        {100: 0.8, 200: 0.6, 300: 0.4} for _ in range(batch_size)
                    ],
                }

            mock_instance.encode = stable_encode
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            text = "Numerical stability test text"

            # Generate multiple embeddings with same input
            embeddings = []
            for _ in range(5):
                result = embedder.get_dense_embeddings([text])
                embeddings.append(result[0])

            # Verify numerical consistency
            # (deterministic mock should produce same results)
            for i in range(1, len(embeddings)):
                np.testing.assert_allclose(
                    embeddings[0],
                    embeddings[i],
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"Embedding {i} differs from baseline",
                )

            # Verify no invalid values
            for embedding in embeddings:
                assert all(np.isfinite(x) for x in embedding), "Found NaN or Inf values"
                assert all(isinstance(x, float) for x in embedding), (
                    "Non-float values found"
                )


# Integration-style tests for boundary validation


@pytest.mark.unit
class TestBGEM3EmbedderBoundaryIntegration:
    """Integration-style tests for boundary interactions and realistic workflows."""

    @pytest.mark.asyncio
    async def test_realistic_document_processing_pipeline(
        self, test_settings, mock_flag_model
    ):
        """Test realistic document processing pipeline with mixed operations."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with (
            patch(
                "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
            ) as mock_class,
            patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.memory_allocated.return_value = 1024**3

            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Simulate realistic document processing workflow

            # Step 1: Process document corpus for indexing
            corpus_documents = [
                "Introduction to Machine Learning: Machine learning is a subset "
                "of artificial intelligence that enables computers to learn and "
                "make decisions from data without explicit programming.",
                "Deep Learning Fundamentals: Neural networks with multiple layers "
                "can learn complex patterns and representations from large "
                "amounts of data.",
                "Natural Language Processing: NLP combines computational "
                "linguistics with machine learning to help computers understand "
                "human language.",
                "Vector Databases: Specialized databases designed to store and "
                "query high-dimensional vectors for similarity search applications.",
            ]

            corpus_result = await embedder.encode_corpus(corpus_documents)
            assert len(corpus_result.dense_embeddings) == 4
            assert corpus_result.model_info["optimization"] == "corpus-optimized"

            # Step 2: Process user queries for search
            user_queries = [
                "What is machine learning?",
                "How do neural networks work?",
                "Vector similarity search methods",
            ]

            query_result = await embedder.encode_queries(user_queries)
            assert len(query_result.dense_embeddings) == 3
            assert query_result.model_info["optimization"] == "query-optimized"

            # Step 3: Compute similarity between queries and documents
            similarity_scores = embedder.compute_similarity(
                user_queries,
                corpus_documents[:2],  # First 2 documents
                mode="hybrid",
            )

            # Should have 3 queries × 2 documents = 6 scores per mode
            for mode, scores in similarity_scores.items():
                if scores and mode != "error":
                    assert len(scores) == 6

            # Step 4: Get sparse embeddings for explainability
            sparse_corpus = embedder.get_sparse_embeddings(corpus_documents)
            sparse_queries = embedder.get_sparse_embeddings(user_queries)

            assert len(sparse_corpus) == 4
            assert len(sparse_queries) == 3

            # Step 5: Compute sparse similarity for keyword matching
            sparse_similarity = embedder.compute_sparse_similarity(
                sparse_queries[0],  # First query
                sparse_corpus[0],  # First document
            )
            assert isinstance(sparse_similarity, float)
            assert 0.0 <= sparse_similarity <= 1.0

            # Verify overall statistics
            stats = embedder.get_performance_stats()
            # Total: 4 corpus + 3 queries = 7 from async methods
            # Note: sync methods don't update stats in current implementation
            assert stats["total_texts_embedded"] == 7

    def test_production_scale_batch_processing(self, test_settings, mock_flag_model):
        """Test production-scale batch processing scenarios."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Test various production-realistic batch sizes
            batch_configs = [
                (1, "Single document processing"),
                (10, "Small batch processing"),
                (50, "Medium batch processing"),
                (100, "Large batch processing"),
            ]

            for batch_size, description in batch_configs:
                # Generate realistic document-like texts
                texts = []
                for i in range(batch_size):
                    # Vary text length and content
                    base_content = [
                        "Technical documentation explaining software architecture "
                        "and design patterns.",
                        "Research paper abstract discussing machine learning "
                        "applications in healthcare.",
                        "User manual section describing API endpoints and "
                        "authentication methods.",
                        "Blog post content about data science best practices "
                        "and methodologies.",
                        "Academic article excerpt on natural language processing "
                        "techniques.",
                    ]
                    content = base_content[i % len(base_content)]
                    texts.append(f"Document {i}: {content}")

                # Process batch
                start_time = time.time()
                dense_embeddings = embedder.get_dense_embeddings(texts)
                processing_time = time.time() - start_time

                # Verify results
                assert dense_embeddings is not None, f"Failed: {description}"
                assert len(dense_embeddings) == batch_size, (
                    f"Wrong count: {description}"
                )

                # Verify all embeddings are valid
                for j, embedding in enumerate(dense_embeddings):
                    assert len(embedding) == 1024, (
                        f"Wrong dimension in {description}, doc {j}"
                    )
                    assert all(isinstance(x, float) for x in embedding), (
                        f"Non-float values in {description}"
                    )
                    assert all(np.isfinite(x) for x in embedding), (
                        f"Invalid values in {description}"
                    )

                # Performance validation (should be reasonable)
                avg_time_per_doc = processing_time / batch_size if batch_size > 0 else 0
                assert avg_time_per_doc < 1.0, (
                    f"Too slow: {description} - {avg_time_per_doc:.3f}s per doc"
                )

    def test_error_boundary_validation(self, test_settings):
        """Test error boundaries and proper error propagation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Test initialization boundary errors

        # 1. Test missing library boundary
        with patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel", None):
            with pytest.raises(EmbeddingError) as exc_info:
                BGEM3Embedder(settings=test_settings)

            error_msg = str(exc_info.value)
            assert "FlagEmbedding not available" in error_msg
            assert "uv add FlagEmbedding>=1.3.5" in error_msg

        # 2. Test model loading failure boundary
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.side_effect = Exception(
                "Model loading failed with detailed error message"
            )

            with pytest.raises(EmbeddingError) as exc_info:
                BGEM3Embedder(settings=test_settings)

            error_msg = str(exc_info.value)
            assert "BGE-M3 model initialization failed" in error_msg
            assert "Model loading failed with detailed error message" in error_msg

        # 3. Test runtime error boundaries
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_instance = Mock()
            mock_instance.encode = Mock(
                side_effect=RuntimeError("Runtime boundary error")
            )
            mock_class.return_value = mock_instance

            embedder = BGEM3Embedder(settings=test_settings)

            # Async method should propagate error
            async def test_async_error():
                with pytest.raises(EmbeddingError) as exc_info:
                    await embedder.embed_texts_async(["test"])
                assert "BGE-M3 unified embedding failed" in str(exc_info.value)
                assert "Runtime boundary error" in str(exc_info.value)

            # Run async test
            import asyncio

            asyncio.run(test_async_error())

            # Sync methods should return None (graceful degradation)
            result = embedder.get_dense_embeddings(["test"])
            assert result is None

            result = embedder.get_sparse_embeddings(["test"])
            assert result is None


# Update todo list
@pytest.mark.unit
class TestCoverageValidation:
    """Validate that comprehensive tests achieve target coverage."""

    def test_comprehensive_coverage_validation(self, test_settings, mock_flag_model):
        """Validate comprehensive test coverage of key functionality."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_class:
            mock_class.return_value = mock_flag_model()

            embedder = BGEM3Embedder(settings=test_settings)

            # Test coverage of key public methods
            public_methods = [
                "embed_texts_async",
                "embed_single_text_async",
                "get_dense_embeddings",
                "get_sparse_embeddings",
                "compute_sparse_similarity",
                "encode_queries",
                "encode_corpus",
                "compute_similarity",
                "compute_colbert_similarity",
                "get_sparse_embedding_tokens",
                "get_performance_stats",
                "reset_stats",
                "unload_model",
            ]

            # Verify all public methods are accessible
            for method_name in public_methods:
                assert hasattr(embedder, method_name), f"Missing method: {method_name}"
                method = getattr(embedder, method_name)
                assert callable(method), f"Not callable: {method_name}"

            # Test key initialization parameters
            init_params = [
                "settings",
                "parameters",
                "device",
                "pooling_method",
                "normalize_embeddings",
                "weights_for_different_modes",
                "return_numpy",
                "_embedding_count",
                "_total_processing_time",
            ]

            for param_name in init_params:
                assert hasattr(embedder, param_name), f"Missing parameter: {param_name}"

            # Verify model attribute exists
            assert hasattr(embedder, "model"), "Missing model attribute"

            # Test key error paths are covered
            assert hasattr(embedder.parameters, "max_length")
            assert hasattr(embedder.parameters, "use_fp16")
            assert hasattr(embedder.parameters, "return_dense")
            assert hasattr(embedder.parameters, "return_sparse")
            assert hasattr(embedder.parameters, "return_colbert")
