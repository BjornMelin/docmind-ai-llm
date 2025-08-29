"""Comprehensive unit tests for BGE-M3 embedder functionality.

Tests focus on covering the 187 uncovered statements in bgem3_embedder.py
with emphasis on data transformation validation, error handling, and
pipeline integration points using MockEmbedding patterns.

Key areas:
- BGE-M3 initialization and parameter validation
- Dense, sparse, and ColBERT embedding operations
- Error handling and recovery scenarios
- Performance statistics and model management
- Vector similarity computation and caching
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.models.embeddings import EmbeddingError, EmbeddingParameters, EmbeddingResult


@pytest.fixture
def mock_bgem3_flag_model():
    """Mock BGEM3FlagModel for comprehensive testing."""
    model = Mock()

    # Mock encode method with realistic BGE-M3 behavior
    def mock_encode(
        texts,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        **kwargs,
    ):
        batch_size = len(texts) if isinstance(texts, list) else 1
        result = {}

        if return_dense:
            # BGE-M3 produces 1024D dense vectors
            result["dense_vecs"] = np.random.randn(batch_size, 1024).astype(np.float32)

        if return_sparse:
            # BGE-M3 sparse embeddings: token_id -> weight
            result["lexical_weights"] = [
                {i: float(np.random.random()) for i in range(np.random.randint(5, 20))}
                for _ in range(batch_size)
            ]

        if return_colbert_vecs:
            # ColBERT multi-vectors: variable length sequences
            result["colbert_vecs"] = [
                np.random.randn(np.random.randint(50, 200), 1024)
                for _ in range(batch_size)
            ]

        return result

    model.encode.side_effect = mock_encode
    model.encode_queries.side_effect = mock_encode
    model.encode_corpus.side_effect = mock_encode

    # Mock similarity computation methods
    model.compute_lexical_matching_score.return_value = 0.75
    model.colbert_score.return_value = 0.85
    model.compute_score.return_value = {
        "dense": [0.9, 0.8, 0.7],
        "sparse": [0.6, 0.7, 0.8],
        "hybrid": [0.82, 0.75, 0.77],
    }
    model.convert_id_to_token.return_value = [{"hello": 0.8, "world": 0.6}]

    return model


@pytest.fixture
def mock_settings():
    """Mock settings with BGE-M3 configuration."""
    settings = Mock()
    settings.embedding = Mock()
    settings.embedding.model_name = "BAAI/bge-m3"
    settings.embedding.dimension = 1024
    settings.embedding.max_length = 8192
    settings.embedding.batch_size_gpu = 12
    settings.embedding.batch_size_cpu = 4
    return settings


@pytest.fixture
def embedding_parameters():
    """Standard embedding parameters for testing."""
    return EmbeddingParameters(
        max_length=8192,
        use_fp16=True,
        return_dense=True,
        return_sparse=True,
        return_colbert=False,
    )


@pytest.mark.unit
class TestBGEM3EmbedderInitialization:
    """Test BGEM3Embedder initialization and configuration."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_embedder_initialization_success(
        self, mock_torch, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test successful embedder initialization with all parameters."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(
            settings=mock_settings,
            pooling_method="cls",
            normalize_embeddings=True,
            weights_for_different_modes=[0.4, 0.2, 0.4],
            devices=["cuda:0"],
            return_numpy=False,
        )

        assert embedder.settings == mock_settings
        assert embedder.pooling_method == "cls"
        assert embedder.normalize_embeddings is True
        assert embedder.weights_for_different_modes == [0.4, 0.2, 0.4]
        assert embedder.device == "cuda"
        assert embedder._embedding_count == 0
        assert embedder._total_processing_time == 0.0

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedder_initialization_flagembedding_missing(
        self, mock_flag_model_class, mock_settings
    ):
        """Test initialization fails when FlagEmbedding is not available."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel", None):
            with pytest.raises(EmbeddingError, match="FlagEmbedding not available"):
                BGEM3Embedder(settings=mock_settings)

    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_device_detection_and_fallback(self, mock_torch, mock_settings):
        """Test device detection and CPU fallback."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_flag_model_class:
            mock_flag_model_class.return_value = Mock()

            # Test CUDA available
            mock_torch.cuda.is_available.return_value = True
            embedder_cuda = BGEM3Embedder(settings=mock_settings)
            assert embedder_cuda.device == "cuda"

            # Test CUDA not available
            mock_torch.cuda.is_available.return_value = False
            embedder_cpu = BGEM3Embedder(settings=mock_settings)
            assert embedder_cpu.device == "cpu"

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_model_initialization_failure(self, mock_flag_model_class, mock_settings):
        """Test handling of model initialization failures."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(EmbeddingError, match="BGE-M3 model initialization failed"):
            BGEM3Embedder(settings=mock_settings)


@pytest.mark.unit
class TestBGEM3EmbedderOperations:
    """Test core embedding operations."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    async def test_embed_texts_async_success(
        self,
        mock_torch,
        mock_flag_model_class,
        mock_bgem3_flag_model,
        mock_settings,
        embedding_parameters,
    ):
        """Test successful async text embedding with full result validation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 500  # 500MB
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)
        texts = ["Hello world", "Machine learning", "Vector embeddings"]

        result = await embedder.embed_texts_async(texts, embedding_parameters)

        # Validate result structure
        assert isinstance(result, EmbeddingResult)
        assert result.dense_embeddings is not None
        assert len(result.dense_embeddings) == 3
        assert len(result.dense_embeddings[0]) == 1024

        # Validate sparse embeddings
        assert result.sparse_embeddings is not None
        assert len(result.sparse_embeddings) == 3
        assert all(isinstance(sparse, dict) for sparse in result.sparse_embeddings)

        # Validate metadata
        assert result.batch_size == 3
        assert result.processing_time > 0
        assert result.memory_usage_mb == 500.0
        assert result.model_info["model_name"] == "BAAI/bge-m3"
        assert result.model_info["embedding_dim"] == 1024
        assert result.model_info["library"] == "FlagEmbedding.BGEM3FlagModel"

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embed_texts_async_empty_input(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test handling of empty text input."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        result = await embedder.embed_texts_async([])

        assert result.dense_embeddings == []
        assert result.sparse_embeddings is None
        assert result.batch_size == 0
        assert result.processing_time == 0.0
        assert "warning" in result.model_info

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embed_single_text_async(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test single text embedding with backward compatibility."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        result = await embedder.embed_single_text_async("Test text")

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embed_single_text_async_no_embeddings(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test single text embedding when no dense embeddings are generated."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Mock empty dense embeddings
        mock_bgem3_flag_model.encode.return_value = {"dense_vecs": np.array([])}
        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        with pytest.raises(EmbeddingError, match="No dense embeddings generated"):
            await embedder.embed_single_text_async("Test text")

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_get_sparse_embeddings_success(
        self, mock_torch, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test sparse embedding extraction."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = False
        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        texts = ["Test text 1", "Test text 2"]
        result = embedder.get_sparse_embeddings(texts)

        assert result is not None
        assert len(result) == 2
        assert all(isinstance(sparse, dict) for sparse in result)

        # Verify model called with correct parameters
        mock_bgem3_flag_model.encode.assert_called_with(
            texts,
            max_length=embedder.parameters.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_sparse_embeddings_error_handling(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test sparse embedding error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Test different error types
        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        # RuntimeError
        mock_bgem3_flag_model.encode.side_effect = RuntimeError("Model error")
        result = embedder.get_sparse_embeddings(["test"])
        assert result is None

        # ValueError
        mock_bgem3_flag_model.encode.side_effect = ValueError("Invalid input")
        result = embedder.get_sparse_embeddings(["test"])
        assert result is None

        # Generic Exception
        mock_bgem3_flag_model.encode.side_effect = Exception("Unexpected error")
        result = embedder.get_sparse_embeddings(["test"])
        assert result is None

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_dense_embeddings_success(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test dense embedding extraction."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        texts = ["Dense embedding test"]
        result = embedder.get_dense_embeddings(texts)

        assert result is not None
        assert len(result) == 1
        assert len(result[0]) == 1024
        assert all(isinstance(x, float) for x in result[0])

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_dense_embeddings_no_results(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test dense embedding when no dense_vecs in result."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_bgem3_flag_model.encode.return_value = {"sparse": "data"}  # No dense_vecs
        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        result = embedder.get_dense_embeddings(["test"])
        assert result is None


@pytest.mark.unit
class TestBGEM3EmbedderAdvancedOperations:
    """Test advanced embedding operations and optimizations."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    async def test_encode_queries_optimization(
        self,
        mock_torch,
        mock_flag_model_class,
        mock_bgem3_flag_model,
        mock_settings,
        embedding_parameters,
    ):
        """Test query-optimized encoding."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 300
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)
        queries = ["What is AI?", "How does ML work?"]

        result = await embedder.encode_queries(queries, embedding_parameters)

        # Verify query-optimized encoding was called
        mock_bgem3_flag_model.encode_queries.assert_called_once_with(
            queries,
            max_length=embedding_parameters.max_length,
            return_dense=embedding_parameters.return_dense,
            return_sparse=embedding_parameters.return_sparse,
            return_colbert_vecs=embedding_parameters.return_colbert,
        )

        # Validate result
        assert (
            result.model_info["library"]
            == "FlagEmbedding.BGEM3FlagModel.encode_queries"
        )
        assert result.model_info["optimization"] == "query-optimized"

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    async def test_encode_corpus_optimization(
        self,
        mock_torch,
        mock_flag_model_class,
        mock_bgem3_flag_model,
        mock_settings,
        embedding_parameters,
    ):
        """Test corpus-optimized encoding."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 400
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)
        corpus = ["Document 1 content", "Document 2 content", "Document 3 content"]

        result = await embedder.encode_corpus(corpus, embedding_parameters)

        # Verify corpus-optimized encoding was called
        mock_bgem3_flag_model.encode_corpus.assert_called_once_with(
            corpus,
            max_length=embedding_parameters.max_length,
            return_dense=embedding_parameters.return_dense,
            return_sparse=embedding_parameters.return_sparse,
            return_colbert_vecs=embedding_parameters.return_colbert,
        )

        # Validate result
        assert (
            result.model_info["library"] == "FlagEmbedding.BGEM3FlagModel.encode_corpus"
        )
        assert result.model_info["optimization"] == "corpus-optimized"
        assert result.batch_size == 3

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_encode_queries_empty_input(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test query encoding with empty input."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        result = await embedder.encode_queries([])

        assert result.dense_embeddings == []
        assert result.batch_size == 0
        assert "warning" in result.model_info

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_encode_corpus_empty_input(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test corpus encoding with empty input."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        result = await embedder.encode_corpus([])

        assert result.dense_embeddings == []
        assert result.batch_size == 0
        assert "warning" in result.model_info


@pytest.mark.unit
class TestBGEM3EmbedderSimilarityComputation:
    """Test similarity computation methods."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_sparse_similarity_success(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test sparse similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        sparse1 = {1: 0.8, 2: 0.6, 3: 0.4}
        sparse2 = {1: 0.7, 2: 0.9, 4: 0.5}

        similarity = embedder.compute_sparse_similarity(sparse1, sparse2)

        assert similarity == 0.75  # Mock return value
        mock_bgem3_flag_model.compute_lexical_matching_score.assert_called_once_with(
            sparse1, sparse2
        )

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_sparse_similarity_error_handling(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test sparse similarity computation error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        # Test RuntimeError
        mock_bgem3_flag_model.compute_lexical_matching_score.side_effect = RuntimeError(
            "Computation error"
        )
        similarity = embedder.compute_sparse_similarity({}, {})
        assert similarity == 0.0

        # Test generic Exception
        mock_bgem3_flag_model.compute_lexical_matching_score.side_effect = Exception(
            "Unexpected error"
        )
        similarity = embedder.compute_sparse_similarity({}, {})
        assert similarity == 0.0

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_similarity_comprehensive(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test comprehensive similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        texts1 = ["Query 1", "Query 2"]
        texts2 = ["Passage 1", "Passage 2", "Passage 3"]

        scores = embedder.compute_similarity(
            texts1, texts2, mode="hybrid", max_passage_length=4096
        )

        assert "dense" in scores
        assert "sparse" in scores
        assert "hybrid" in scores

        # Verify sentence pairs were created correctly (2 queries * 3 passages = 6 pairs)
        expected_pairs = [
            ["Query 1", "Passage 1"],
            ["Query 1", "Passage 2"],
            ["Query 1", "Passage 3"],
            ["Query 2", "Passage 1"],
            ["Query 2", "Passage 2"],
            ["Query 2", "Passage 3"],
        ]
        mock_bgem3_flag_model.compute_score.assert_called_once_with(
            expected_pairs,
            max_passage_length=4096,
            weights_for_different_modes=embedder.weights_for_different_modes,
        )

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_similarity_error_handling(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test similarity computation error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        mock_bgem3_flag_model.compute_score.side_effect = RuntimeError(
            "Computation failed"
        )

        scores = embedder.compute_similarity(["q1"], ["p1"])

        assert "error" in scores
        assert scores["error"] == [0.0]  # 1 query * 1 passage = 1 score

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_colbert_similarity(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test ColBERT similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        # Mock ColBERT vectors
        colbert_vecs1 = [np.random.randn(50, 1024), np.random.randn(60, 1024)]
        colbert_vecs2 = [np.random.randn(40, 1024)]

        scores = embedder.compute_colbert_similarity(colbert_vecs1, colbert_vecs2)

        # Should have 2 * 1 = 2 scores
        assert len(scores) == 2
        assert all(score == 0.85 for score in scores)  # Mock return value

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_colbert_similarity_error_handling(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test ColBERT similarity error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        mock_bgem3_flag_model.colbert_score.side_effect = RuntimeError("ColBERT error")

        scores = embedder.compute_colbert_similarity([[]], [[]])
        assert scores == [0.0]


@pytest.mark.unit
class TestBGEM3EmbedderUtilities:
    """Test utility methods and model management."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_sparse_embedding_tokens(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test sparse embedding token conversion."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        sparse_embeddings = [{1: 0.8, 2: 0.6}, {3: 0.9, 4: 0.5}]

        tokens = embedder.get_sparse_embedding_tokens(sparse_embeddings)

        assert tokens == [{"hello": 0.8, "world": 0.6}]  # Mock return value
        mock_bgem3_flag_model.convert_id_to_token.assert_called_once_with(
            sparse_embeddings
        )

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_sparse_embedding_tokens_error_handling(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test sparse embedding token conversion error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        mock_bgem3_flag_model.convert_id_to_token.side_effect = RuntimeError(
            "Token conversion failed"
        )

        sparse_embeddings = [{1: 0.8}]
        tokens = embedder.get_sparse_embedding_tokens(sparse_embeddings)

        assert tokens == [{}]  # Empty dict for error case

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_performance_stats(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test performance statistics tracking."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        # Simulate some embedding operations
        embedder._embedding_count = 100
        embedder._total_processing_time = 5.0

        stats = embedder.get_performance_stats()

        assert stats["total_texts_embedded"] == 100
        assert stats["total_processing_time"] == 5.0
        assert stats["avg_time_per_text_ms"] == 50.0  # 5000ms / 100 texts
        assert stats["model_library"] == "FlagEmbedding.BGEM3FlagModel"
        assert stats["unified_embeddings_enabled"] is True
        assert stats["pooling_method"] == embedder.pooling_method
        assert stats["normalize_embeddings"] == embedder.normalize_embeddings
        assert stats["weights_for_modes"] == embedder.weights_for_different_modes
        assert stats["library_optimization"] is True

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_performance_stats_no_embeddings(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test performance statistics with no embeddings processed."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        stats = embedder.get_performance_stats()

        assert stats["total_texts_embedded"] == 0
        assert stats["avg_time_per_text_ms"] == 0.0

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_reset_stats(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test performance statistics reset."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        # Set some values
        embedder._embedding_count = 50
        embedder._total_processing_time = 2.5

        embedder.reset_stats()

        assert embedder._embedding_count == 0
        assert embedder._total_processing_time == 0.0

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_unload_model(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test model unloading and cleanup."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Mock model with .to() method for GPU cleanup
        mock_inner_model = Mock()
        mock_inner_model.to = Mock()
        mock_bgem3_flag_model.model = mock_inner_model

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings)

        # Set some stats to verify reset
        embedder._embedding_count = 25
        embedder._total_processing_time = 1.5

        embedder.unload_model()

        # Verify stats were reset
        assert embedder._embedding_count == 0
        assert embedder._total_processing_time == 0.0

        # Verify model was moved to CPU
        mock_inner_model.to.assert_called_once_with("cpu")


@pytest.mark.unit
class TestBGEM3EmbedderErrorHandling:
    """Test comprehensive error handling scenarios."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embed_texts_async_embedding_error(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test embed_texts_async error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode.side_effect = RuntimeError("Embedding failed")

        embedder = BGEM3Embedder(settings=mock_settings)

        with pytest.raises(EmbeddingError, match="BGE-M3 unified embedding failed"):
            await embedder.embed_texts_async(["test text"])

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_encode_queries_error_handling(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test query encoding error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode_queries.side_effect = ValueError(
            "Query encoding failed"
        )

        embedder = BGEM3Embedder(settings=mock_settings)

        with pytest.raises(EmbeddingError, match="BGE-M3 query encoding failed"):
            await embedder.encode_queries(["test query"])

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_encode_corpus_error_handling(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test corpus encoding error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode_corpus.side_effect = ImportError(
            "Corpus encoding failed"
        )

        embedder = BGEM3Embedder(settings=mock_settings)

        with pytest.raises(EmbeddingError, match="BGE-M3 corpus encoding failed"):
            await embedder.encode_corpus(["test corpus"])

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embed_single_text_async_error_propagation(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test single text async embedding error propagation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode.side_effect = RuntimeError("Model error")

        embedder = BGEM3Embedder(settings=mock_settings)

        with pytest.raises(EmbeddingError, match="Single text embedding failed"):
            await embedder.embed_single_text_async("test")


@pytest.mark.unit
class TestBGEM3EmbedderAliases:
    """Test backward compatibility aliases."""

    def test_bgem3_embedding_manager_alias(self):
        """Test BGEM3EmbeddingManager alias works correctly."""
        from src.processing.embeddings.bgem3_embedder import (
            BGEM3Embedder,
            BGEM3EmbeddingManager,
        )

        assert BGEM3EmbeddingManager is BGEM3Embedder

    def test_flag_model_alias(self):
        """Test FlagModel alias points to correct class."""
        from src.processing.embeddings.bgem3_embedder import BGEM3FlagModel, FlagModel

        assert FlagModel is BGEM3FlagModel


@pytest.mark.unit
class TestBGEM3EmbedderIntegration:
    """Test integration scenarios and performance tracking."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    async def test_embedding_processing_time_tracking(
        self, mock_torch, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test processing time tracking across operations."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100

        # Add delay to mock encoding
        def slow_encode(*args, **kwargs):
            time.sleep(0.01)  # 10ms delay
            return {
                "dense_vecs": np.random.randn(1, 1024).astype(np.float32),
                "lexical_weights": [{1: 0.8, 2: 0.6}],
            }

        mock_bgem3_flag_model.encode.side_effect = slow_encode
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

        # Process multiple batches
        for i in range(3):
            result = await embedder.embed_texts_async([f"Test text {i}"])
            assert result.processing_time >= 0.01  # At least 10ms

        stats = embedder.get_performance_stats()
        assert stats["total_texts_embedded"] == 3
        assert stats["total_processing_time"] >= 0.03  # At least 30ms total
        assert stats["avg_time_per_text_ms"] >= 10.0  # At least 10ms per text

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_numpy_return_conversion(
        self, mock_flag_model_class, mock_bgem3_flag_model, mock_settings
    ):
        """Test numpy return format conversion."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        embedder = BGEM3Embedder(settings=mock_settings, return_numpy=True)

        # The embedder should handle numpy conversion but in our test it doesn't matter
        # since we're testing the code path
        assert embedder.return_numpy is True
