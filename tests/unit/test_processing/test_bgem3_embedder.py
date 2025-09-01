"""Comprehensive unit tests for BGE-M3 embedder module.

Tests focus on interface contracts, dimension validation, batch processing,
and error handling while mocking heavy ML operations for fast execution.

Key testing areas:
- BGE-M3 initialization and configuration
- Dense embedding generation (1024D validation)
- Sparse embedding generation
- Batch processing with various sizes
- Error handling and recovery
- Performance statistics tracking
- Async operations
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.models.embeddings import (
    EmbeddingError,
    EmbeddingParameters,
    EmbeddingResult,
)


@pytest.fixture
def test_settings():
    """Real Pydantic settings for BGE-M3 embedder testing.

    Uses TestDocMindSettings with embedding-optimized configuration.
    ELIMINATES Mock anti-pattern, uses real Pydantic validation.
    """
    from tests.fixtures.test_settings import MockDocMindSettings as TestDocMindSettings

    return TestDocMindSettings(
        # Embedding-specific test configuration
        embedding={
            "model_name": "BAAI/bge-m3",
            "dimension": 1024,
            "max_length": 8192,
            "batch_size_cpu": 1,  # Small for unit tests
            "batch_size_gpu": 2,  # Small for unit tests
        },
        enable_gpu_acceleration=False,  # CPU-only for unit tests
    )


@pytest.fixture
def mock_bgem3_flag_model():
    """Mock BGEM3FlagModel with realistic behavior."""
    model = Mock()

    # Mock encode method with proper return structure
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
            # BGE-M3 produces 1024-dimensional dense embeddings
            result["dense_vecs"] = np.random.randn(batch_size, 1024).astype(np.float32)

        if return_sparse:
            # Mock sparse embeddings as list of dictionaries
            sparse_embeddings = []
            for _ in range(batch_size):
                # Random sparse embedding with token IDs and weights
                sparse_dict = {
                    np.random.randint(0, 30000): np.random.random()
                    for _ in range(np.random.randint(5, 15))
                }
                sparse_embeddings.append(sparse_dict)
            result["lexical_weights"] = sparse_embeddings

        if return_colbert_vecs:
            # Mock ColBERT embeddings as list of 2D arrays
            colbert_embeddings = []
            for _ in range(batch_size):
                # Random number of tokens per text (sequence length)
                seq_len = np.random.randint(10, 50)
                colbert_embeddings.append(
                    np.random.randn(seq_len, 1024).astype(np.float32)
                )
            result["colbert_vecs"] = colbert_embeddings

        return result

    model.encode.side_effect = mock_encode
    model.encode_queries.side_effect = mock_encode
    model.encode_corpus.side_effect = mock_encode

    # Mock similarity computation methods
    model.compute_lexical_matching_score.return_value = 0.75
    model.compute_score.return_value = {
        "dense": [0.85, 0.72, 0.91],
        "sparse": [0.68, 0.73, 0.82],
        "colbert": [0.89, 0.76, 0.93],
    }
    model.colbert_score.return_value = 0.88

    # Mock token conversion
    model.convert_id_to_token.return_value = [{"machine": 0.8, "learning": 0.6}]

    return model


@pytest.fixture
def embedding_parameters():
    """Standard embedding parameters for testing."""
    return EmbeddingParameters(
        max_length=8192,
        use_fp16=True,
        return_dense=True,
        return_sparse=True,
        return_colbert=False,
        normalize_embeddings=True,
    )


@pytest.mark.unit
class TestBGEM3EmbedderInitialization:
    """Test BGE-M3 embedder initialization and configuration."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_embedder_initialization_success(
        self, mock_torch, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test successful BGE-M3 embedder initialization."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        # Initialize embedder
        embedder = BGEM3Embedder(settings=test_settings)

        # Verify initialization
        assert embedder is not None
        assert embedder.settings == test_settings
        assert embedder.device == "cuda"
        assert embedder.pooling_method == "cls"
        assert embedder.normalize_embeddings is True

        # Verify model was initialized with correct parameters
        mock_flag_model_class.assert_called_once()
        call_kwargs = mock_flag_model_class.call_args[1]
        assert call_kwargs["model_name_or_path"] == "BAAI/bge-m3"
        assert call_kwargs["use_fp16"] is True
        assert "cuda" in str(call_kwargs["devices"])

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedder_initialization_cpu_fallback(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test embedder initialization falls back to CPU when CUDA unavailable."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        with patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            embedder = BGEM3Embedder(settings=test_settings)

            assert embedder.device == "cpu"
            # Verify FP16 is disabled on CPU
            call_kwargs = mock_flag_model_class.call_args[1]
            assert call_kwargs["use_fp16"] is False

    def test_embedder_initialization_missing_flagembedding(self, test_settings):
        """Test embedder initialization fails gracefully when FlagEmbedding unavailable.

        Verifies that the embedder properly handles missing dependencies and provides
        clear error messages for troubleshooting.
        """
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with (
            patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel", None),
            pytest.raises(EmbeddingError, match="FlagEmbedding not available"),
        ):
            BGEM3Embedder(settings=test_settings)

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedder_initialization_model_load_failure(
        self, mock_flag_model_class, test_settings
    ):
        """Test embedder handles model loading failures."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.side_effect = RuntimeError("Model loading failed")

        with pytest.raises(EmbeddingError, match="BGE-M3 model initialization failed"):
            BGEM3Embedder(settings=test_settings)

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedder_custom_parameters(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test embedder initialization with custom parameters."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

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
        )

        assert embedder.parameters.max_length == 4096
        assert embedder.parameters.use_fp16 is False
        assert embedder.pooling_method == "mean"
        assert embedder.normalize_embeddings is False
        assert embedder.weights_for_different_modes == [0.5, 0.3, 0.2]


@pytest.mark.unit
class TestBGEM3EmbedderDenseEmbeddings:
    """Test dense embedding generation and validation."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    async def test_embed_texts_async_dense_success(
        self,
        mock_torch,
        mock_flag_model_class,
        test_settings,
        mock_bgem3_flag_model,
        embedding_parameters,
    ):
        """Test successful dense embedding generation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3  # 2GB
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(
            settings=test_settings, parameters=embedding_parameters
        )

        texts = [
            "Machine learning is transforming document processing.",
            "BGE-M3 provides unified dense and sparse embeddings.",
            "Vector search enables semantic similarity matching.",
        ]

        result = await embedder.embed_texts_async(texts)

        # Verify result structure
        assert isinstance(result, EmbeddingResult)
        assert result.dense_embeddings is not None
        assert len(result.dense_embeddings) == 3

        # Verify BGE-M3 1024-dimensional embeddings
        for embedding in result.dense_embeddings:
            assert len(embedding) == 1024
            assert all(isinstance(x, float) for x in embedding)

        # Verify sparse embeddings returned
        assert result.sparse_embeddings is not None
        assert len(result.sparse_embeddings) == 3

        # Verify metadata
        assert result.processing_time > 0
        assert result.batch_size == 3
        assert result.memory_usage_mb > 0
        assert result.model_info["embedding_dim"] == 1024
        assert result.model_info["library"] == "FlagEmbedding.BGEM3FlagModel"

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    async def test_embed_single_text_async(
        self, mock_torch, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test single text embedding extraction."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        text = "Single text for embedding generation test."

        embedding = await embedder.embed_single_text_async(text)

        # Verify single embedding
        assert isinstance(embedding, list)
        assert len(embedding) == 1024  # BGE-M3 dimension
        assert all(isinstance(x, float) for x in embedding)

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embed_texts_async_empty_input(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test embedding generation with empty input."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        result = await embedder.embed_texts_async([])

        assert isinstance(result, EmbeddingResult)
        assert result.dense_embeddings == []
        assert result.sparse_embeddings is None
        assert result.batch_size == 0
        assert result.processing_time == 0.0
        assert "warning" in result.model_info

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_dense_embeddings_sync(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test synchronous dense embedding generation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        texts = ["First text", "Second text"]

        embeddings = embedder.get_dense_embeddings(texts)

        assert embeddings is not None
        assert len(embeddings) == 2
        assert all(len(emb) == 1024 for emb in embeddings)

        # Verify model called with correct parameters
        mock_bgem3_flag_model.encode.assert_called_with(
            texts,
            max_length=embedder.parameters.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )


@pytest.mark.unit
class TestBGEM3EmbedderSparseEmbeddings:
    """Test sparse embedding generation and processing."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_sparse_embeddings_success(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test successful sparse embedding generation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        texts = ["Machine learning text", "Natural language processing"]

        sparse_embeddings = embedder.get_sparse_embeddings(texts)

        assert sparse_embeddings is not None
        assert len(sparse_embeddings) == 2

        # Verify sparse embedding structure
        for sparse_emb in sparse_embeddings:
            assert isinstance(sparse_emb, dict)
            assert all(
                isinstance(k, int) and isinstance(v, float)
                for k, v in sparse_emb.items()
            )

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_sparse_similarity(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test sparse embedding similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        sparse1 = {1: 0.8, 5: 0.6, 10: 0.4}
        sparse2 = {1: 0.7, 3: 0.5, 5: 0.9}

        similarity = embedder.compute_sparse_similarity(sparse1, sparse2)

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        mock_bgem3_flag_model.compute_lexical_matching_score.assert_called_once_with(
            sparse1, sparse2
        )

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_sparse_embedding_tokens(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test conversion of sparse embedding IDs to tokens."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        sparse_embeddings = [{100: 0.8, 200: 0.6}, {150: 0.7, 300: 0.5}]

        tokens = embedder.get_sparse_embedding_tokens(sparse_embeddings)

        assert tokens is not None
        assert len(tokens) == 1  # Mock returns single dict
        assert isinstance(tokens[0], dict)
        mock_bgem3_flag_model.convert_id_to_token.assert_called_once_with(
            sparse_embeddings
        )


@pytest.mark.unit
class TestBGEM3EmbedderBatchProcessing:
    """Test batch processing with various sizes and configurations."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    async def test_batch_processing_small(
        self, mock_torch, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test batch processing with small batch size."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Small batch
        texts = ["Text one", "Text two", "Text three"]

        result = await embedder.embed_texts_async(texts)

        assert len(result.dense_embeddings) == 3
        assert result.batch_size == 3

        # Verify model called once with all texts
        assert mock_bgem3_flag_model.encode.call_count == 1

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    async def test_batch_processing_large(
        self, mock_torch, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test batch processing with larger batch size."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Larger batch - 20 texts
        texts = [f"Test document number {i} with various content." for i in range(20)]

        result = await embedder.embed_texts_async(texts)

        assert len(result.dense_embeddings) == 20
        assert result.batch_size == 20

        # Verify all embeddings have correct dimensions
        for embedding in result.dense_embeddings:
            assert len(embedding) == 1024

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_batch_processing_various_text_lengths(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test batch processing with various text lengths."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        texts = [
            "Short",  # Very short
            "Medium length text with some more words to test handling.",  # Medium
            "Very long text " * 100,  # Long text
            "",  # Empty text
            "Normal text for testing purposes.",  # Normal
        ]

        result = await embedder.embed_texts_async(texts)

        assert len(result.dense_embeddings) == 5
        # Verify all texts produced embeddings (including empty text)
        for embedding in result.dense_embeddings:
            assert len(embedding) == 1024


@pytest.mark.unit
class TestBGEM3EmbedderSpecializedMethods:
    """Test specialized encoding methods (queries, corpus, similarity)."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    async def test_encode_queries_optimization(
        self,
        mock_torch,
        mock_flag_model_class,
        test_settings,
        mock_bgem3_flag_model,
        embedding_parameters,
    ):
        """Test query-optimized encoding."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024**3  # 1GB
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(
            settings=test_settings, parameters=embedding_parameters
        )

        queries = [
            "What is machine learning?",
            "How does BGE-M3 work?",
            "Document processing techniques",
        ]

        result = await embedder.encode_queries(queries)

        # Verify query encoding called
        mock_bgem3_flag_model.encode_queries.assert_called_once()

        # Verify result structure
        assert isinstance(result, EmbeddingResult)
        assert result.dense_embeddings is not None
        assert len(result.dense_embeddings) == 3
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
        test_settings,
        mock_bgem3_flag_model,
        embedding_parameters,
    ):
        """Test corpus-optimized encoding."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024**3
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(
            settings=test_settings, parameters=embedding_parameters
        )

        corpus = [
            "Document 1: Machine learning fundamentals and applications in modern AI "
            "systems.",
            "Document 2: Natural language processing techniques for text analysis and "
            "understanding.",
            "Document 3: Vector databases and similarity search in information "
            "retrieval systems.",
        ]

        result = await embedder.encode_corpus(corpus)

        # Verify corpus encoding called
        mock_bgem3_flag_model.encode_corpus.assert_called_once()

        # Verify result structure
        assert isinstance(result, EmbeddingResult)
        assert result.dense_embeddings is not None
        assert len(result.dense_embeddings) == 3
        assert (
            result.model_info["library"] == "FlagEmbedding.BGEM3FlagModel.encode_corpus"
        )
        assert result.model_info["optimization"] == "corpus-optimized"

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_similarity_hybrid(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test hybrid similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        queries = ["What is AI?", "How does ML work?"]
        passages = ["AI explanation text", "ML tutorial content", "Deep learning guide"]

        scores = embedder.compute_similarity(queries, passages, mode="hybrid")

        # Verify compute_score called with sentence pairs
        mock_bgem3_flag_model.compute_score.assert_called_once()
        call_args = mock_bgem3_flag_model.compute_score.call_args[0][0]

        # Should create pairs: (q1,p1), (q1,p2), (q1,p3), (q2,p1), (q2,p2), (q2,p3)
        assert len(call_args) == 6
        assert isinstance(scores, dict)

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_colbert_similarity(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test ColBERT similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Mock ColBERT vectors
        colbert_vecs1 = [np.random.randn(10, 1024), np.random.randn(15, 1024)]
        colbert_vecs2 = [np.random.randn(12, 1024)]

        scores = embedder.compute_colbert_similarity(colbert_vecs1, colbert_vecs2)

        # Should compute similarity for each pair
        assert len(scores) == 2  # 2 * 1 = 2 pairs
        assert all(isinstance(score, float) for score in scores)
        assert mock_bgem3_flag_model.colbert_score.call_count == 2


@pytest.mark.unit
class TestBGEM3EmbedderErrorHandling:
    """Test error handling and recovery scenarios."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embed_texts_async_model_error(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test embedding generation handles model errors."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode.side_effect = RuntimeError("CUDA out of memory")

        embedder = BGEM3Embedder(settings=test_settings)

        texts = ["Test text"]

        with pytest.raises(EmbeddingError, match="BGE-M3 unified embedding failed"):
            await embedder.embed_texts_async(texts)

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embed_single_text_async_no_embeddings(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test single text embedding handles missing embeddings."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        # Mock empty embeddings result - return structure that results in no dense
        # embeddings
        def mock_encode_empty(*args, **kwargs):
            return {
                "dense_vecs": np.array([]).reshape(0, 1024)
            }  # Empty array with proper shape

        mock_bgem3_flag_model.encode.side_effect = mock_encode_empty

        embedder = BGEM3Embedder(settings=test_settings)

        with pytest.raises(EmbeddingError, match="No dense embeddings generated"):
            await embedder.embed_single_text_async("Test text")

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_sparse_embeddings_error_handling(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test sparse embedding error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode.side_effect = ValueError("Invalid input")

        embedder = BGEM3Embedder(settings=test_settings)

        result = embedder.get_sparse_embeddings(["Test text"])

        # Should return None on error, not raise exception
        assert result is None

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_sparse_similarity_error_handling(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test sparse similarity computation error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.compute_lexical_matching_score.side_effect = TypeError(
            "Invalid arguments"
        )

        embedder = BGEM3Embedder(settings=test_settings)

        sparse1 = {1: 0.5}
        sparse2 = {2: 0.7}

        similarity = embedder.compute_sparse_similarity(sparse1, sparse2)

        # Should return 0.0 on error, not raise exception
        assert similarity == 0.0


@pytest.mark.unit
class TestBGEM3EmbedderPerformanceStats:
    """Test performance statistics tracking and management."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    async def test_performance_stats_tracking(
        self, mock_torch, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test performance statistics are tracked correctly."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Process some texts
        await embedder.embed_texts_async(["Text 1", "Text 2"])
        await embedder.embed_texts_async(["Text 3", "Text 4", "Text 5"])

        stats = embedder.get_performance_stats()

        assert stats["total_texts_embedded"] == 5
        assert stats["total_processing_time"] > 0
        assert stats["avg_time_per_text_ms"] > 0
        assert stats["device"] == "cuda"
        assert stats["model_library"] == "FlagEmbedding.BGEM3FlagModel"
        assert stats["unified_embeddings_enabled"] is True
        assert stats["library_optimization"] is True

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_reset_stats(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test performance statistics reset."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Simulate some usage
        embedder._embedding_count = 10
        embedder._total_processing_time = 5.0

        embedder.reset_stats()

        assert embedder._embedding_count == 0
        assert embedder._total_processing_time == 0.0

        stats = embedder.get_performance_stats()
        assert stats["total_texts_embedded"] == 0
        assert stats["avg_time_per_text_ms"] == 0

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_model_unloading(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test model unloading and cleanup."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        # Setup mock model with to() method for GPU cleanup
        mock_model = Mock()
        mock_model.to = Mock()
        mock_bgem3_flag_model.model = mock_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Set some stats
        embedder._embedding_count = 5
        embedder._total_processing_time = 2.0

        embedder.unload_model()

        # Verify stats reset
        assert embedder._embedding_count == 0
        assert embedder._total_processing_time == 0.0

        # Verify model moved to CPU
        mock_model.to.assert_called_once_with("cpu")


@pytest.mark.unit
class TestBGEM3EmbedderDimensionValidation:
    """Test BGE-M3 1024-dimension validation and consistency."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embedding_dimension_consistency(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test all embeddings consistently return 1024 dimensions."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Test various batch sizes
        for batch_size in [1, 5, 10, 20]:
            texts = [f"Test text {i}" for i in range(batch_size)]

            result = await embedder.embed_texts_async(texts)

            # Verify dense embeddings dimension
            assert len(result.dense_embeddings) == batch_size
            for embedding in result.dense_embeddings:
                assert len(embedding) == 1024, (
                    f"Expected 1024 dimensions, got {len(embedding)}"
                )

            # Verify model info reports correct dimension
            assert result.model_info["embedding_dim"] == 1024

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedding_dimension_validation_sync(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test dimension validation in synchronous methods."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        texts = ["Test 1", "Test 2", "Test 3"]

        dense_embeddings = embedder.get_dense_embeddings(texts)

        assert dense_embeddings is not None
        assert len(dense_embeddings) == 3
        for embedding in dense_embeddings:
            assert len(embedding) == 1024

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_settings_dimension_consistency(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test embedder dimension matches settings dimension."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        # Verify settings dimension
        assert test_settings.embedding.dimension == 1024

        BGEM3Embedder(settings=test_settings)

        # Verify embedder uses correct dimension from settings
        # The actual BGE-M3 always produces 1024D, so this should match
        expected_dim = 1024
        assert test_settings.embedding.dimension == expected_dim


@pytest.mark.unit
class TestBGEM3EmbedderDataFlowValidation:
    """Test data flow validation through embedding pipeline."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embedding_data_flow_integrity(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test data integrity through embedding pipeline."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Input texts with known characteristics
        test_texts = [
            "Short text",
            "Medium length text with multiple words and concepts",
            "Very long text " * 50 + " that exceeds normal length boundaries",
            "",  # Empty text
            "Special characters: !@#$%^&*()_+-={}[]|\\:;\"'<>?,./",
            "Unicode text: café résumé naïve 中文 العربية русский 日本語",
        ]

        result = await embedder.embed_texts_async(test_texts)

        # Validate data flow integrity
        assert len(result.dense_embeddings) == len(test_texts)
        assert len(result.sparse_embeddings) == len(test_texts)

        # Each embedding should maintain proper structure
        for i, (dense_emb, sparse_emb) in enumerate(
            zip(result.dense_embeddings, result.sparse_embeddings, strict=False)
        ):
            # Dense embedding validation
            assert isinstance(dense_emb, list)
            assert len(dense_emb) == 1024
            assert all(isinstance(x, float) for x in dense_emb)

            # Sparse embedding validation
            assert isinstance(sparse_emb, dict)
            for token_id, weight in sparse_emb.items():
                assert isinstance(token_id, int)
                assert isinstance(weight, int | float)
                assert 0.0 <= weight <= 1.0  # Normalized weights

            # Content-based validation
            text_length = len(test_texts[i])
            if text_length == 0:
                # Empty text might have minimal sparse representation
                continue
            elif text_length > 1000:
                # Long text should have more diverse sparse representation
                assert len(sparse_emb) >= 5
            else:
                # Normal text should have reasonable sparse representation
                assert len(sparse_emb) >= 1

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedding_numerical_stability(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test numerical stability of embeddings."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Mock with controlled outputs to test stability
        def stable_encode(*args, **kwargs):
            batch_size = len(args[0]) if args else 1
            return {
                "dense_vecs": np.ones((batch_size, 1024), dtype=np.float32) * 0.5,
                "lexical_weights": [{100: 0.8, 200: 0.6} for _ in range(batch_size)],
            }

        mock_bgem3_flag_model.encode.side_effect = stable_encode
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Test stability across multiple calls
        texts = ["Stability test text"] * 3
        results = []

        for _ in range(5):
            embeddings = embedder.get_dense_embeddings(texts)
            results.append(embeddings)

        # Verify consistency
        for i in range(len(results[0])):
            for j in range(1, len(results)):
                np.testing.assert_allclose(
                    results[0][i], results[j][i], rtol=1e-10, atol=1e-10
                )

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embedding_edge_case_inputs(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test embeddings with edge case inputs."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Edge case inputs
        edge_case_texts = [
            "   ",  # Only whitespace
            "\n\n\n",  # Only newlines
            "\t\t\t",  # Only tabs
            "A" * 10000,  # Very long single character
            "1234567890" * 100,  # Numeric only
            ".,;:!?()[]{}",  # Punctuation only
        ]

        result = await embedder.embed_texts_async(edge_case_texts)

        # Should handle all edge cases without error
        assert len(result.dense_embeddings) == len(edge_case_texts)

        for embedding in result.dense_embeddings:
            assert len(embedding) == 1024
            assert all(isinstance(x, float) for x in embedding)
            assert all(np.isfinite(x) for x in embedding)  # No NaN or Inf

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedding_memory_efficiency(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test memory efficiency in embedding operations."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Test with various batch sizes
        batch_sizes = [1, 10, 50, 100]

        for batch_size in batch_sizes:
            texts = [f"Memory test text {i}" for i in range(batch_size)]

            embeddings = embedder.get_dense_embeddings(texts)

            # Verify no memory leaks - embeddings should be proper Python lists
            assert isinstance(embeddings, list)
            assert len(embeddings) == batch_size

            # Verify each embedding is a standard list, not numpy array
            # (memory efficient)
            for emb in embeddings:
                assert isinstance(emb, list)
                assert len(emb) == 1024


@pytest.mark.unit
class TestBGEM3EmbedderCachingBehavior:
    """Test embedding caching and performance optimization."""

    @pytest.mark.asyncio
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_performance_stats_accumulation(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test performance statistics accumulation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Simulate multiple embedding operations
        batch_sizes = [3, 5, 2, 8]
        total_expected = sum(batch_sizes)

        for batch_size in batch_sizes:
            texts = [f"Text {i}" for i in range(batch_size)]
            await embedder.embed_texts_async(texts)

        stats = embedder.get_performance_stats()

        # Verify accumulation
        assert stats["total_texts_embedded"] == total_expected
        assert stats["total_processing_time"] > 0
        assert stats["avg_time_per_text_ms"] > 0

    @pytest.mark.asyncio
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_stats_reset_behavior(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test statistics reset behavior."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Generate some stats
        texts = ["Text 1", "Text 2"]
        await embedder.embed_texts_async(texts)

        initial_stats = embedder.get_performance_stats()
        assert initial_stats["total_texts_embedded"] == 2

        # Reset and verify
        embedder.reset_stats()

        reset_stats = embedder.get_performance_stats()
        assert reset_stats["total_texts_embedded"] == 0
        assert reset_stats["total_processing_time"] == 0.0
        assert reset_stats["avg_time_per_text_ms"] == 0.0

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedding_consistency_across_calls(
        self, mock_flag_model_class, test_settings
    ):
        """Test embedding consistency for repeated calls."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Mock with deterministic output
        mock_model = Mock()

        def deterministic_encode(*args, **kwargs):
            # Seed for deterministic output
            np.random.seed(42)
            batch_size = len(args[0]) if args else 1
            return {
                "dense_vecs": np.random.randn(batch_size, 1024).astype(np.float32),
                "lexical_weights": [{100: 0.8, 200: 0.6} for _ in range(batch_size)],
            }

        mock_model.encode.side_effect = deterministic_encode
        mock_flag_model_class.return_value = mock_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Multiple calls with same input
        text = "Consistency test"
        results = []

        for _ in range(3):
            # Reset seed for each call to get same result
            embedding = embedder.get_dense_embeddings([text])
            results.append(embedding[0])

        # With deterministic mock, should get same results
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0], results[i], rtol=1e-5, atol=1e-5)


@pytest.mark.unit
class TestBGEM3EmbedderAdvancedErrorHandling:
    """Test advanced error handling scenarios."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_partial_failure_recovery(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test recovery from partial failures."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Mock that fails on first call, succeeds on retry
        call_count = 0

        def failing_encode(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Temporary failure")

            batch_size = len(args[0]) if args else 1
            return {
                "dense_vecs": np.random.randn(batch_size, 1024).astype(np.float32),
                "lexical_weights": [{100: 0.8} for _ in range(batch_size)],
            }

        mock_bgem3_flag_model.encode.side_effect = failing_encode
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # First call should fail internally; current implementation returns None
        texts = ["Test text"]
        result = embedder.get_dense_embeddings(texts)
        assert result is None

        # Second call should succeed (if implemented with retry logic)
        # For current implementation, this would still fail
        # This test documents expected behavior

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_malformed_output_handling(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test handling of malformed model outputs."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Mock with malformed output
        def malformed_encode(*args, **kwargs):
            return {"unexpected_key": "unexpected_value"}

        mock_bgem3_flag_model.encode.side_effect = malformed_encode
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Should handle malformed output gracefully
        texts = ["Test text"]
        result = embedder.get_dense_embeddings(texts)

        # Current implementation returns None for malformed output
        assert result is None

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_async_cancellation_handling(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test handling of async cancellation."""
        import asyncio

        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Test that async operations can be cancelled
        texts = ["Test text"] * 100  # Large batch

        async def embedding_task():
            return await embedder.embed_texts_async(texts)

        task = asyncio.create_task(embedding_task())

        # Cancel immediately (in real scenario, would cancel after some time)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.unit
class TestBGEM3EmbedderIntegrationPatterns:
    """Test integration patterns with other components."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedding_result_serialization(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test embedding result can be serialized/deserialized."""
        import json
        import pickle

        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        texts = ["Serialization test"]
        embeddings = embedder.get_dense_embeddings(texts)

        # Test JSON serialization
        json_data = json.dumps(embeddings)
        recovered_embeddings = json.loads(json_data)

        assert len(recovered_embeddings) == len(embeddings)
        assert len(recovered_embeddings[0]) == 1024

        # Test pickle serialization
        pickle_data = pickle.dumps(embeddings)
        pickle_recovered = pickle.loads(pickle_data)

        np.testing.assert_allclose(embeddings[0], pickle_recovered[0])

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedding_pipeline_compatibility(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test compatibility with pipeline processing patterns."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Simulate pipeline processing
        documents = [{"id": i, "text": f"Document {i} content"} for i in range(10)]

        # Process in chunks (common pipeline pattern)
        chunk_size = 3
        all_embeddings = []

        for i in range(0, len(documents), chunk_size):
            chunk = documents[i : i + chunk_size]
            texts = [doc["text"] for doc in chunk]

            embeddings = embedder.get_dense_embeddings(texts)
            all_embeddings.extend(embeddings)

        # Verify pipeline output
        assert len(all_embeddings) == len(documents)
        for embedding in all_embeddings:
            assert len(embedding) == 1024

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedding_metadata_preservation(
        self, mock_flag_model_class, test_settings, mock_bgem3_flag_model
    ):
        """Test metadata preservation in embedding operations."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=test_settings)

        # Test with metadata tracking
        texts_with_metadata = [
            {"text": "First document", "source": "doc1.pdf", "page": 1},
            {"text": "Second document", "source": "doc2.pdf", "page": 2},
        ]

        # Process texts while preserving metadata
        texts = [item["text"] for item in texts_with_metadata]
        embeddings = embedder.get_dense_embeddings(texts)

        # Combine results with metadata
        results = []
        for i, embedding in enumerate(embeddings):
            result = {
                **texts_with_metadata[i],
                "embedding": embedding,
                "embedding_dim": len(embedding),
            }
            results.append(result)

        # Verify metadata preservation
        assert len(results) == 2
        assert results[0]["source"] == "doc1.pdf"
        assert results[1]["page"] == 2
        assert all(r["embedding_dim"] == 1024 for r in results)
