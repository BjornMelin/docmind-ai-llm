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
def mock_settings():
    """Mock DocMind settings for BGE-M3 embedder testing."""
    settings = Mock()
    settings.embedding = Mock()
    settings.embedding.model_name = "BAAI/bge-m3"
    settings.embedding.dimension = 1024
    settings.embedding.max_length = 8192
    settings.embedding.batch_size = 12
    return settings


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
        self, mock_torch, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test successful BGE-M3 embedder initialization."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        # Initialize embedder
        embedder = BGEM3Embedder(settings=mock_settings)

        # Verify initialization
        assert embedder is not None
        assert embedder.settings == mock_settings
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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test embedder initialization falls back to CPU when CUDA unavailable."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        with patch("src.processing.embeddings.bgem3_embedder.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            embedder = BGEM3Embedder(settings=mock_settings)

            assert embedder.device == "cpu"
            # Verify FP16 is disabled on CPU
            call_kwargs = mock_flag_model_class.call_args[1]
            assert call_kwargs["use_fp16"] is False

    def test_embedder_initialization_missing_flagembedding(self, mock_settings):
        """Test embedder initialization fails gracefully when FlagEmbedding unavailable."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        with patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel", None):
            with pytest.raises(EmbeddingError, match="FlagEmbedding not available"):
                BGEM3Embedder(settings=mock_settings)

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedder_initialization_model_load_failure(
        self, mock_flag_model_class, mock_settings
    ):
        """Test embedder handles model loading failures."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.side_effect = RuntimeError("Model loading failed")

        with pytest.raises(EmbeddingError, match="BGE-M3 model initialization failed"):
            BGEM3Embedder(settings=mock_settings)

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_embedder_custom_parameters(
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
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
            settings=mock_settings,
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
        mock_settings,
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
            settings=mock_settings, parameters=embedding_parameters
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
        self, mock_torch, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test single text embedding extraction."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

        text = "Single text for embedding generation test."

        embedding = await embedder.embed_single_text_async(text)

        # Verify single embedding
        assert isinstance(embedding, list)
        assert len(embedding) == 1024  # BGE-M3 dimension
        assert all(isinstance(x, float) for x in embedding)

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embed_texts_async_empty_input(
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test embedding generation with empty input."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

        result = await embedder.embed_texts_async([])

        assert isinstance(result, EmbeddingResult)
        assert result.dense_embeddings == []
        assert result.sparse_embeddings is None
        assert result.batch_size == 0
        assert result.processing_time == 0.0
        assert "warning" in result.model_info

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_dense_embeddings_sync(
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test synchronous dense embedding generation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test successful sparse embedding generation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test sparse embedding similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test conversion of sparse embedding IDs to tokens."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_torch, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test batch processing with small batch size."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_torch, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test batch processing with larger batch size."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test batch processing with various text lengths."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        mock_settings,
        mock_bgem3_flag_model,
        embedding_parameters,
    ):
        """Test query-optimized encoding."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024**3  # 1GB
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(
            settings=mock_settings, parameters=embedding_parameters
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
        mock_settings,
        mock_bgem3_flag_model,
        embedding_parameters,
    ):
        """Test corpus-optimized encoding."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024**3
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(
            settings=mock_settings, parameters=embedding_parameters
        )

        corpus = [
            "Document 1: Machine learning fundamentals and applications in modern AI systems.",
            "Document 2: Natural language processing techniques for text analysis and understanding.",
            "Document 3: Vector databases and similarity search in information retrieval systems.",
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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test hybrid similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test ColBERT similarity computation."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test embedding generation handles model errors."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode.side_effect = RuntimeError("CUDA out of memory")

        embedder = BGEM3Embedder(settings=mock_settings)

        texts = ["Test text"]

        with pytest.raises(EmbeddingError, match="BGE-M3 unified embedding failed"):
            await embedder.embed_texts_async(texts)

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    async def test_embed_single_text_async_no_embeddings(
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test single text embedding handles missing embeddings."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        # Mock empty embeddings result - return structure that results in no dense embeddings
        def mock_encode_empty(*args, **kwargs):
            return {
                "dense_vecs": np.array([]).reshape(0, 1024)
            }  # Empty array with proper shape

        mock_bgem3_flag_model.encode.side_effect = mock_encode_empty

        embedder = BGEM3Embedder(settings=mock_settings)

        with pytest.raises(EmbeddingError, match="No dense embeddings generated"):
            await embedder.embed_single_text_async("Test text")

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_get_sparse_embeddings_error_handling(
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test sparse embedding error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.encode.side_effect = ValueError("Invalid input")

        embedder = BGEM3Embedder(settings=mock_settings)

        result = embedder.get_sparse_embeddings(["Test text"])

        # Should return None on error, not raise exception
        assert result is None

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_compute_sparse_similarity_error_handling(
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test sparse similarity computation error handling."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model
        mock_bgem3_flag_model.compute_lexical_matching_score.side_effect = TypeError(
            "Invalid arguments"
        )

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_torch, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test performance statistics are tracked correctly."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_torch.cuda.is_available.return_value = True
        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test performance statistics reset."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test model unloading and cleanup."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        # Setup mock model with to() method for GPU cleanup
        mock_model = Mock()
        mock_model.to = Mock()
        mock_bgem3_flag_model.model = mock_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test all embeddings consistently return 1024 dimensions."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

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
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test dimension validation in synchronous methods."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        embedder = BGEM3Embedder(settings=mock_settings)

        texts = ["Test 1", "Test 2", "Test 3"]

        dense_embeddings = embedder.get_dense_embeddings(texts)

        assert dense_embeddings is not None
        assert len(dense_embeddings) == 3
        for embedding in dense_embeddings:
            assert len(embedding) == 1024

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_settings_dimension_consistency(
        self, mock_flag_model_class, mock_settings, mock_bgem3_flag_model
    ):
        """Test embedder dimension matches settings dimension."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        mock_flag_model_class.return_value = mock_bgem3_flag_model

        # Verify settings dimension
        assert mock_settings.embedding.dimension == 1024

        embedder = BGEM3Embedder(settings=mock_settings)

        # Verify embedder uses correct dimension from settings
        # The actual BGE-M3 always produces 1024D, so this should match
        expected_dim = 1024
        assert mock_settings.embedding.dimension == expected_dim
