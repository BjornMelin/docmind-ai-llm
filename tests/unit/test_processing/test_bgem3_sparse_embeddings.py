"""Unit tests for BGE-M3 sparse embeddings functionality.

Tests the enhanced BGE-M3 implementation that generates both dense AND sparse
embeddings for hybrid search capability (ADR-002 FR-4).

New features tested:
- encode_queries() and encode_corpus() methods from FlagEmbedding
- Sparse embedding structure for Qdrant compatibility
- 8192 token context window
- Library-first FlagEmbedding integration
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.processing.embeddings.bgem3_embedder import BGEM3Embedder


@pytest.fixture
def mock_settings(tmp_path):
    """Mock settings for BGE-M3 embedder with proper temporary paths."""
    settings = Mock()
    settings.bge_m3_model_name = "BAAI/bge-m3"
    settings.embedding_dimension = 1024
    settings.max_context_length = 8192
    settings.enable_gpu_acceleration = True
    settings.pooling_method = "cls"
    settings.normalize_embeddings = True
    # CRITICAL: Provide real paths instead of mock objects to prevent directory creation
    settings.cache_dir = str(tmp_path / "cache")
    settings.data_dir = str(tmp_path / "data")
    settings.log_file = str(tmp_path / "logs" / "test.log")
    return settings


@pytest.fixture
def sample_texts():
    """Sample texts for embedding testing."""
    return [
        "DocMind AI processes documents with BGE-M3 unified embeddings.",
        "Sparse embeddings enable precise keyword matching for retrieval.",
        "Dense embeddings capture semantic relationships between concepts.",
    ]


@pytest.fixture
def sample_long_text():
    """Long text that approaches 8K context limit."""
    base_text = (
        "This is a comprehensive document about artificial intelligence and machine "
        "learning. BGE-M3 provides unified dense and sparse embeddings for enhanced "
        "retrieval capabilities. The model supports up to 8192 tokens for processing "
        "large document chunks effectively. "
    )
    # Repeat to create ~7K tokens
    return base_text * 100


class TestBGEM3SparseEmbeddings:
    """Test suite for BGE-M3 sparse embeddings functionality."""

    @pytest.mark.unit
    def test_bgem3_embedder_initialization_with_sparse_support(self, mock_settings):
        """Test BGEM3Embedder initializes with sparse embedding support.

        Should pass after implementation:
        - Initializes FlagEmbedding BGEM3FlagModel
        - Sets up both dense and sparse embedding capability
        - Configures 8K context window
        - Enables GPU acceleration if available
        """
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_model:
            embedder = BGEM3Embedder(mock_settings)

            assert embedder is not None
            assert hasattr(embedder, "model")
            assert hasattr(embedder, "parameters")
            assert embedder.parameters.max_length == 8192

            # Verify model initialization with correct parameters
            mock_model.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_encode_queries_method_integration(self, mock_settings, sample_texts):
        """Test encode_queries() method from FlagEmbedding library.

        Should pass after implementation:
        - Uses encode_queries() for query-specific optimization
        - Returns both dense and sparse embeddings
        - Handles batch processing efficiently
        - Maintains embedding quality for queries
        """
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model

            # Mock encode_queries response with numpy arrays
            mock_model.encode_queries.return_value = {
                "dense_vecs": np.array([[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]),
                "lexical_weights": [
                    {101: 0.5, 102: 0.3},  # Use integer token IDs
                    {103: 0.7, 104: 0.2},
                    {105: 0.9, 106: 0.1},
                ],
            }

            embedder = BGEM3Embedder(mock_settings)
            result = await embedder.encode_queries(sample_texts)

            # Verify method was called correctly
            mock_model.encode_queries.assert_called_once_with(
                sample_texts,
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

            # Verify result structure
            assert hasattr(result, "dense_embeddings")
            assert hasattr(result, "sparse_embeddings")
            assert len(result.dense_embeddings) == 3
            assert len(result.sparse_embeddings) == 3

            # Verify dense embeddings
            for dense_emb in result.dense_embeddings:
                assert len(dense_emb) == 1024
                assert all(isinstance(val, float) for val in dense_emb)

            # Verify sparse embeddings structure
            for sparse_emb in result.sparse_embeddings:
                assert isinstance(sparse_emb, dict)
                assert len(sparse_emb) > 0
                assert all(isinstance(weight, float) for weight in sparse_emb.values())

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_encode_corpus_method_integration(self, mock_settings, sample_texts):
        """Test encode_corpus() method from FlagEmbedding library.

        Should pass after implementation:
        - Uses encode_corpus() for document-specific optimization
        - Returns dense and sparse embeddings optimized for corpus
        - Handles large document batches efficiently
        - Maintains semantic quality for documents
        """
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model

            # Mock encode_corpus response
            mock_model.encode_corpus.return_value = {
                "dense_vecs": np.array([[0.4] * 1024, [0.5] * 1024, [0.6] * 1024]),
                "lexical_weights": [
                    {201: 0.8, 202: 0.6},  # Use integer token IDs
                    {203: 0.7, 204: 0.5},
                    {205: 0.9, 206: 0.4},
                ],
            }

            embedder = BGEM3Embedder(mock_settings)
            result = await embedder.encode_corpus(sample_texts)

            # Verify method was called correctly
            mock_model.encode_corpus.assert_called_once_with(
                sample_texts, batch_size=32, max_length=8192
            )

            # Verify result structure matches encode_queries
            assert hasattr(result, "dense_embeddings")
            assert hasattr(result, "sparse_embeddings")
            assert len(result.dense_embeddings) == 3
            assert len(result.sparse_embeddings) == 3

            # Verify different from query embeddings (corpus-optimized)
            assert result.dense_embeddings[0][0] == 0.4  # Different from query value

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sparse_embedding_qdrant_structure(self, mock_settings, sample_texts):
        """Test sparse embeddings are structured for Qdrant compatibility.

        Should pass after implementation:
        - Sparse embeddings formatted as {token_id: weight} dict
        - Token weights are normalized floats
        - Compatible with Qdrant sparse vector indexing
        - Maintains high-value tokens for retrieval effectiveness
        """
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model

            # Mock realistic sparse embedding structure
            mock_model.encode_corpus.return_value = {
                "dense_vecs": np.array([[0.1] * 1024]),
                "lexical_weights": [
                    {
                        301: 0.95,  # High-weight key terms (use integer token IDs)
                        302: 0.87,
                        303: 0.82,
                        304: 0.78,
                        305: 0.65,
                        306: 0.12,  # Lower-weight common terms
                        307: 0.08,
                        308: 0.05,
                    }
                ],
            }

            embedder = BGEM3Embedder(mock_settings)
            result = await embedder.encode_corpus(sample_texts[:1])

            sparse_emb = result.sparse_embeddings[0]

            # Verify Qdrant-compatible structure
            assert isinstance(sparse_emb, dict)
            assert len(sparse_emb) > 0

            # Verify weight normalization
            weights = list(sparse_emb.values())
            assert all(0.0 <= weight <= 1.0 for weight in weights)
            assert max(weights) > 0.5  # High-value terms present

            # Verify token diversity (not just high-frequency terms)
            assert len(sparse_emb) >= 5  # Reasonable token diversity
            assert "docmind" in sparse_emb  # Domain-specific terms captured
            assert sparse_emb["docmind"] > sparse_emb.get(
                "the", 0
            )  # Proper term weighting

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_8k_context_window_support(self, mock_settings, sample_long_text):
        """Test 8192 token context window processing.

        Should pass after implementation:
        - Processes up to 8K tokens without truncation
        - Maintains embedding quality for long contexts
        - Handles context length validation
        - Optimizes processing for large documents
        """
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model

            # Mock successful long text processing
            mock_model.encode_corpus.return_value = {
                "dense_vecs": np.array([[0.7] * 1024]),
                "lexical_weights": [
                    {401: 0.8, 402: 0.7, 403: 0.6}
                ],  # Use integer token IDs
            }

            embedder = BGEM3Embedder(mock_settings)
            result = await embedder.encode_corpus([sample_long_text])

            # Verify max_length parameter passed correctly
            call_args = mock_model.encode_corpus.call_args
            assert call_args[1]["max_length"] == 8192

            # Verify successful processing
            assert len(result.dense_embeddings) == 1
            assert len(result.sparse_embeddings) == 1
            assert len(result.dense_embeddings[0]) == 1024

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, mock_settings):
        """Test batch processing efficiency for multiple documents.

        Should pass after implementation:
        - Processes batches efficiently with configurable batch_size
        - Maintains consistent output structure across batches
        - Optimizes memory usage for large document sets
        - Provides progress tracking for long operations
        """
        large_batch = [
            f"Document {i} content for batch processing test." for i in range(100)
        ]

        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model

            # Mock batch processing response
            mock_model.encode_corpus.return_value = {
                "dense_vecs": np.array([[0.1] * 1024] * 100),
                "lexical_weights": [{501: 0.5, 502: 0.4}]
                * 100,  # Use integer token IDs
            }

            embedder = BGEM3Embedder(mock_settings)
            result = await embedder.encode_corpus(large_batch)

            # Verify batch processing
            assert len(result.dense_embeddings) == 100
            assert len(result.sparse_embeddings) == 100

            # Verify batch_size parameter used
            call_args = mock_model.encode_corpus.call_args
            assert call_args[1]["batch_size"] == 32  # Default batch size

    @pytest.mark.unit
    def test_library_first_flagembedding_integration(self, mock_settings):
        """Test library-first approach using FlagEmbedding directly.

        Should pass after implementation:
        - Uses FlagEmbedding BGEM3FlagModel directly (no custom wrappers)
        - Leverages built-in encode_queries/encode_corpus methods
        - Maintains compatibility with latest FlagEmbedding features
        - Follows library-first engineering principles
        """
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_model_class:
            embedder = BGEM3Embedder(mock_settings)

            # Verify direct FlagEmbedding usage
            mock_model_class.assert_called_once()

            # Verify no custom wrapper classes
            assert hasattr(embedder, "model")
            assert not hasattr(embedder, "custom_encoder")
            assert not hasattr(embedder, "wrapper_model")

            # Verify access to library methods
            mock_model = mock_model_class.return_value
            assert hasattr(mock_model, "encode_queries")
            assert hasattr(mock_model, "encode_corpus")


class TestBGEM3PerformanceOptimizations:
    """Test performance optimizations for BGE-M3 sparse embeddings."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_efficient_sparse_processing(
        self, mock_settings, sample_texts
    ):
        """Test memory-efficient processing of sparse embeddings.

        Should pass after implementation:
        - Processes sparse embeddings without excessive memory usage
        - Optimizes token weight storage
        - Handles garbage collection for large batches
        - Maintains performance with memory constraints
        """
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model

            # Mock memory-efficient processing
            mock_model.encode_corpus.return_value = {
                "dense_vecs": np.array([[0.1] * 1024] * 3),
                "lexical_weights": [
                    {601: 0.9, 602: 0.5},  # Use integer token IDs
                    {603: 0.8, 604: 0.6},
                    {605: 0.7, 606: 0.4},
                ],
            }

            embedder = BGEM3Embedder(mock_settings)
            result = await embedder.encode_corpus(sample_texts)

            # Verify sparse embeddings are memory-efficient
            for sparse_emb in result.sparse_embeddings:
                # Should contain only significant terms (not full vocabulary)
                assert len(sparse_emb) <= 10  # Reasonable sparsity
                assert all(
                    weight >= 0.3 for weight in sparse_emb.values()
                )  # High-value terms only

    @pytest.mark.unit
    def test_pooling_and_normalization_parameters(self, mock_settings):
        """Test pooling method and normalization configuration.

        Should pass after implementation:
        - Configures CLS pooling for optimal performance
        - Enables embedding normalization for similarity calculations
        - Uses library-provided pooling methods
        - Optimizes for retrieval effectiveness
        """
        with patch(
            "src.processing.embeddings.bgem3_embedder.BGEM3FlagModel"
        ) as mock_model_class:
            # Test CLS pooling configuration
            mock_settings.pooling_method = "cls"
            mock_settings.normalize_embeddings = True

            BGEM3Embedder(mock_settings)

            # Verify pooling and normalization parameters
            call_args = mock_model_class.call_args
            assert call_args[1]["pooling_method"] == "cls"
            assert call_args[1]["normalize_embeddings"] is True

            # Test alternative pooling methods
            mock_settings.pooling_method = "mean"
            BGEM3Embedder(mock_settings)

            call_args2 = mock_model_class.call_args
            assert call_args2[1]["pooling_method"] == "mean"
