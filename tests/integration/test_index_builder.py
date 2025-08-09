"""Comprehensive tests for index builder functionality.

This module tests index creation with GPU optimization, knowledge graph generation,
hybrid retrieval with RRF fusion, multimodal processing, and async operations
following 2025 best practices.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import ImageDocument

from models import AppSettings
from utils.index_builder import (
    create_hybrid_retriever,
    create_index,
    create_index_async,
    create_multimodal_index,
    create_multimodal_index_async,
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return AppSettings(
        dense_embedding_model="BAAI/bge-large-en-v1.5",
        sparse_embedding_model="prithivida/Splade_PP_en_v1",
        dense_embedding_dimension=1024,
        gpu_acceleration=True,
        embedding_batch_size=32,
        rrf_fusion_weight_dense=0.7,
        rrf_fusion_weight_sparse=0.3,
        rrf_fusion_alpha=60,
        prefetch_factor=2,
        similarity_top_k=5,
        max_entities=50,
        debug_mode=False,
        default_model="google/gemma-3n-E4B-it",
        qdrant_url="http://localhost:6333",
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            text="SPLADE++ sparse embeddings enable efficient hybrid retrieval.",
            metadata={"source": "doc1.pdf", "page": 1, "type": "text"},
        ),
        Document(
            text="BGE-Large provides rich semantic understanding for dense search.",
            metadata={"source": "doc2.pdf", "page": 2, "type": "text"},
        ),
        Document(
            text="RRF fusion combines dense and sparse results optimally.",
            metadata={"source": "doc3.pdf", "page": 1, "type": "text"},
        ),
    ]


@pytest.fixture
def sample_image_documents():
    """Create sample image documents for multimodal testing."""
    return [
        ImageDocument(
            text="Image description: Technical diagram",
            image_path="/fake/path/image1.jpg",
            metadata={"source": "doc1.pdf", "page": 1, "type": "image"},
        ),
        ImageDocument(
            text="Image description: Chart with data",
            image_path="/fake/path/image2.png",
            metadata={"source": "doc2.pdf", "page": 2, "type": "image"},
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    mock_store = MagicMock()
    mock_store.client = MagicMock()
    return mock_store


@pytest.fixture
def mock_vector_index():
    """Create a mock vector store index for testing."""
    mock_index = MagicMock(spec=VectorStoreIndex)
    mock_index.as_retriever.return_value = MagicMock()
    return mock_index


class TestHybridRetriever:
    """Test hybrid retriever creation and configuration."""

    def test_create_hybrid_retriever_success(self, mock_vector_index, mock_settings):
        """Test successful hybrid retriever creation with RRF fusion."""
        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.VectorIndexRetriever") as mock_retriever,
            patch("utils.index_builder.QueryFusionRetriever") as mock_fusion
        ):
                    mock_dense_retriever = MagicMock()
                    mock_sparse_retriever = MagicMock()
                    mock_fusion_retriever = MagicMock()

                    mock_retriever.side_effect = [
                        mock_dense_retriever,
                        mock_sparse_retriever,
                    ]
                    mock_fusion.return_value = mock_fusion_retriever

                    result = create_hybrid_retriever(mock_vector_index)

                    # Verify dense retriever creation
                    dense_call = mock_retriever.call_args_list[0]
                    assert dense_call[1]["index"] == mock_vector_index
                    assert (
                        dense_call[1]["similarity_top_k"] == 10
                    )  # prefetch_factor * similarity_top_k
                    assert dense_call[1]["vector_store_query_mode"] == "default"

                    # Verify sparse retriever creation
                    sparse_call = mock_retriever.call_args_list[1]
                    assert sparse_call[1]["index"] == mock_vector_index
                    assert sparse_call[1]["similarity_top_k"] == 10
                    assert sparse_call[1]["vector_store_query_mode"] == "sparse"

                    # Verify fusion retriever creation
                    mock_fusion.assert_called_once()
                    fusion_args = mock_fusion.call_args[1]
                    assert len(fusion_args["retrievers"]) == 2
                    assert fusion_args["similarity_top_k"] == 5
                    assert fusion_args["mode"] == "reciprocal_rerank"
                    assert fusion_args["use_async"] is True

                    assert result == mock_fusion_retriever

    def test_create_hybrid_retriever_fallback(self, mock_vector_index, mock_settings):
        """Test fallback to dense-only retriever when fusion fails."""
        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.VectorIndexRetriever") as mock_retriever,
            patch("utils.index_builder.QueryFusionRetriever") as mock_fusion
        ):
                    # Mock fusion retriever creation failure
                    mock_fusion.side_effect = Exception("Fusion failed")

                    mock_fallback_retriever = MagicMock()
                    mock_retriever.return_value = mock_fallback_retriever

                    with patch("utils.index_builder.logging.error") as mock_log_error:
                        with patch(
                            "utils.index_builder.logging.warning"
                        ) as mock_log_warning:
                            result = create_hybrid_retriever(mock_vector_index)

                            # Verify error logging
                            mock_log_error.assert_called_once()
                            mock_log_warning.assert_called_with(
                                "Using fallback dense-only retriever"
                            )

                            # Verify fallback retriever creation
                            assert result == mock_fallback_retriever


class TestIndexCreation:
    """Test vector index creation with GPU optimization and hybrid search."""

    @patch("utils.index_builder.verify_rrf_configuration")
    @patch("utils.index_builder.QdrantClient")
    @patch("utils.index_builder.setup_hybrid_qdrant")
    @patch("utils.index_builder.FastEmbedEmbedding")
    @patch("utils.index_builder.SparseTextEmbedding")
    @patch("utils.index_builder.VectorStoreIndex.from_documents")
    @patch("utils.index_builder.ensure_spacy_model")
    @patch("utils.index_builder.KnowledgeGraphIndex.from_documents")
    @patch("utils.index_builder.create_hybrid_retriever")
    @patch("utils.index_builder.torch.cuda.is_available", return_value=True)
    def test_create_index_with_gpu_optimization(
        self,
        mock_cuda_available,
        mock_create_retriever,
        mock_kg_index,
        mock_spacy_model,
        mock_vector_index,
        mock_sparse_embed,
        mock_dense_embed,
        mock_setup_qdrant,
        mock_client,
        mock_verify_rrf,
        sample_documents,
        mock_settings,
    ):
        """Test index creation with GPU optimization and CUDA streams."""
        # Mock RRF verification
        mock_verify_rrf.return_value = {"issues": [], "recommendations": []}

        # Mock vector store setup
        mock_vector_store = MagicMock()
        mock_setup_qdrant.return_value = mock_vector_store

        # Mock embedding models
        mock_dense_model = MagicMock()
        mock_sparse_model = MagicMock()
        mock_dense_embed.return_value = mock_dense_model
        mock_sparse_embed.return_value = mock_sparse_model

        # Mock index creation
        mock_index = MagicMock()
        mock_vector_index.return_value = mock_index

        # Mock spaCy model and KG index
        mock_nlp = MagicMock()
        mock_spacy_model.return_value = mock_nlp
        mock_kg = MagicMock()
        mock_kg_index.return_value = mock_kg

        # Mock hybrid retriever
        mock_retriever = MagicMock()
        mock_create_retriever.return_value = mock_retriever

        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.torch.cuda.Stream") as mock_stream,
            patch(
        ):
                    "utils.index_builder.StorageContext.from_defaults"
                ) as mock_storage:
                    mock_stream_instance = MagicMock()
                    mock_stream.return_value = mock_stream_instance
                    mock_storage_context = MagicMock()
                    mock_storage.return_value = mock_storage_context

                    result = create_index(sample_documents, use_gpu=True)

                    # Verify GPU stream usage
                    mock_stream.assert_called_once()
                    mock_stream_instance.synchronize.assert_called_once()

                    # Verify dense embedding model setup
                    mock_dense_embed.assert_called_once()
                    dense_args = mock_dense_embed.call_args[1]
                    assert dense_args["model_name"] == "BAAI/bge-large-en-v1.5"
                    assert dense_args["providers"] == [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ]
                    assert dense_args["batch_size"] == 32

                    # Verify sparse embedding model setup
                    mock_sparse_embed.assert_called_once()
                    sparse_args = mock_sparse_embed.call_args[1]
                    assert sparse_args["model_name"] == "prithivida/Splade_PP_en_v1"
                    assert sparse_args["providers"] == [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ]

                    # Verify Qdrant setup
                    mock_setup_qdrant.assert_called_once()
                    qdrant_args = mock_setup_qdrant.call_args[1]
                    assert qdrant_args["collection_name"] == "docmind"
                    assert qdrant_args["dense_embedding_size"] == 1024

                    # Verify vector index creation
                    mock_vector_index.assert_called_once()

                    # Verify KG index creation
                    mock_spacy_model.assert_called_once_with("en_core_web_sm")
                    mock_kg_index.assert_called_once()

                    # Verify result structure
                    assert result["vector"] == mock_index
                    assert result["kg"] == mock_kg
                    assert result["retriever"] == mock_retriever

    @patch("utils.index_builder.verify_rrf_configuration")
    @patch("utils.index_builder.QdrantClient")
    @patch("utils.index_builder.setup_hybrid_qdrant")
    @patch("utils.index_builder.FastEmbedEmbedding")
    @patch("utils.index_builder.SparseTextEmbedding")
    @patch("utils.index_builder.VectorStoreIndex.from_documents")
    @patch("utils.index_builder.torch.cuda.is_available", return_value=False)
    def test_create_index_cpu_mode(
        self,
        mock_cuda_available,
        mock_vector_index,
        mock_sparse_embed,
        mock_dense_embed,
        mock_setup_qdrant,
        mock_client,
        mock_verify_rrf,
        sample_documents,
        mock_settings,
    ):
        """Test index creation in CPU-only mode."""
        # Mock RRF verification
        mock_verify_rrf.return_value = {"issues": [], "recommendations": []}

        mock_vector_store = MagicMock()
        mock_setup_qdrant.return_value = mock_vector_store

        mock_dense_model = MagicMock()
        mock_sparse_model = MagicMock()
        mock_dense_embed.return_value = mock_dense_model
        mock_sparse_embed.return_value = mock_sparse_model

        mock_index = MagicMock()
        mock_vector_index.return_value = mock_index

        with patch("utils.index_builder.settings", mock_settings):
            with patch(
                "utils.index_builder.StorageContext.from_defaults"
            ) as mock_storage:
                mock_storage.return_value = MagicMock()

                result = create_index(sample_documents, use_gpu=False)

                # Verify CPU-only providers
                dense_args = mock_dense_embed.call_args[1]
                assert dense_args["providers"] == ["CPUExecutionProvider"]

                sparse_args = mock_sparse_embed.call_args[1]
                assert sparse_args["providers"] == ["CPUExecutionProvider"]

                # Verify index creation without CUDA streams
                mock_vector_index.assert_called_once()
                assert result["vector"] == mock_index

    @patch("utils.index_builder.verify_rrf_configuration")
    @patch("utils.index_builder.QdrantClient")
    def test_create_index_rrf_configuration_warnings(
        self, mock_client, mock_verify_rrf, sample_documents, mock_settings
    ):
        """Test index creation with RRF configuration warnings."""
        # Mock RRF verification with issues
        mock_verify_rrf.return_value = {
            "issues": ["Weights not research-backed", "Alpha out of range"],
            "recommendations": ["Use 0.7/0.3 weights", "Set alpha to 60"],
        }

        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.logging.warning") as mock_log_warning,
            patch("utils.index_builder.logging.info") as mock_log_info,
            patch("utils.index_builder.setup_hybrid_qdrant") as mock_setup,
            patch("utils.index_builder.FastEmbedEmbedding"),
            patch("utils.index_builder.SparseTextEmbedding"),
            patch(
        ):
                                    "utils.index_builder.VectorStoreIndex.from_documents"
                                ):
                                    create_index(sample_documents, use_gpu=False)

                                    # Verify warnings were logged
                                    mock_log_warning.assert_called()
                                    warning_message = mock_log_warning.call_args[0][0]
                                    assert "RRF Configuration Issues" in warning_message

                                    # Verify recommendations were logged
                                    assert mock_log_info.call_count >= 2


class TestAsyncIndexCreation:
    """Test asynchronous index creation with performance optimization."""

    @pytest.mark.asyncio
    @patch("utils.index_builder.verify_rrf_configuration")
    @patch("utils.index_builder.AsyncQdrantClient")
    @patch("utils.index_builder.setup_hybrid_qdrant_async")
    @patch("utils.index_builder.FastEmbedEmbedding")
    @patch("utils.index_builder.SparseTextEmbedding")
    @patch("utils.index_builder.VectorStoreIndex.from_documents")
    @patch("utils.index_builder.ensure_spacy_model")
    @patch("utils.index_builder.KnowledgeGraphIndex.from_documents")
    @patch("utils.index_builder.create_hybrid_retriever")
    @patch("utils.index_builder.torch.cuda.is_available", return_value=True)
    async def test_create_index_async_success(
        self,
        mock_cuda_available,
        mock_create_retriever,
        mock_kg_index,
        mock_spacy_model,
        mock_vector_index,
        mock_sparse_embed,
        mock_dense_embed,
        mock_setup_qdrant_async,
        mock_async_client,
        mock_verify_rrf,
        sample_documents,
        mock_settings,
    ):
        """Test successful async index creation with GPU optimization."""
        # Mock RRF verification
        mock_verify_rrf.return_value = {"issues": [], "recommendations": []}

        # Mock async client
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance

        # Mock vector store setup
        mock_vector_store = MagicMock()
        mock_setup_qdrant_async.return_value = mock_vector_store

        # Mock embedding models
        mock_dense_model = MagicMock()
        mock_sparse_model = MagicMock()
        mock_dense_embed.return_value = mock_dense_model
        mock_sparse_embed.return_value = mock_sparse_model

        # Mock index creation
        mock_index = MagicMock()
        mock_vector_index.return_value = mock_index

        # Mock spaCy and KG
        mock_nlp = MagicMock()
        mock_spacy_model.return_value = mock_nlp
        mock_kg = MagicMock()
        mock_kg_index.return_value = mock_kg

        # Mock hybrid retriever
        mock_retriever = MagicMock()
        mock_create_retriever.return_value = mock_retriever

        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.torch.cuda.Stream") as mock_stream,
            patch(
        ):
                    "utils.index_builder.StorageContext.from_defaults"
                ) as mock_storage:
                    mock_stream_instance = MagicMock()
                    mock_stream.return_value = mock_stream_instance
                    mock_storage.return_value = MagicMock()

                    result = await create_index_async(sample_documents, use_gpu=True)

                    # Verify async client was created and closed
                    mock_async_client.assert_called_once()
                    mock_client_instance.close.assert_called_once()

                    # Verify async Qdrant setup
                    mock_setup_qdrant_async.assert_called_once()
                    setup_args = mock_setup_qdrant_async.call_args[1]
                    assert setup_args["client"] == mock_client_instance
                    assert setup_args["collection_name"] == "docmind"
                    assert setup_args["dense_embedding_size"] == 1024

                    # Verify GPU stream usage
                    mock_stream.assert_called_once()
                    mock_stream_instance.synchronize.assert_called_once()

                    # Verify result structure
                    assert result["vector"] == mock_index
                    assert result["kg"] == mock_kg
                    assert result["retriever"] == mock_retriever

    @pytest.mark.asyncio
    @patch("utils.index_builder.verify_rrf_configuration")
    @patch("utils.index_builder.AsyncQdrantClient")
    @patch("utils.index_builder.ensure_spacy_model")
    @patch("utils.index_builder.KnowledgeGraphIndex.from_documents")
    async def test_create_index_async_kg_failure(
        self,
        mock_kg_index,
        mock_spacy_model,
        mock_async_client,
        mock_verify_rrf,
        sample_documents,
        mock_settings,
    ):
        """Test async index creation with KG index failure."""
        # Mock RRF verification
        mock_verify_rrf.return_value = {"issues": [], "recommendations": []}

        # Mock async client
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance

        # Mock spaCy model failure
        mock_spacy_model.side_effect = Exception("spaCy model failed")

        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.setup_hybrid_qdrant_async"),
            patch("utils.index_builder.FastEmbedEmbedding"),
            patch("utils.index_builder.SparseTextEmbedding"),
            patch(
        ):
                            "utils.index_builder.VectorStoreIndex.from_documents"
                        ) as mock_vector_index:
                            with patch(
                                "utils.index_builder.logging.warning"
                            ) as mock_log_warning:
                                mock_index = MagicMock()
                                mock_vector_index.return_value = mock_index

                                result = await create_index_async(
                                    sample_documents, use_gpu=False
                                )

                                # Verify warning was logged
                                mock_log_warning.assert_called()
                                warning_message = mock_log_warning.call_args[0][0]
                                assert "Failed to create KG index" in warning_message

                                # Verify KG index is None
                                assert result["kg"] is None
                                assert result["vector"] == mock_index


class TestMultimodalIndex:
    """Test multimodal index creation for text and image documents."""

    @patch("utils.index_builder.QdrantClient")
    @patch("utils.index_builder.QdrantVectorStore")
    @patch("utils.index_builder.HuggingFaceEmbedding")
    @patch("utils.index_builder.MultiModalVectorStoreIndex.from_documents")
    @patch("utils.index_builder.torch.cuda.is_available", return_value=True)
    def test_create_multimodal_index_success(
        self,
        mock_cuda_available,
        mock_multimodal_index,
        mock_hf_embedding,
        mock_vector_store,
        mock_client,
        sample_documents,
        sample_image_documents,
        mock_settings,
    ):
        """Test successful multimodal index creation with Jina v3 embeddings."""
        # Mock Jina v3 embedding model
        mock_embed_model = MagicMock()
        mock_hf_embedding.return_value = mock_embed_model

        # Mock vector store
        mock_store = MagicMock()
        mock_vector_store.return_value = mock_store

        # Mock multimodal index
        mock_index = MagicMock()
        mock_multimodal_index.return_value = mock_index

        # Combine text and image documents
        all_documents = sample_documents + sample_image_documents

        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.torch.cuda.Stream") as mock_stream,
            patch(
        ):
                    "utils.index_builder.StorageContext.from_defaults"
                ) as mock_storage:
                    mock_stream_instance = MagicMock()
                    mock_stream.return_value = mock_stream_instance
                    mock_storage.return_value = MagicMock()

                    result = create_multimodal_index(all_documents, use_gpu=True)

                    # Verify Jina v3 embedding model setup
                    mock_hf_embedding.assert_called_once()
                    embed_args = mock_hf_embedding.call_args[1]
                    assert embed_args["model_name"] == "jinaai/jina-embeddings-v3"
                    assert embed_args["device"] == "cuda"
                    assert embed_args["trust_remote_code"] is True

                    # Verify GPU stream usage
                    mock_stream.assert_called_once()
                    mock_stream_instance.synchronize.assert_called_once()

                    # Verify multimodal index creation
                    mock_multimodal_index.assert_called_once()
                    index_args = mock_multimodal_index.call_args[1]
                    assert len(index_args["documents"]) == 5  # 3 text + 2 image
                    assert index_args["embed_model"] == mock_embed_model

                    assert result == mock_index

    @patch("utils.index_builder.torch.cuda.is_available", return_value=False)
    def test_create_multimodal_index_cpu_mode(
        self, mock_cuda_available, sample_documents, mock_settings
    ):
        """Test multimodal index creation in CPU mode."""
        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.HuggingFaceEmbedding") as mock_hf_embedding,
            patch("utils.index_builder.QdrantClient"),
            patch("utils.index_builder.QdrantVectorStore"),
            patch(
        ):
                            "utils.index_builder.MultiModalVectorStoreIndex.from_documents"
                        ) as mock_multimodal_index:
                            mock_index = MagicMock()
                            mock_multimodal_index.return_value = mock_index

                            result = create_multimodal_index(
                                sample_documents, use_gpu=False
                            )

                            # Verify CPU device configuration
                            embed_args = mock_hf_embedding.call_args[1]
                            assert embed_args["device"] == "cpu"

                            assert result == mock_index

    @patch("utils.index_builder.HuggingFaceEmbedding")
    def test_create_multimodal_index_fallback(
        self, mock_hf_embedding, sample_documents, mock_settings
    ):
        """Test multimodal index creation with fallback to text-only."""
        # Mock embedding model failure
        mock_hf_embedding.side_effect = Exception("Embedding model failed")

        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.create_index") as mock_create_index,
            patch("utils.index_builder.logging.error") as mock_log_error,
            patch("utils.index_builder.logging.info") as mock_log_info
        ):
                        mock_fallback_result = {"vector": MagicMock()}
                        mock_create_index.return_value = mock_fallback_result

                        result = create_multimodal_index(
                            sample_documents, use_gpu=False
                        )

                        # Verify error and fallback messages
                        mock_log_error.assert_called()
                        mock_log_info.assert_called_with(
                            "Falling back to text-only vector index"
                        )

                        # Verify fallback index creation
                        mock_create_index.assert_called_once()
                        assert result == mock_fallback_result["vector"]

    @pytest.mark.asyncio
    @patch("utils.index_builder.AsyncQdrantClient")
    @patch("utils.index_builder.QdrantVectorStore")
    @patch("utils.index_builder.HuggingFaceEmbedding")
    @patch("utils.index_builder.MultiModalVectorStoreIndex.from_documents")
    @patch("utils.index_builder.torch.cuda.is_available", return_value=True)
    async def test_create_multimodal_index_async_success(
        self,
        mock_cuda_available,
        mock_multimodal_index,
        mock_hf_embedding,
        mock_vector_store,
        mock_async_client,
        sample_documents,
        sample_image_documents,
        mock_settings,
    ):
        """Test successful async multimodal index creation."""
        # Mock async client
        mock_client_instance = AsyncMock()
        mock_async_client.return_value = mock_client_instance

        # Mock embedding model and vector store
        mock_embed_model = MagicMock()
        mock_hf_embedding.return_value = mock_embed_model
        mock_store = MagicMock()
        mock_vector_store.return_value = mock_store

        # Mock multimodal index
        mock_index = MagicMock()
        mock_multimodal_index.return_value = mock_index

        all_documents = sample_documents + sample_image_documents

        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.torch.cuda.Stream") as mock_stream,
            patch("utils.index_builder.StorageContext.from_defaults")
        ):
                    mock_stream_instance = MagicMock()
                    mock_stream.return_value = mock_stream_instance

                    result = await create_multimodal_index_async(
                        all_documents, use_gpu=True
                    )

                    # Verify async client was used and closed
                    mock_async_client.assert_called_once()
                    mock_client_instance.close.assert_called_once()

                    # Verify async vector store setup
                    vector_store_args = mock_vector_store.call_args[1]
                    assert vector_store_args["aclient"] == mock_client_instance

                    assert result == mock_index

    @pytest.mark.asyncio
    @patch("utils.index_builder.create_multimodal_index")
    async def test_create_multimodal_index_async_fallback(
        self, mock_create_multimodal, sample_documents, mock_settings
    ):
        """Test async multimodal index creation with fallback to sync version."""
        with patch("utils.index_builder.AsyncQdrantClient") as mock_async_client:
            # Mock async client creation failure
            mock_async_client.side_effect = Exception("Async client failed")

            with patch("utils.index_builder.logging.error") as mock_log_error:
                mock_fallback_result = MagicMock()
                mock_create_multimodal.return_value = mock_fallback_result

                result = await create_multimodal_index_async(
                    sample_documents, use_gpu=False
                )

                # Verify error was logged
                mock_log_error.assert_called()

                # Verify fallback to sync version
                mock_create_multimodal.assert_called_once()
                assert result == mock_fallback_result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_create_index_empty_documents(self, mock_settings):
        """Test index creation with empty document list."""
        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.verify_rrf_configuration") as mock_verify
        ):
                mock_verify.return_value = {"issues": [], "recommendations": []}

                with (
                    patch("utils.index_builder.QdrantClient"),
                    patch("utils.index_builder.setup_hybrid_qdrant"),
                    patch("utils.index_builder.FastEmbedEmbedding"),
                    patch("utils.index_builder.SparseTextEmbedding"),
                    patch(
                ):
                                    "utils.index_builder.VectorStoreIndex.from_documents"
                                ) as mock_vector_index:
                                    mock_index = MagicMock()
                                    mock_vector_index.return_value = mock_index

                                    result = create_index([], use_gpu=False)

                                    # Should still create index with empty documents
                                    assert result["vector"] == mock_index

    @pytest.mark.parametrize("debug_mode", [True, False])
    @patch("utils.index_builder.torch.cuda.is_available", return_value=True)
    def test_create_index_debug_mode(
        self, mock_cuda_available, debug_mode, sample_documents, mock_settings
    ):
        """Test index creation with debug mode profiling."""
        mock_settings.debug_mode = debug_mode

        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.verify_rrf_configuration") as mock_verify
        ):
                mock_verify.return_value = {"issues": [], "recommendations": []}

                with patch(
                    "utils.index_builder.torch.profiler.profile"
                ) as mock_profiler:
                    with (
                        patch("utils.index_builder.QdrantClient"),
                        patch("utils.index_builder.setup_hybrid_qdrant"),
                        patch("utils.index_builder.FastEmbedEmbedding"),
                        patch("utils.index_builder.SparseTextEmbedding"),
                        patch(
                    ):
                                        "utils.index_builder.VectorStoreIndex.from_documents"
                                    ):
                                        with patch(
                                            "utils.index_builder.torch.cuda.Stream"
                                        ) as mock_stream:
                                            mock_stream_instance = MagicMock()
                                            mock_stream.return_value = (
                                                mock_stream_instance
                                            )

                                            create_index(sample_documents, use_gpu=True)

                                            if debug_mode:
                                                # Verify profiler was used in debug mode
                                                mock_profiler.assert_called_once()
                                            else:
                                                # Verify profiler was not used in normal mode
                                                mock_profiler.assert_not_called()

    def test_create_index_configuration_error(self, sample_documents, mock_settings):
        """Test index creation with configuration validation error."""
        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.verify_rrf_configuration"),
            patch("utils.index_builder.QdrantClient") as mock_client
        ):
                    # Mock client creation failure
                    mock_client.side_effect = ValueError("Invalid configuration")

                    with pytest.raises(ValueError):
                        create_index(sample_documents, use_gpu=False)

    @patch("utils.index_builder.logging.info")
    def test_hybrid_retriever_logging(
        self, mock_log_info, mock_vector_index, mock_settings
    ):
        """Test that hybrid retriever creation logs appropriate information."""
        with (
            patch("utils.index_builder.settings", mock_settings),
            patch("utils.index_builder.VectorIndexRetriever") as mock_retriever,
            patch("utils.index_builder.QueryFusionRetriever") as mock_fusion
        ):
                    mock_fusion_retriever = MagicMock()
                    mock_fusion.return_value = mock_fusion_retriever

                    create_hybrid_retriever(mock_vector_index)

                    # Verify logging information
                    logged_messages = [
                        call[0][0] for call in mock_log_info.call_args_list
                    ]
                    hybrid_message = next(
                        (
                            msg
                            for msg in logged_messages
                            if "HybridFusionRetriever created" in msg
                        ),
                        None,
                    )
                    assert hybrid_message is not None
                    assert "Dense prefetch: 10" in hybrid_message
                    assert "Final top_k: 5" in hybrid_message
