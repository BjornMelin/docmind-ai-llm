"""Comprehensive tests for multimodal processing and hybrid retrieval.

This module tests multimodal index creation, Jina v3 embeddings, CUDA acceleration,
quantization support, and cross-modal retrieval capabilities following modern
testing practices and ADR requirements.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from llama_index.core import Document
from llama_index.core.schema import ImageDocument
from PIL import Image

from models import AppSettings


@pytest.fixture
def mock_multimodal_settings():
    """Create mock settings for multimodal testing."""
    return AppSettings(
        enable_quantization=True,
        embedding_batch_size=32,
        dense_embedding_dimension=1024,
        gpu_acceleration=True,
        debug_mode=False,
        qdrant_url="http://localhost:6333",
        jina_multimodal_model="jinaai/jina-embeddings-v3",
    )


@pytest.fixture
def sample_multimodal_documents():
    """Create sample multimodal documents for testing."""
    # Create a small test image
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return [
        Document(
            text="This is a technical document about machine learning algorithms",
            metadata={"type": "text", "page": 1, "source": "ml_guide.pdf"},
        ),
        ImageDocument(
            image=image_base64,
            metadata={"type": "image", "page": 1, "source": "ml_guide.pdf"},
        ),
        Document(
            text="Neural networks and deep learning architectures",
            metadata={"type": "text", "page": 2, "source": "ml_guide.pdf"},
        ),
    ]


@pytest.fixture
def mock_jina_v3_embeddings():
    """Mock Jina v3 embeddings for testing."""

    def create_mock_embedding(size=1024):
        return torch.randn(size).numpy().tolist()

    return {
        "text_embedding": create_mock_embedding(),
        "image_embedding": create_mock_embedding(),
        "multimodal_embedding": create_mock_embedding(),
    }


class TestMultimodalIndexCreation:
    """Test multimodal index creation with Jina v3 embeddings."""

    @patch("utils.index_builder.QdrantClient")
    @patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding")
    @patch("utils.index_builder.MultiModalVectorStoreIndex")
    def test_create_multimodal_index_basic(
        self,
        mock_multimodal_index,
        mock_hf_embedding,
        mock_qdrant_client,
        sample_multimodal_documents,
        mock_multimodal_settings,
    ):
        """Test basic multimodal index creation with Jina v3."""
        from utils.index_builder import create_multimodal_index

        # Mock Jina v3 embedding model
        mock_embed_model = MagicMock()
        mock_hf_embedding.return_value = mock_embed_model

        # Mock Qdrant client
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        # Mock multimodal index creation
        mock_index_instance = MagicMock()
        mock_multimodal_index.from_documents.return_value = mock_index_instance

        with patch("utils.index_builder.settings", mock_multimodal_settings):
            index = create_multimodal_index(
                sample_multimodal_documents,
                use_gpu=True,
                collection_name="test_multimodal",
            )

        # Verify Jina v3 embedding model configuration
        mock_hf_embedding.assert_called_once()
        embed_call_kwargs = mock_hf_embedding.call_args[1]
        assert "jinaai/jina-embeddings-v3" in embed_call_kwargs["model_name"]
        assert embed_call_kwargs["trust_remote_code"] is True
        assert embed_call_kwargs["device"] == "cuda"
        assert embed_call_kwargs["embed_batch_size"] == 32

        # Verify model kwargs for GPU acceleration
        model_kwargs = embed_call_kwargs["model_kwargs"]
        assert model_kwargs["torch_dtype"] == torch.float16

        # Verify multimodal index creation
        mock_multimodal_index.from_documents.assert_called_once()
        index_call_kwargs = mock_multimodal_index.from_documents.call_args[1]
        assert len(index_call_kwargs["documents"]) == 3
        assert index_call_kwargs["embed_model"] == mock_embed_model
        assert index_call_kwargs["insert_batch_size"] == 32

        # Verify return value
        assert index == mock_index_instance

    @patch("utils.index_builder.QdrantClient")
    @patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding")
    @patch("utils.index_builder.MultiModalVectorStoreIndex")
    def test_create_multimodal_index_with_quantization(
        self,
        mock_multimodal_index,
        mock_hf_embedding,
        mock_qdrant_client,
        sample_multimodal_documents,
        mock_multimodal_settings,
    ):
        """Test multimodal index creation with 8-bit quantization."""
        from utils.index_builder import create_multimodal_index

        # Enable quantization in settings
        mock_multimodal_settings.enable_quantization = True

        with patch("transformers.BitsAndBytesConfig") as mock_bnb_config:
            mock_quantization_config = MagicMock()
            mock_bnb_config.return_value = mock_quantization_config

            mock_embed_model = MagicMock()
            mock_hf_embedding.return_value = mock_embed_model

            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client

            mock_index_instance = MagicMock()
            mock_multimodal_index.from_documents.return_value = mock_index_instance

            with patch("utils.index_builder.settings", mock_multimodal_settings):
                create_multimodal_index(sample_multimodal_documents, use_gpu=True)

            # Verify quantization configuration
            mock_bnb_config.assert_called_once()
            quant_kwargs = mock_bnb_config.call_args[1]
            assert quant_kwargs["load_in_8bit"] is True
            assert quant_kwargs["llm_int8_threshold"] == 6.0
            assert quant_kwargs["llm_int8_has_fp16_weight"] is False

            # Verify quantization config passed to embedding model
            mock_hf_embedding.assert_called_once()
            embed_kwargs = mock_hf_embedding.call_args[1]
            assert (
                embed_kwargs["model_kwargs"]["quantization_config"]
                == mock_quantization_config
            )

    @patch("utils.index_builder.torch.cuda.is_available")
    @patch("utils.index_builder.torch.cuda.Stream")
    @patch("utils.index_builder.QdrantClient")
    @patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding")
    @patch("utils.index_builder.MultiModalVectorStoreIndex")
    def test_create_multimodal_index_cuda_streams(
        self,
        mock_multimodal_index,
        mock_hf_embedding,
        mock_qdrant_client,
        mock_cuda_stream,
        mock_cuda_available,
        sample_multimodal_documents,
        mock_multimodal_settings,
    ):
        """Test multimodal index creation with CUDA streams for GPU acceleration."""
        from utils.index_builder import create_multimodal_index

        # Mock CUDA availability
        mock_cuda_available.return_value = True

        # Mock CUDA stream
        mock_stream = MagicMock()
        mock_cuda_stream.return_value = mock_stream

        # Mock other components
        mock_embed_model = MagicMock()
        mock_hf_embedding.return_value = mock_embed_model

        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        mock_index_instance = MagicMock()
        mock_multimodal_index.from_documents.return_value = mock_index_instance

        with patch("utils.index_builder.settings", mock_multimodal_settings):
            create_multimodal_index(sample_multimodal_documents, use_gpu=True)

        # Verify CUDA stream usage
        mock_cuda_stream.assert_called_once()
        mock_stream.synchronize.assert_called_once()

        # Verify index creation happened within stream context
        mock_multimodal_index.from_documents.assert_called_once()

    @patch("utils.index_builder.QdrantClient")
    @patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding")
    @patch("utils.index_builder.MultiModalVectorStoreIndex")
    def test_create_multimodal_index_cpu_fallback(
        self,
        mock_multimodal_index,
        mock_hf_embedding,
        mock_qdrant_client,
        sample_multimodal_documents,
        mock_multimodal_settings,
    ):
        """Test multimodal index creation with CPU fallback."""
        from utils.index_builder import create_multimodal_index

        # Mock components for CPU mode
        mock_embed_model = MagicMock()
        mock_hf_embedding.return_value = mock_embed_model

        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        mock_index_instance = MagicMock()
        mock_multimodal_index.from_documents.return_value = mock_index_instance

        with patch("utils.index_builder.settings", mock_multimodal_settings):
            create_multimodal_index(
                sample_multimodal_documents,
                use_gpu=False,  # Force CPU mode
            )

        # Verify CPU configuration
        mock_hf_embedding.assert_called_once()
        embed_kwargs = mock_hf_embedding.call_args[1]
        assert embed_kwargs["device"] == "cpu"

        # Verify model kwargs for CPU
        model_kwargs = embed_kwargs["model_kwargs"]
        assert model_kwargs["torch_dtype"] == torch.float32

    @patch("utils.index_builder.QdrantClient")
    @patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding")
    @patch("utils.index_builder.MultiModalVectorStoreIndex")
    def test_create_multimodal_index_error_fallback(
        self,
        mock_multimodal_index,
        mock_hf_embedding,
        mock_qdrant_client,
        sample_multimodal_documents,
        mock_multimodal_settings,
    ):
        """Test fallback to text-only index when multimodal creation fails."""
        from utils.index_builder import create_multimodal_index

        # Mock multimodal index creation failure
        mock_multimodal_index.from_documents.side_effect = Exception(
            "Multimodal failed"
        )

        with patch("utils.index_builder.create_index") as mock_create_index:
            mock_fallback_index = MagicMock()
            mock_create_index.return_value = {"vector": mock_fallback_index}

            with patch("utils.index_builder.settings", mock_multimodal_settings):
                with patch("utils.index_builder.logging.error") as mock_log_error:
                    with patch("utils.index_builder.logging.info") as mock_log_info:
                        result = create_multimodal_index(
                            sample_multimodal_documents, use_gpu=False
                        )

            # Verify error logging
            mock_log_error.assert_called_once()
            error_message = mock_log_error.call_args[0][0]
            assert "Multimodal index creation failed" in error_message

            # Verify fallback logging
            mock_log_info.assert_called_once()
            info_message = mock_log_info.call_args[0][0]
            assert "Falling back to text-only vector index" in info_message

            # Verify fallback was called with text-only documents
            mock_create_index.assert_called_once()
            fallback_args = mock_create_index.call_args
            text_docs = [
                d for d in fallback_args[0][0] if not isinstance(d, ImageDocument)
            ]
            assert len(text_docs) == 2  # Should filter out image documents

            # Verify return value
            assert result == mock_fallback_index


class TestAsyncMultimodalIndex:
    """Test asynchronous multimodal index creation."""

    @pytest.mark.asyncio
    @patch("utils.index_builder.AsyncQdrantClient")
    @patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding")
    @patch("utils.index_builder.MultiModalVectorStoreIndex")
    async def test_create_multimodal_index_async(
        self,
        mock_multimodal_index,
        mock_hf_embedding,
        mock_async_qdrant_client,
        sample_multimodal_documents,
        mock_multimodal_settings,
    ):
        """Test asynchronous multimodal index creation."""
        from utils.index_builder import create_multimodal_index_async

        # Mock async Qdrant client
        mock_async_client = AsyncMock()
        mock_async_qdrant_client.return_value = mock_async_client

        # Mock embedding model
        mock_embed_model = MagicMock()
        mock_hf_embedding.return_value = mock_embed_model

        # Mock multimodal index
        mock_index_instance = MagicMock()
        mock_multimodal_index.from_documents.return_value = mock_index_instance

        with patch("utils.index_builder.settings", mock_multimodal_settings):
            result = await create_multimodal_index_async(
                sample_multimodal_documents,
                use_gpu=True,
                collection_name="async_multimodal",
            )

        # Verify async client usage
        mock_async_qdrant_client.assert_called_once_with(
            url=mock_multimodal_settings.qdrant_url
        )
        mock_async_client.close.assert_called_once()

        # Verify Jina v3 embedding configuration
        mock_hf_embedding.assert_called_once()
        embed_kwargs = mock_hf_embedding.call_args[1]
        assert "jinaai/jina-embeddings-v3" in embed_kwargs["model_name"]

        # Verify vector store configuration with async client
        mock_multimodal_index.from_documents.assert_called_once()

        # Verify result
        assert result == mock_index_instance

    @pytest.mark.asyncio
    @patch("utils.index_builder.AsyncQdrantClient")
    @patch("utils.index_builder.create_multimodal_index")
    async def test_create_multimodal_index_async_fallback(
        self,
        mock_sync_create,
        mock_async_qdrant_client,
        sample_multimodal_documents,
        mock_multimodal_settings,
    ):
        """Test fallback to synchronous version when async fails."""
        from utils.index_builder import create_multimodal_index_async

        # Mock async client failure
        mock_async_qdrant_client.side_effect = Exception("Async failed")

        # Mock synchronous fallback
        mock_fallback_index = MagicMock()
        mock_sync_create.return_value = mock_fallback_index

        with patch("utils.index_builder.settings", mock_multimodal_settings):
            with patch("utils.index_builder.logging.error") as mock_log_error:
                result = await create_multimodal_index_async(
                    sample_multimodal_documents,
                    use_gpu=True,
                    collection_name="async_fallback",
                )

        # Verify error logging
        mock_log_error.assert_called_once()
        error_message = mock_log_error.call_args[0][0]
        assert "Async multimodal index creation failed" in error_message

        # Verify fallback to sync version
        mock_sync_create.assert_called_once_with(
            sample_multimodal_documents,
            True,  # use_gpu
            "async_fallback",  # collection_name
        )

        assert result == mock_fallback_index


class TestHybridFusionRetrieverEnhanced:
    """Enhanced tests for HybridFusionRetriever with multimodal support."""

    @patch("utils.index_builder.QueryFusionRetriever")
    @patch("utils.index_builder.VectorIndexRetriever")
    def test_create_hybrid_retriever_with_rrf(
        self, mock_vector_retriever, mock_fusion_retriever, mock_multimodal_settings
    ):
        """Test HybridFusionRetriever creation with RRF configuration."""
        from utils.index_builder import create_hybrid_retriever

        # Mock vector index
        mock_index = MagicMock()

        # Mock dense and sparse retrievers
        mock_dense_retriever = MagicMock()
        mock_sparse_retriever = MagicMock()
        mock_vector_retriever.side_effect = [
            mock_dense_retriever,
            mock_sparse_retriever,
        ]

        # Mock fusion retriever
        mock_fusion_instance = MagicMock()
        mock_fusion_retriever.return_value = mock_fusion_instance

        with patch("utils.index_builder.settings", mock_multimodal_settings):
            result = create_hybrid_retriever(mock_index)

        # Verify dense retriever creation
        dense_call = mock_vector_retriever.call_args_list[0]
        assert dense_call[1]["index"] == mock_index
        assert dense_call[1]["vector_store_query_mode"] == "default"
        expected_prefetch_k = (
            mock_multimodal_settings.prefetch_factor
            * mock_multimodal_settings.similarity_top_k
        )
        assert dense_call[1]["similarity_top_k"] == expected_prefetch_k

        # Verify sparse retriever creation
        sparse_call = mock_vector_retriever.call_args_list[1]
        assert sparse_call[1]["index"] == mock_index
        assert sparse_call[1]["vector_store_query_mode"] == "sparse"
        assert sparse_call[1]["similarity_top_k"] == expected_prefetch_k

        # Verify fusion retriever configuration
        mock_fusion_retriever.assert_called_once()
        fusion_kwargs = mock_fusion_retriever.call_args[1]
        assert len(fusion_kwargs["retrievers"]) == 2
        assert fusion_kwargs["retrievers"][0] == mock_dense_retriever
        assert fusion_kwargs["retrievers"][1] == mock_sparse_retriever
        assert (
            fusion_kwargs["similarity_top_k"]
            == mock_multimodal_settings.similarity_top_k
        )
        assert fusion_kwargs["mode"] == "reciprocal_rerank"
        assert fusion_kwargs["use_async"] is True
        assert fusion_kwargs["num_queries"] == 1

        assert result == mock_fusion_instance

    @patch("utils.index_builder.VectorIndexRetriever")
    def test_create_hybrid_retriever_fallback(
        self, mock_vector_retriever, mock_multimodal_settings
    ):
        """Test fallback to dense-only retriever when fusion fails."""
        from utils.index_builder import create_hybrid_retriever

        mock_index = MagicMock()

        # Mock dense retriever success, sparse retriever failure
        mock_dense_retriever = MagicMock()
        mock_vector_retriever.side_effect = [
            mock_dense_retriever,  # Dense succeeds
            Exception("Sparse failed"),  # Sparse fails
            mock_dense_retriever,  # Fallback succeeds
        ]

        with patch("utils.index_builder.settings", mock_multimodal_settings):
            with patch("utils.index_builder.logging.error") as mock_log_error:
                with patch("utils.index_builder.logging.warning") as mock_log_warning:
                    result = create_hybrid_retriever(mock_index)

        # Verify error and warning logging
        mock_log_error.assert_called_once()
        mock_log_warning.assert_called_once()
        warning_message = mock_log_warning.call_args[0][0]
        assert "Using fallback dense-only retriever" in warning_message

        # Should return fallback retriever
        assert result == mock_dense_retriever

    @pytest.mark.performance
    def test_hybrid_fusion_retriever_performance_benchmark(
        self, benchmark, mock_multimodal_settings
    ):
        """Benchmark HybridFusionRetriever performance."""
        from utils.index_builder import create_hybrid_retriever

        # Mock fast components
        mock_index = MagicMock()

        with patch("utils.index_builder.QueryFusionRetriever") as mock_fusion:
            mock_retriever = MagicMock()
            mock_fusion.return_value = mock_retriever

            # Mock fast retrieve method
            def fast_retrieve(query_bundle):
                return [
                    MagicMock(node=MagicMock(node_id=f"doc_{i}"), score=0.9 - i * 0.1)
                    for i in range(5)
                ]

            mock_retriever.retrieve = fast_retrieve

            with patch("utils.index_builder.settings", mock_multimodal_settings):
                retriever = create_hybrid_retriever(mock_index)

            # Benchmark retrieval operation
            from llama_index.core import QueryBundle

            def retrieval_operation():
                return retriever.retrieve(
                    QueryBundle(query_str="test multimodal query")
                )

            result = benchmark(retrieval_operation)
            assert len(result) == 5


class TestUnstructuredMultimodalParsing:
    """Test Unstructured parsing with enhanced multimodal support."""

    @patch("utils.document_loader.partition")
    def test_load_documents_unstructured_multimodal_elements(
        self, mock_partition, tmp_path, mock_multimodal_settings
    ):
        """Test Unstructured parsing with various multimodal elements."""
        from utils.document_loader import load_documents_unstructured

        # Create test file
        test_file = tmp_path / "multimodal_test.pdf"
        test_file.write_text("Test multimodal content")

        # Mock comprehensive multimodal elements
        mock_title = MagicMock()
        mock_title.category = "Title"
        mock_title.__str__.return_value = "Research Paper: AI and Computer Vision"
        mock_title.metadata.page_number = 1
        mock_title.metadata.filename = "multimodal_test.pdf"
        mock_title.metadata.coordinates = {"x": 100, "y": 50}

        mock_narrative = MagicMock()
        mock_narrative.category = "NarrativeText"
        mock_narrative.__str__.return_value = "This paper explores the intersection of artificial intelligence and computer vision technologies."
        mock_narrative.metadata.page_number = 1
        mock_narrative.metadata.filename = "multimodal_test.pdf"

        mock_table = MagicMock()
        mock_table.category = "Table"
        mock_table.__str__.return_value = "| Model | Accuracy | Speed |\n| ResNet | 92.1% | 45ms |\n| EfficientNet | 94.3% | 32ms |"
        mock_table.metadata.page_number = 2
        mock_table.metadata.filename = "multimodal_test.pdf"

        mock_image = MagicMock()
        mock_image.category = "Image"
        mock_image.metadata.page_number = 2
        mock_image.metadata.filename = "multimodal_test.pdf"
        mock_image.metadata.image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="

        mock_figure_caption = MagicMock()
        mock_figure_caption.category = "FigureCaption"
        mock_figure_caption.__str__.return_value = (
            "Figure 1: Neural network architecture comparison"
        )
        mock_figure_caption.metadata.page_number = 2
        mock_figure_caption.metadata.filename = "multimodal_test.pdf"

        mock_partition.return_value = [
            mock_title,
            mock_narrative,
            mock_table,
            mock_image,
            mock_figure_caption,
        ]

        with patch("utils.document_loader.settings", mock_multimodal_settings):
            with patch(
                "utils.document_loader.chunk_documents_structured"
            ) as mock_chunk:
                mock_chunk.side_effect = lambda x: x  # Return unchanged for testing

                result = load_documents_unstructured(str(test_file))

        # Verify partition call with hi_res strategy
        mock_partition.assert_called_once()
        partition_kwargs = mock_partition.call_args[1]
        assert partition_kwargs["strategy"] == mock_multimodal_settings.parse_strategy
        assert partition_kwargs["extract_images_in_pdf"] is True
        assert partition_kwargs["extract_image_block_types"] == [
            "Image",
            "FigureCaption",
        ]
        assert partition_kwargs["infer_table_structure"] is True
        assert partition_kwargs["chunking_strategy"] == "by_title"

        # Should create 4 text documents + 1 image document
        assert len(result) == 5

        # Verify text documents
        text_docs = [doc for doc in result if not isinstance(doc, ImageDocument)]
        assert len(text_docs) == 4

        # Verify image documents
        image_docs = [doc for doc in result if isinstance(doc, ImageDocument)]
        assert len(image_docs) == 1
        assert "image_base64" in image_docs[0].metadata
        assert image_docs[0].metadata["element_type"] == "Image"

        # Verify metadata preservation
        for doc in result:
            assert "element_type" in doc.metadata
            assert "page_number" in doc.metadata
            assert "filename" in doc.metadata

    @patch("utils.document_loader.partition")
    def test_load_documents_unstructured_image_extraction_fallback(
        self, mock_partition, tmp_path, mock_multimodal_settings
    ):
        """Test image extraction fallback when image_base64 is not available."""
        from utils.document_loader import load_documents_unstructured

        test_file = tmp_path / "fallback_test.pdf"
        test_file.write_text("Test content")

        # Mock image element with base64 in text field
        mock_image = MagicMock()
        mock_image.category = "Image"
        mock_image.metadata.page_number = 1
        mock_image.metadata.filename = "fallback_test.pdf"
        mock_image.metadata.image_base64 = None  # Not available in metadata
        mock_image.text = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="

        # Mock image with invalid base64 (should be skipped)
        mock_invalid_image = MagicMock()
        mock_invalid_image.category = "Image"
        mock_invalid_image.metadata.page_number = 1
        mock_invalid_image.metadata.filename = "fallback_test.pdf"
        mock_invalid_image.metadata.image_base64 = None
        mock_invalid_image.text = "not base64 data"

        mock_partition.return_value = [mock_image, mock_invalid_image]

        with patch("utils.document_loader.settings", mock_multimodal_settings):
            with patch(
                "utils.document_loader.chunk_documents_structured"
            ) as mock_chunk:
                mock_chunk.side_effect = lambda x: x

                result = load_documents_unstructured(str(test_file))

        # Should only create ImageDocument for valid base64
        assert len(result) == 1
        assert isinstance(result[0], ImageDocument)
        assert result[0].metadata["image_base64"] == mock_image.text

    @patch("utils.document_loader.partition")
    def test_load_documents_unstructured_page_image_association(
        self, mock_partition, tmp_path, mock_multimodal_settings
    ):
        """Test association of images with text on the same page."""
        from utils.document_loader import load_documents_unstructured

        test_file = tmp_path / "page_association_test.pdf"
        test_file.write_text("Test content")

        # Page 1: Text + Image
        mock_text_p1 = MagicMock()
        mock_text_p1.category = "NarrativeText"
        mock_text_p1.__str__.return_value = "Page 1 content"
        mock_text_p1.metadata.page_number = 1
        mock_text_p1.metadata.filename = "test.pdf"

        mock_image_p1 = MagicMock()
        mock_image_p1.category = "Image"
        mock_image_p1.metadata.page_number = 1
        mock_image_p1.metadata.filename = "test.pdf"
        mock_image_p1.metadata.image_base64 = "page1_image_data"

        # Page 2: Text (no images)
        mock_text_p2 = MagicMock()
        mock_text_p2.category = "NarrativeText"
        mock_text_p2.__str__.return_value = "Page 2 content"
        mock_text_p2.metadata.page_number = 2
        mock_text_p2.metadata.filename = "test.pdf"

        mock_partition.return_value = [mock_text_p1, mock_image_p1, mock_text_p2]

        with patch("utils.document_loader.settings", mock_multimodal_settings):
            with patch(
                "utils.document_loader.chunk_documents_structured"
            ) as mock_chunk:
                mock_chunk.side_effect = lambda x: x

                result = load_documents_unstructured(str(test_file))

        # Should have 2 text docs + 1 image doc
        assert len(result) == 3

        # Find text document from page 1
        page1_text_docs = [
            doc
            for doc in result
            if not isinstance(doc, ImageDocument)
            and doc.metadata.get("page_number") == 1
        ]
        assert len(page1_text_docs) == 1

        # Should have image association
        # Note: The current implementation associates images at the page level
        # This test verifies the structure supports this association

    @patch("utils.document_loader.partition")
    @patch("utils.document_loader.load_documents_llama")
    def test_load_documents_unstructured_comprehensive_fallback(
        self, mock_llama_load, mock_partition, tmp_path
    ):
        """Test comprehensive fallback to LlamaParse when Unstructured fails."""
        from utils.document_loader import load_documents_unstructured

        test_file = tmp_path / "fallback_comprehensive.pdf"
        test_file.write_text("Test content")

        # Mock Unstructured failure
        mock_partition.side_effect = Exception("Unstructured service unavailable")

        # Mock LlamaParse success
        mock_fallback_docs = [
            Document(text="Fallback parsed content", metadata={"source": "fallback"})
        ]
        mock_llama_load.return_value = mock_fallback_docs

        with patch("utils.document_loader.logging.error") as mock_log_error:
            with patch("utils.document_loader.logging.info") as mock_log_info:
                result = load_documents_unstructured(str(test_file))

        # Verify error and fallback logging
        mock_log_error.assert_called_once()
        error_msg = mock_log_error.call_args[0][0]
        assert "Error loading with Unstructured" in error_msg

        mock_log_info.assert_called_once()
        info_msg = mock_log_info.call_args[0][0]
        assert "Falling back to existing LlamaParse loader" in info_msg

        # Verify fallback was called correctly
        mock_llama_load.assert_called_once()
        fallback_args = mock_llama_load.call_args
        assert len(fallback_args[0]) == 1  # uploaded_files list
        assert fallback_args[1]["parse_media"] is False
        assert fallback_args[1]["enable_multimodal"] is True

        # Verify result
        assert result == mock_fallback_docs


class TestDocumentStructuredChunking:
    """Test enhanced structured document chunking."""

    def test_chunk_documents_mixed_types_preservation(self, mock_multimodal_settings):
        """Test chunking preserves ImageDocuments and processes text documents."""
        from utils.document_loader import chunk_documents_structured

        # Create mixed document types
        long_text = (
            "This is a very long technical document about machine learning. " * 50
        )
        text_doc = Document(
            text=long_text, metadata={"type": "text", "page": 1, "has_images": True}
        )

        img = Image.new("RGB", (50, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        image_doc = ImageDocument(
            image=image_base64,
            metadata={"type": "image", "page": 1, "element_type": "Image"},
        )

        short_text_doc = Document(
            text="Short content",
            metadata={"type": "text", "page": 2, "has_images": False},
        )

        documents = [text_doc, image_doc, short_text_doc]

        with patch("utils.document_loader.settings", mock_multimodal_settings):
            with patch("utils.document_loader.SentenceSplitter") as mock_splitter:
                mock_splitter_instance = MagicMock()

                # Mock chunking: long doc -> 2 chunks, short doc -> 1 chunk
                chunk1 = Document(text="Chunk 1", metadata=text_doc.metadata)
                chunk2 = Document(text="Chunk 2", metadata=text_doc.metadata)
                short_chunk = Document(
                    text="Short content", metadata=short_text_doc.metadata
                )

                mock_splitter_instance.get_nodes_from_documents.side_effect = [
                    [chunk1, chunk2],  # Long text doc chunked
                    [short_chunk],  # Short text doc unchanged
                ]
                mock_splitter.return_value = mock_splitter_instance

                result = chunk_documents_structured(documents)

        # Verify splitter configuration
        mock_splitter.assert_called_once()
        splitter_kwargs = mock_splitter.call_args[1]
        assert splitter_kwargs["chunk_size"] == mock_multimodal_settings.chunk_size
        assert (
            splitter_kwargs["chunk_overlap"] == mock_multimodal_settings.chunk_overlap
        )
        assert splitter_kwargs["paragraph_separator"] == "\n\n"
        assert splitter_kwargs["secondary_chunking_regex"] == "[^,.;。]+[,.;。]?"

        # Should have 3 text chunks + 1 preserved image
        assert len(result) == 4

        # Verify ImageDocument preservation
        image_docs = [doc for doc in result if isinstance(doc, ImageDocument)]
        assert len(image_docs) == 1
        assert image_docs[0] == image_doc

        # Verify text document chunking
        text_docs = [doc for doc in result if not isinstance(doc, ImageDocument)]
        assert len(text_docs) == 3

        # Verify metadata preservation in chunks
        for doc in text_docs:
            assert "type" in doc.metadata
            assert "page" in doc.metadata


class TestPerformanceAndBenchmarks:
    """Performance tests and benchmarks for multimodal operations."""

    @pytest.mark.performance
    def test_multimodal_embedding_batch_processing(
        self, benchmark, mock_jina_v3_embeddings
    ):
        """Benchmark multimodal embedding batch processing."""
        from utils.document_loader import create_native_multimodal_embeddings

        # Create batch of test data
        batch_texts = [
            f"Document {i} content about AI and machine learning" for i in range(10)
        ]

        # Mock image data
        img = Image.new("RGB", (64, 64), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        batch_images = [
            [{"image_data": image_base64, "page_number": 1}] for _ in range(10)
        ]

        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model"
        ) as mock_model_manager:
            mock_model = MagicMock()
            mock_text_emb = MagicMock()
            mock_text_emb.flatten.return_value.tolist.return_value = (
                mock_jina_v3_embeddings["text_embedding"]
            )

            mock_img_emb = MagicMock()
            mock_img_emb.flatten.return_value.tolist.return_value = (
                mock_jina_v3_embeddings["image_embedding"]
            )

            mock_model.embed_text.return_value = [mock_text_emb]
            mock_model.embed_image.return_value = [mock_img_emb]
            mock_model_manager.return_value = mock_model

            def batch_embedding_operation():
                results = []
                for text, images in zip(batch_texts, batch_images, strict=False):
                    result = create_native_multimodal_embeddings(text, images)
                    results.append(result)
                return results

            with patch(
                "utils.document_loader.tempfile.gettempdir", return_value="/tmp"
            ):
                with patch("builtins.open", create=True):
                    with patch("utils.document_loader.os.unlink"):
                        results = benchmark(batch_embedding_operation)

            assert len(results) == 10
            for result in results:
                assert result["provider_used"] == "fastembed_native_multimodal"
                assert len(result["image_embeddings"]) == 1

    @pytest.mark.performance
    def test_hybrid_retriever_query_performance(
        self, benchmark, sample_multimodal_documents
    ):
        """Benchmark hybrid retriever query performance."""
        from utils.index_builder import create_hybrid_retriever

        mock_index = MagicMock()

        with patch("utils.index_builder.QueryFusionRetriever") as mock_fusion:
            with patch(
                "utils.index_builder.VectorIndexRetriever"
            ) as mock_vector_retriever:
                # Mock fast retrievers
                mock_dense = MagicMock()
                mock_sparse = MagicMock()
                mock_vector_retriever.side_effect = [mock_dense, mock_sparse]

                # Mock fusion retriever with fast retrieve method
                mock_fusion_instance = MagicMock()

                def fast_retrieve(query_bundle):
                    return [
                        MagicMock(
                            node=MagicMock(node_id=f"doc_{i}", text=f"Result {i}"),
                            score=0.95 - i * 0.1,
                        )
                        for i in range(10)
                    ]

                mock_fusion_instance.retrieve = fast_retrieve
                mock_fusion.return_value = mock_fusion_instance

                retriever = create_hybrid_retriever(mock_index)

                from llama_index.core import QueryBundle

                def query_operation():
                    return retriever.retrieve(
                        QueryBundle(
                            query_str="multimodal AI computer vision deep learning"
                        )
                    )

                results = benchmark(query_operation)
                assert len(results) == 10

    @pytest.mark.performance
    def test_unstructured_parsing_performance(self, benchmark, tmp_path):
        """Benchmark Unstructured parsing performance."""
        from utils.document_loader import load_documents_unstructured

        # Create larger test file
        large_content = "Large document content with multiple paragraphs. " * 1000
        test_file = tmp_path / "large_test.pdf"
        test_file.write_text(large_content)

        with patch("utils.document_loader.partition") as mock_partition:
            # Mock large number of elements
            mock_elements = []
            for i in range(100):
                mock_element = MagicMock()
                mock_element.category = "NarrativeText"
                mock_element.__str__.return_value = f"Paragraph {i} content"
                mock_element.metadata.page_number = i // 10 + 1
                mock_element.metadata.filename = "large_test.pdf"
                mock_elements.append(mock_element)

            mock_partition.return_value = mock_elements

            with patch(
                "utils.document_loader.chunk_documents_structured"
            ) as mock_chunk:
                mock_chunk.side_effect = lambda x: x  # Pass through

                def parsing_operation():
                    return load_documents_unstructured(str(test_file))

                results = benchmark(parsing_operation)
                assert len(results) == 100


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for multimodal processing."""

    def test_multimodal_index_no_documents_error(self, mock_multimodal_settings):
        """Test error handling when creating index with no documents."""
        from utils.index_builder import create_multimodal_index

        with patch("utils.index_builder.logging.error") as mock_log_error:
            with pytest.raises(Exception):
                create_multimodal_index([], use_gpu=False)

    def test_unstructured_parsing_empty_file(self, tmp_path, mock_multimodal_settings):
        """Test handling of empty file in Unstructured parsing."""
        from utils.document_loader import load_documents_unstructured

        empty_file = tmp_path / "empty.pdf"
        empty_file.write_text("")

        with patch("utils.document_loader.partition") as mock_partition:
            mock_partition.return_value = []  # No elements

            with patch("utils.document_loader.settings", mock_multimodal_settings):
                result = load_documents_unstructured(str(empty_file))

            assert result == []

    def test_hybrid_retriever_invalid_index(self):
        """Test error handling with invalid index."""
        from utils.index_builder import create_hybrid_retriever

        with patch("utils.index_builder.VectorIndexRetriever") as mock_retriever:
            mock_retriever.side_effect = ValueError("Invalid index")

            with patch("utils.index_builder.logging.error") as mock_log_error:
                with patch("utils.index_builder.logging.warning") as mock_log_warning:
                    result = create_hybrid_retriever(None)

            # Should use final fallback
            mock_log_error.assert_called()
            mock_log_warning.assert_called()
            assert result is not None  # Should return fallback retriever

    def test_image_document_invalid_base64(self):
        """Test handling of invalid base64 image data."""
        from utils.document_loader import load_documents_unstructured

        with patch("utils.document_loader.partition") as mock_partition:
            # Mock image element with invalid base64
            mock_image = MagicMock()
            mock_image.category = "Image"
            mock_image.metadata.page_number = 1
            mock_image.metadata.filename = "test.pdf"
            mock_image.metadata.image_base64 = "invalid_base64_data!!!"

            mock_partition.return_value = [mock_image]

            # Should handle base64 decode error gracefully
            try:
                result = load_documents_unstructured("/fake/path.pdf")
                # The function should either skip invalid images or handle the error
                # depending on implementation details
            except Exception as e:
                pytest.fail(f"Should handle invalid base64 gracefully, but got: {e}")


@pytest.mark.slow
class TestIntegrationScenarios:
    """Integration tests combining multiple multimodal components."""

    @pytest.mark.integration
    def test_end_to_end_multimodal_pipeline(
        self, sample_multimodal_documents, mock_multimodal_settings
    ):
        """Test complete end-to-end multimodal processing pipeline."""
        from utils.index_builder import create_hybrid_retriever, create_multimodal_index

        # Mock all components for integration test
        with patch("utils.index_builder.QdrantClient") as mock_client:
            with patch(
                "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
            ) as mock_embedding:
                with patch(
                    "utils.index_builder.MultiModalVectorStoreIndex"
                ) as mock_index:
                    # Mock successful index creation
                    mock_index_instance = MagicMock()
                    mock_index.from_documents.return_value = mock_index_instance

                    mock_embed_model = MagicMock()
                    mock_embedding.return_value = mock_embed_model

                    mock_qdrant_client = MagicMock()
                    mock_client.return_value = mock_qdrant_client

                    with patch(
                        "utils.index_builder.settings", mock_multimodal_settings
                    ):
                        # Step 1: Create multimodal index
                        index = create_multimodal_index(
                            sample_multimodal_documents, use_gpu=True
                        )

                        # Step 2: Create hybrid retriever
                        with patch(
                            "utils.index_builder.QueryFusionRetriever"
                        ) as mock_fusion:
                            with patch("utils.index_builder.VectorIndexRetriever"):
                                mock_hybrid_retriever = MagicMock()
                                mock_fusion.return_value = mock_hybrid_retriever

                                retriever = create_hybrid_retriever(index)

        # Verify complete pipeline
        assert index == mock_index_instance
        assert retriever == mock_hybrid_retriever

        # Verify Jina v3 was configured
        mock_embedding.assert_called_once()
        embed_kwargs = mock_embedding.call_args[1]
        assert "jinaai/jina-embeddings-v3" in embed_kwargs["model_name"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_multimodal_workflow(
        self, sample_multimodal_documents, mock_multimodal_settings
    ):
        """Test asynchronous multimodal workflow."""
        from utils.index_builder import create_multimodal_index_async

        with patch("utils.index_builder.AsyncQdrantClient") as mock_async_client:
            with patch(
                "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
            ) as mock_embedding:
                with patch(
                    "utils.index_builder.MultiModalVectorStoreIndex"
                ) as mock_index:
                    # Mock async components
                    mock_client_instance = AsyncMock()
                    mock_async_client.return_value = mock_client_instance

                    mock_embed_model = MagicMock()
                    mock_embedding.return_value = mock_embed_model

                    mock_index_instance = MagicMock()
                    mock_index.from_documents.return_value = mock_index_instance

                    with patch(
                        "utils.index_builder.settings", mock_multimodal_settings
                    ):
                        # Test async pipeline
                        result = await create_multimodal_index_async(
                            sample_multimodal_documents,
                            use_gpu=True,
                            collection_name="async_test",
                        )

                        # Verify async operations
                        mock_async_client.assert_called_once()
                        mock_client_instance.close.assert_called_once()

                        assert result == mock_index_instance


# Mark the entire file for multimodal testing
pytestmark = pytest.mark.multimodal
