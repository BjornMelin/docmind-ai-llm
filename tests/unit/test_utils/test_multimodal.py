"""Comprehensive unit tests for src.utils.multimodal utility functions.

Tests focus on multimodal CLIP integration, VRAM validation, cross-modal search,
and batch processing with proper mocking of external dependencies. All tests
are designed for fast execution (<0.05s each) with parametrization.

Coverage areas:
- CLIP image embedding generation
- VRAM usage validation and monitoring
- Cross-modal search functionality (text-to-image, image-to-image)
- End-to-end pipeline validation
- Image document creation
- Batch image processing

Mocked external dependencies:
- PyTorch CUDA operations
- CLIP embedding model operations
- PIL Image processing
- LlamaIndex multimodal components
- Numpy array operations
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from src.utils.multimodal import (
    EMBEDDING_DIMENSIONS,
    MAX_TEST_IMAGES,
    RANK_ADJUSTMENT,
    TEXT_TRUNCATION_LIMIT,
    batch_process_images,
    create_image_documents,
    cross_modal_search,
    generate_image_embeddings,
    validate_end_to_end_pipeline,
    validate_vram_usage,
)


@pytest.mark.unit
@pytest.mark.asyncio
class TestGenerateImageEmbeddings:
    """Test CLIP image embedding generation functionality."""

    async def test_generate_image_embeddings_success(self):
        """Test successful image embedding generation."""
        mock_clip = Mock()
        mock_image = Mock(spec=Image.Image)

        # Mock embedding as torch tensor
        mock_embedding = Mock()
        mock_embedding.cpu.return_value.numpy.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4]
        )
        mock_clip.get_image_embedding.return_value = mock_embedding

        with patch("asyncio.to_thread", return_value=mock_embedding):
            result = await generate_image_embeddings(mock_clip, mock_image)

            assert isinstance(result, np.ndarray)
            # Should be normalized
            norm = np.linalg.norm(result)
            assert abs(norm - 1.0) < 1e-6

    async def test_generate_image_embeddings_numpy_input(self):
        """Test embedding generation with numpy array input."""
        mock_clip = Mock()
        mock_image = Mock(spec=Image.Image)

        # Mock embedding as numpy array
        raw_embedding = np.array([3.0, 4.0])  # Will normalize to [0.6, 0.8]
        mock_clip.get_image_embedding.return_value = raw_embedding

        with patch("asyncio.to_thread", return_value=raw_embedding):
            result = await generate_image_embeddings(mock_clip, mock_image)

            expected = np.array([0.6, 0.8])
            np.testing.assert_array_almost_equal(result, expected)

    async def test_generate_image_embeddings_list_input(self):
        """Test embedding generation with list input."""
        mock_clip = Mock()
        mock_image = Mock(spec=Image.Image)

        # Mock embedding as list
        raw_embedding = [1.0, 0.0, 0.0]  # Already normalized
        mock_clip.get_image_embedding.return_value = raw_embedding

        with patch("asyncio.to_thread", return_value=raw_embedding):
            result = await generate_image_embeddings(mock_clip, mock_image)

            expected = np.array([1.0, 0.0, 0.0])
            np.testing.assert_array_almost_equal(result, expected)

    async def test_generate_image_embeddings_zero_norm(self):
        """Test embedding generation with zero vector (edge case)."""
        mock_clip = Mock()
        mock_image = Mock(spec=Image.Image)

        # Mock embedding as zero vector
        raw_embedding = np.array([0.0, 0.0, 0.0])
        mock_clip.get_image_embedding.return_value = raw_embedding

        with patch("asyncio.to_thread", return_value=raw_embedding):
            result = await generate_image_embeddings(mock_clip, mock_image)

            # Should return zero vector unchanged when norm is 0
            np.testing.assert_array_equal(result, raw_embedding)

    @pytest.mark.parametrize(
        "embedding_input,expected_shape",
        [
            (np.random.rand(512), (512,)),
            (np.random.rand(256), (256,)),
            ([0.5, 0.5, 0.5, 0.5], (4,)),
        ],
    )
    async def test_generate_image_embeddings_different_dimensions(
        self, embedding_input, expected_shape
    ):
        """Test embedding generation with different dimensions."""
        mock_clip = Mock()
        mock_image = Mock(spec=Image.Image)
        mock_clip.get_image_embedding.return_value = embedding_input

        with patch("asyncio.to_thread", return_value=embedding_input):
            result = await generate_image_embeddings(mock_clip, mock_image)

            assert result.shape == expected_shape


@pytest.mark.unit
class TestValidateVramUsage:
    """Test VRAM usage validation functionality."""

    @patch("torch.cuda.is_available")
    def test_validate_vram_usage_no_cuda(self, mock_cuda_available):
        """Test VRAM validation when CUDA is not available."""
        mock_cuda_available.return_value = False
        mock_clip = Mock()

        result = validate_vram_usage(mock_clip)

        assert result == 0.0

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.is_available")
    def test_validate_vram_usage_success(
        self, mock_cuda_available, mock_empty_cache, mock_memory_allocated
    ):
        """Test successful VRAM validation."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.side_effect = [1000000000, 1200000000]  # 1GB -> 1.2GB

        mock_clip = Mock()
        mock_clip.get_image_embedding.return_value = np.random.rand(512)

        result = validate_vram_usage(mock_clip)

        assert result > 0.0
        mock_empty_cache.assert_called_once()

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.is_available")
    def test_validate_vram_usage_with_images(
        self, mock_cuda_available, mock_empty_cache, mock_memory_allocated
    ):
        """Test VRAM validation with image processing."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.side_effect = [500000000, 800000000]  # 0.5GB -> 0.8GB

        mock_clip = Mock()
        mock_clip.get_image_embedding.return_value = np.random.rand(512)

        # Create mock images
        mock_images = [Mock(spec=Image.Image) for _ in range(5)]

        result = validate_vram_usage(mock_clip, mock_images)

        assert result > 0.0
        # Should process up to MAX_TEST_IMAGES
        assert mock_clip.get_image_embedding.call_count <= MAX_TEST_IMAGES

    @pytest.mark.parametrize(
        "exception_type,expected_result",
        [
            (RuntimeError("CUDA out of memory"), 0.0),
            (ValueError("Invalid image"), 0.0),
            (TypeError("Type error"), 0.0),
        ],
    )
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.is_available")
    def test_validate_vram_usage_error_handling(
        self,
        mock_cuda_available,
        mock_empty_cache,
        mock_memory_allocated,
        exception_type,
        expected_result,
    ):
        """Test VRAM validation error handling."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 1000000000

        mock_clip = Mock()
        mock_clip.get_image_embedding.side_effect = exception_type

        mock_images = [Mock(spec=Image.Image)]

        result = validate_vram_usage(mock_clip, mock_images)

        # Should still return baseline VRAM measurement
        assert isinstance(result, float)
        assert result >= 0.0

    @patch("torch.cuda.is_available")
    def test_validate_vram_usage_cuda_error_during_init(self, mock_cuda_available):
        """Test VRAM validation when CUDA operations fail during initialization."""
        mock_cuda_available.side_effect = RuntimeError("CUDA driver error")

        mock_clip = Mock()

        result = validate_vram_usage(mock_clip)

        assert result == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
class TestCrossModalSearch:
    """Test cross-modal search functionality."""

    async def test_cross_modal_search_text_to_image(self):
        """Test text-to-image cross-modal search."""
        mock_index = Mock()
        mock_query_engine = Mock()

        # Mock search response
        mock_response = Mock()
        mock_node = Mock()
        mock_node.score = 0.85
        mock_node.node.metadata = {"image_path": "/path/to/image.jpg"}
        mock_node.node.text = "A beautiful sunset over the ocean"
        mock_response.source_nodes = [mock_node]

        mock_query_engine.query.return_value = mock_response
        mock_index.as_query_engine.return_value = mock_query_engine

        with patch("asyncio.to_thread", return_value=mock_response):
            results = await cross_modal_search(
                mock_index, query="sunset", search_type="text_to_image", top_k=5
            )

        assert len(results) == 1
        assert results[0]["score"] == 0.85
        assert results[0]["image_path"] == "/path/to/image.jpg"
        assert results[0]["text"] == "A beautiful sunset over the ocean"
        assert results[0]["rank"] == 1

    async def test_cross_modal_search_image_to_image(self):
        """Test image-to-image cross-modal search."""
        mock_index = Mock()
        mock_retriever = Mock()
        mock_query_image = Mock(spec=Image.Image)

        # Mock search node
        mock_node = Mock()
        mock_node.score = 0.92
        mock_node.node.metadata = {"image_path": "/path/to/similar.jpg"}
        mock_node.node.text = "Similar image content"

        mock_retriever.retrieve.return_value = [mock_node]
        mock_index.as_retriever.return_value = mock_retriever
        mock_index.embed_model = Mock()

        # Mock embedding generation
        with (
            patch(
                "src.utils.multimodal.generate_image_embeddings",
                return_value=np.random.rand(512),
            ),
            patch("asyncio.to_thread", return_value=[mock_node]),
        ):
            results = await cross_modal_search(
                mock_index,
                query_image=mock_query_image,
                search_type="image_to_image",
                top_k=3,
            )

        assert len(results) == 1
        assert results[0]["similarity"] == 0.92
        assert results[0]["image_path"] == "/path/to/similar.jpg"
        assert results[0]["rank"] == 1

    async def test_cross_modal_search_text_truncation(self):
        """Test text truncation in search results."""
        mock_index = Mock()
        mock_query_engine = Mock()

        # Create long text that should be truncated
        long_text = "A" * (TEXT_TRUNCATION_LIMIT + 100)

        mock_response = Mock()
        mock_node = Mock()
        mock_node.score = 0.75
        mock_node.node.metadata = {"image_path": "/path/to/image.jpg"}
        mock_node.node.text = long_text
        mock_response.source_nodes = [mock_node]

        mock_query_engine.query.return_value = mock_response
        mock_index.as_query_engine.return_value = mock_query_engine

        with patch("asyncio.to_thread", return_value=mock_response):
            results = await cross_modal_search(
                mock_index, query="test", search_type="text_to_image"
            )

        assert len(results[0]["text"]) == TEXT_TRUNCATION_LIMIT

    async def test_cross_modal_search_no_results(self):
        """Test cross-modal search with no results."""
        mock_index = Mock()

        # Test with unsupported search type
        results = await cross_modal_search(
            mock_index, query="test", search_type="unsupported_type"
        )

        assert results == []

    @pytest.mark.parametrize(
        "search_type,query,query_image,expected_calls",
        [
            ("text_to_image", "sunset", None, "as_query_engine"),
            ("image_to_image", None, Mock(spec=Image.Image), "as_retriever"),
        ],
    )
    async def test_cross_modal_search_method_selection(
        self, search_type, query, query_image, expected_calls
    ):
        """Test that correct methods are called for different search types."""
        mock_index = Mock()

        # Setup appropriate mocks based on search type
        if search_type == "text_to_image":
            mock_query_engine = Mock()
            mock_response = Mock()
            mock_response.source_nodes = []
            mock_query_engine.query.return_value = mock_response
            mock_index.as_query_engine.return_value = mock_query_engine

            with patch("asyncio.to_thread", return_value=mock_response):
                await cross_modal_search(
                    mock_index, query=query, search_type=search_type
                )

            mock_index.as_query_engine.assert_called_once()

        elif search_type == "image_to_image":
            mock_retriever = Mock()
            mock_retriever.retrieve.return_value = []
            mock_index.as_retriever.return_value = mock_retriever
            mock_index.embed_model = Mock()

            with (
                patch(
                    "src.utils.multimodal.generate_image_embeddings",
                    return_value=np.random.rand(512),
                ),
                patch("asyncio.to_thread", return_value=[]),
            ):
                await cross_modal_search(
                    mock_index, query_image=query_image, search_type=search_type
                )

            mock_index.as_retriever.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
class TestValidateEndToEndPipeline:
    """Test end-to-end pipeline validation functionality."""

    async def test_validate_end_to_end_pipeline_success(self):
        """Test successful end-to-end pipeline validation."""
        query = "Test query"
        mock_query_image = Mock(spec=Image.Image)
        mock_clip = Mock()
        mock_property_graph = Mock()
        mock_llm = Mock()

        # Mock image embedding
        test_embedding = np.array([0.5, 0.5, 0.707, 0.0])  # Normalized

        with patch(
            "src.utils.multimodal.generate_image_embeddings",
            return_value=test_embedding,
        ):
            with patch("time.perf_counter", side_effect=[0.0, 2.5]):
                result = await validate_end_to_end_pipeline(
                    query, mock_query_image, mock_clip, mock_property_graph, mock_llm
                )

        # Verify result structure
        assert "visual_similarity" in result
        assert "entity_relationships" in result
        assert "final_response" in result
        assert "pipeline_time" in result

        # Verify visual similarity data
        assert result["visual_similarity"]["embedding_dim"] == 4
        assert abs(result["visual_similarity"]["norm"] - 1.0) < 1e-4

        # Verify timing
        assert result["pipeline_time"] == 2.5

    async def test_validate_end_to_end_pipeline_entity_extraction(self):
        """Test entity extraction in end-to-end pipeline."""
        query = "LlamaIndex BGE-M3 integration"
        mock_query_image = Mock(spec=Image.Image)
        mock_clip = Mock()
        mock_property_graph = Mock()
        mock_llm = Mock()

        test_embedding = np.random.rand(10)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)

        with patch(
            "src.utils.multimodal.generate_image_embeddings",
            return_value=test_embedding,
        ):
            with patch("time.perf_counter", side_effect=[0.0, 1.0]):
                result = await validate_end_to_end_pipeline(
                    query, mock_query_image, mock_clip, mock_property_graph, mock_llm
                )

        # Verify entity relationships
        entities = result["entity_relationships"]["entities_found"]
        assert "LlamaIndex" in entities
        assert "BGE-M3" in entities
        assert (
            result["entity_relationships"]["relationship_count"]
            == len(entities) - RANK_ADJUSTMENT
        )

    async def test_validate_end_to_end_pipeline_response_generation(self):
        """Test response generation in end-to-end pipeline."""
        query = "multimodal search"
        mock_query_image = Mock(spec=Image.Image)
        mock_clip = Mock()
        mock_property_graph = Mock()
        mock_llm = Mock()

        with patch(
            "src.utils.multimodal.generate_image_embeddings",
            return_value=np.array([1.0]),
        ):
            with patch("time.perf_counter", side_effect=[0.0, 0.5]):
                result = await validate_end_to_end_pipeline(
                    query, mock_query_image, mock_clip, mock_property_graph, mock_llm
                )

        # Verify response contains query
        assert query in result["final_response"]
        assert "visual similarity" in result["final_response"]
        assert "entity relationships" in result["final_response"]


@pytest.mark.unit
class TestCreateImageDocuments:
    """Test image document creation functionality."""

    def test_create_image_documents_success(self):
        """Test successful image document creation."""
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.png"]
        metadata = {"source": "test_dataset", "batch_id": 123}

        with patch("src.utils.multimodal.ImageDocument") as mock_image_doc:
            mock_doc1 = Mock()
            mock_doc2 = Mock()
            mock_image_doc.side_effect = [mock_doc1, mock_doc2]

            result = create_image_documents(image_paths, metadata)

            assert len(result) == 2
            assert result == [mock_doc1, mock_doc2]

            # Verify ImageDocument calls
            assert mock_image_doc.call_count == 2
            mock_image_doc.assert_any_call(
                image_path="/path/to/image1.jpg", metadata=metadata
            )
            mock_image_doc.assert_any_call(
                image_path="/path/to/image2.png", metadata=metadata
            )

    def test_create_image_documents_default_metadata(self):
        """Test image document creation with default metadata."""
        image_paths = ["/path/to/image.jpg"]

        with patch("src.utils.multimodal.ImageDocument") as mock_image_doc:
            mock_doc = Mock()
            mock_image_doc.return_value = mock_doc

            result = create_image_documents(image_paths)

            assert len(result) == 1
            mock_image_doc.assert_called_once_with(
                image_path="/path/to/image.jpg", metadata={"source": "multimodal"}
            )

    def test_create_image_documents_with_errors(self):
        """Test image document creation with some errors."""
        image_paths = ["/valid/path.jpg", "/invalid/path.jpg", "/another/valid.png"]

        with patch("src.utils.multimodal.ImageDocument") as mock_image_doc:
            mock_doc1 = Mock()
            mock_doc3 = Mock()
            mock_image_doc.side_effect = [
                mock_doc1,  # First path succeeds
                OSError("File not found"),  # Second path fails
                mock_doc3,  # Third path succeeds
            ]

            with patch("src.utils.multimodal.logger") as mock_logger:
                result = create_image_documents(image_paths)

                # Should return only successful documents
                assert len(result) == 2
                assert result == [mock_doc1, mock_doc3]

                # Should log error for failed path
                mock_logger.error.assert_called_once()

    @pytest.mark.parametrize(
        "exception_type",
        [OSError("File error"), ValueError("Invalid path")],
    )
    def test_create_image_documents_error_types(self, exception_type):
        """Test image document creation with different error types."""
        image_paths = ["/error/path.jpg"]

        with patch("src.utils.multimodal.ImageDocument", side_effect=exception_type):
            with patch("src.utils.multimodal.logger") as mock_logger:
                result = create_image_documents(image_paths)

                assert result == []
                mock_logger.error.assert_called_once()


@pytest.mark.unit
class TestBatchProcessImages:
    """Test batch image processing functionality."""

    def test_batch_process_images_success(self):
        """Test successful batch image processing."""
        mock_clip = Mock()
        mock_images = [Mock(spec=Image.Image) for _ in range(3)]

        # Mock embeddings for each image
        embeddings = [np.random.rand(EMBEDDING_DIMENSIONS) for _ in range(3)]
        mock_clip.get_image_embedding.side_effect = embeddings

        result = batch_process_images(mock_clip, mock_images, batch_size=2)

        assert result.shape == (3, EMBEDDING_DIMENSIONS)
        assert mock_clip.get_image_embedding.call_count == 3

    def test_batch_process_images_with_batch_size(self):
        """Test batch processing with specific batch size."""
        mock_clip = Mock()
        mock_images = [Mock(spec=Image.Image) for _ in range(5)]

        embeddings = [np.random.rand(EMBEDDING_DIMENSIONS) for _ in range(5)]
        mock_clip.get_image_embedding.side_effect = embeddings

        result = batch_process_images(mock_clip, mock_images, batch_size=2)

        # Should process all images despite batching
        assert result.shape == (5, EMBEDDING_DIMENSIONS)
        assert mock_clip.get_image_embedding.call_count == 5

    def test_batch_process_images_with_errors(self):
        """Test batch processing with some images causing errors."""
        mock_clip = Mock()
        mock_images = [Mock(spec=Image.Image) for _ in range(3)]

        # First image succeeds, second fails, third succeeds
        mock_clip.get_image_embedding.side_effect = [
            np.random.rand(EMBEDDING_DIMENSIONS),
            RuntimeError("Processing error"),
            np.random.rand(EMBEDDING_DIMENSIONS),
        ]

        with patch("src.utils.multimodal.logger") as mock_logger:
            result = batch_process_images(mock_clip, mock_images)

            assert result.shape == (3, EMBEDDING_DIMENSIONS)
            # Second embedding should be zeros due to error
            np.testing.assert_array_equal(result[1], np.zeros(EMBEDDING_DIMENSIONS))

            # Should log error
            mock_logger.error.assert_called_once()

    def test_batch_process_images_torch_tensor_conversion(self):
        """Test batch processing with torch tensor embeddings."""
        mock_clip = Mock()
        mock_images = [Mock(spec=Image.Image)]

        # Mock torch tensor that needs .cpu().numpy()
        mock_tensor = Mock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.rand(
            EMBEDDING_DIMENSIONS
        )
        mock_clip.get_image_embedding.return_value = mock_tensor

        result = batch_process_images(mock_clip, mock_images)

        assert result.shape == (1, EMBEDDING_DIMENSIONS)
        mock_tensor.cpu.assert_called_once()

    @pytest.mark.parametrize(
        "error_type,expected_zero_embeddings",
        [
            (RuntimeError("CUDA error"), 1),
            (ValueError("Invalid image format"), 1),
        ],
    )
    def test_batch_process_images_error_handling(
        self, error_type, expected_zero_embeddings
    ):
        """Test error handling in batch processing."""
        mock_clip = Mock()
        mock_images = [Mock(spec=Image.Image)]

        mock_clip.get_image_embedding.side_effect = error_type

        with patch("src.utils.multimodal.logger"):
            result = batch_process_images(mock_clip, mock_images)

            # Should have zero embeddings for failed images
            zero_count = np.sum(np.all(result == 0, axis=1))
            assert zero_count == expected_zero_embeddings


@pytest.mark.unit
class TestMultimodalUtilsConstants:
    """Test multimodal utility constants and configuration."""

    def test_constants_values(self):
        """Test that constants have expected values."""
        assert MAX_TEST_IMAGES == 10
        assert TEXT_TRUNCATION_LIMIT == 200
        assert RANK_ADJUSTMENT == 1
        assert EMBEDDING_DIMENSIONS == 512

    def test_constants_types(self):
        """Test that constants have correct types."""
        assert isinstance(MAX_TEST_IMAGES, int)
        assert isinstance(TEXT_TRUNCATION_LIMIT, int)
        assert isinstance(RANK_ADJUSTMENT, int)
        assert isinstance(EMBEDDING_DIMENSIONS, int)

    def test_constants_ranges(self):
        """Test that constants are in reasonable ranges."""
        assert MAX_TEST_IMAGES > 0
        assert TEXT_TRUNCATION_LIMIT > 0
        assert RANK_ADJUSTMENT > 0
        assert EMBEDDING_DIMENSIONS > 0


@pytest.mark.unit
class TestMultimodalUtilsEdgeCases:
    """Test edge cases and error scenarios for multimodal utilities."""

    def test_create_image_documents_empty_paths(self):
        """Test create_image_documents with empty path list."""
        result = create_image_documents([])
        assert result == []

    def test_batch_process_images_empty_images(self):
        """Test batch_process_images with empty image list."""
        mock_clip = Mock()
        result = batch_process_images(mock_clip, [])

        assert result.shape == (0,)  # Empty array has shape (0,)
        mock_clip.get_image_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_cross_modal_search_empty_query(self):
        """Test cross-modal search with empty query."""
        mock_index = Mock()

        results = await cross_modal_search(
            mock_index, query="", search_type="text_to_image"
        )

        # Should handle gracefully (implementation dependent)
        assert isinstance(results, list)

    def test_validate_vram_usage_none_images(self):
        """Test VRAM validation with None images parameter."""
        mock_clip = Mock()

        with patch("torch.cuda.is_available", return_value=False):
            result = validate_vram_usage(mock_clip, None)

            assert result == 0.0

    @pytest.mark.asyncio
    async def test_generate_image_embeddings_edge_cases(self):
        """Test image embedding generation with edge cases."""
        mock_clip = Mock()

        # Test with None image (should not crash)
        try:
            await generate_image_embeddings(mock_clip, None)
        except (AttributeError, TypeError):
            # Expected to fail with None image
            pass

    def test_batch_process_images_single_image(self):
        """Test batch processing with single image."""
        mock_clip = Mock()
        mock_image = Mock(spec=Image.Image)

        embedding = np.random.rand(EMBEDDING_DIMENSIONS)
        mock_clip.get_image_embedding.return_value = embedding

        result = batch_process_images(mock_clip, [mock_image])

        assert result.shape == (1, EMBEDDING_DIMENSIONS)
        np.testing.assert_array_equal(result[0], embedding)
