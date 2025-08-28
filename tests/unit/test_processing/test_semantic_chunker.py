"""Comprehensive test suite for SemanticChunker class.

This test suite covers the semantic chunker that uses direct chunk_by_title
integration from unstructured.io, focusing on boundary detection,
chunking parameters, and async processing.
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.models.processing import DocumentElement
from src.processing.chunking.models import (
    BoundaryDetection,
    ChunkingError,
    ChunkingParameters,
    ChunkingResult,
    SemanticChunk,
)
from src.processing.chunking.unstructured_chunker import (
    ElementAdapter,
    ElementMetadata,
    SemanticChunker,
)


@pytest.fixture
def mock_settings():
    """Mock DocMind settings for testing."""
    mock_settings = Mock()
    mock_settings.processing.chunk_size = 1500
    mock_settings.processing.new_after_n_chars = 1200
    mock_settings.processing.combine_text_under_n_chars = 500
    mock_settings.processing.multipage_sections = True
    return mock_settings


@pytest.fixture
def sample_chunking_parameters():
    """Create sample chunking parameters for testing."""
    return ChunkingParameters(
        max_characters=1500,
        new_after_n_chars=1200,
        combine_text_under_n_chars=500,
        multipage_sections=True,
        boundary_detection=BoundaryDetection.TITLE_BASED,
    )


@pytest.fixture
def sample_document_elements():
    """Create sample DocumentElement objects for testing."""
    return [
        DocumentElement(
            text="Introduction",
            category="Title",
            metadata={"page_number": 1, "element_id": "title_1"},
        ),
        DocumentElement(
            text="This is the introduction paragraph with detailed information about the topic.",
            category="NarrativeText",
            metadata={"page_number": 1, "element_id": "text_1"},
        ),
        DocumentElement(
            text="Section 1: Overview",
            category="Title",
            metadata={"page_number": 1, "element_id": "title_2"},
        ),
        DocumentElement(
            text="This section provides an overview of the main concepts and principles.",
            category="NarrativeText",
            metadata={"page_number": 1, "element_id": "text_2"},
        ),
        DocumentElement(
            text="Additional content that expands on the overview with more detailed explanations.",
            category="NarrativeText",
            metadata={"page_number": 1, "element_id": "text_3"},
        ),
    ]


@pytest.fixture
def mock_unstructured_chunk():
    """Mock unstructured chunk returned by chunk_by_title."""
    chunk = Mock()
    chunk.text = "Introduction\n\nThis is the introduction paragraph with detailed information about the topic."
    chunk.category = "CompositeElement"
    chunk.metadata = Mock()
    chunk.metadata.page_number = 1
    chunk.metadata.element_id = "chunk_1"
    chunk.metadata.section_title = "Introduction"
    chunk.metadata.filename = "test.pdf"
    return chunk


@pytest.mark.unit
class TestElementMetadata:
    """Test ElementMetadata dataclass."""

    def test_element_metadata_initialization(self):
        """Test ElementMetadata initialization with defaults."""
        metadata = ElementMetadata()
        assert metadata.page_number == 1
        assert metadata.element_id is None
        assert metadata.filename is None

    def test_element_metadata_with_values(self):
        """Test ElementMetadata initialization with values."""
        metadata = ElementMetadata(
            page_number=2,
            element_id="elem_123",
            filename="test.pdf",
            coordinates=(10, 20, 30, 40),
        )
        assert metadata.page_number == 2
        assert metadata.element_id == "elem_123"
        assert metadata.filename == "test.pdf"
        assert metadata.coordinates == (10, 20, 30, 40)


@pytest.mark.unit
class TestElementAdapter:
    """Test ElementAdapter dataclass."""

    def test_element_adapter_initialization(self):
        """Test ElementAdapter initialization with defaults."""
        adapter = ElementAdapter()
        assert adapter.text == ""
        assert adapter.category == "NarrativeText"
        assert isinstance(adapter.metadata, ElementMetadata)

    def test_element_adapter_with_values(self):
        """Test ElementAdapter initialization with values."""
        metadata = ElementMetadata(page_number=2, element_id="test_id")
        adapter = ElementAdapter(
            text="Test content", category="Title", metadata=metadata
        )
        assert adapter.text == "Test content"
        assert adapter.category == "Title"
        assert adapter.metadata == metadata

    def test_element_adapter_post_init(self):
        """Test ElementAdapter post-init creates metadata if None."""
        adapter = ElementAdapter(text="test", metadata=None)
        assert isinstance(adapter.metadata, ElementMetadata)


@pytest.mark.unit
class TestSemanticChunker:
    """Test SemanticChunker functionality."""

    def test_initialization_with_settings(self, mock_settings):
        """Test chunker initialization with provided settings."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)
            assert chunker.settings == mock_settings
            assert chunker.default_parameters.max_characters == 1500
            assert chunker.default_parameters.new_after_n_chars == 1200

    def test_initialization_without_settings(self):
        """Test chunker initialization without settings uses defaults."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings"
        ) as mock_default:
            mock_default.processing.chunk_size = 1000
            mock_default.processing.new_after_n_chars = 800
            mock_default.processing.combine_text_under_n_chars = 300
            mock_default.processing.multipage_sections = True

            chunker = SemanticChunker()
            assert chunker.settings == mock_default
            assert chunker.default_parameters.max_characters == 1000

    def test_convert_document_elements_to_unstructured(
        self, mock_settings, sample_document_elements
    ):
        """Test conversion of DocumentElements to unstructured format."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)

            converted = chunker._convert_document_elements_to_unstructured(
                sample_document_elements
            )

            assert len(converted) == len(sample_document_elements)

            # Check first element (Title)
            first_element = converted[0]
            assert isinstance(first_element, ElementAdapter)
            assert first_element.text == "Introduction"
            assert first_element.category == "Title"
            assert first_element.metadata.page_number == 1

    def test_convert_document_elements_minimal_metadata(self, mock_settings):
        """Test conversion with minimal metadata creates defaults."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)

            # Element without metadata
            elements = [DocumentElement(text="Test text", category="Text")]
            converted = chunker._convert_document_elements_to_unstructured(elements)

            assert len(converted) == 1
            adapter = converted[0]
            assert adapter.metadata.page_number == 1
            assert adapter.metadata.element_id == "elem_0"
            assert adapter.metadata.filename == "document"

    def test_calculate_boundary_accuracy_title_based(
        self, mock_settings, sample_chunking_parameters
    ):
        """Test boundary accuracy calculation for title-based detection."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)

            # Mock chunks with section titles
            mock_chunks = [Mock(), Mock()]
            for i, chunk in enumerate(mock_chunks):
                chunk.metadata = Mock()
                chunk.metadata.section_title = f"Section {i + 1}"

            # Mock original elements with titles
            original_elements = [Mock(), Mock(), Mock()]
            original_elements[0].category = "Title"
            original_elements[1].category = "NarrativeText"
            original_elements[2].category = "Title"

            accuracy = chunker._calculate_boundary_accuracy(
                mock_chunks, original_elements, sample_chunking_parameters
            )

            assert isinstance(accuracy, float)
            assert 0.0 <= accuracy <= 1.0

    def test_calculate_boundary_accuracy_no_titles(
        self, mock_settings, sample_chunking_parameters
    ):
        """Test boundary accuracy calculation with no title elements."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)

            # Mock chunks without titles
            mock_chunks = [Mock(), Mock()]
            for chunk in mock_chunks:
                chunk.text = "A" * 1000  # 1000 chars
                chunk.metadata = Mock()
                chunk.metadata.section_title = None

            # Mock original elements without titles
            original_elements = [Mock(), Mock()]
            for element in original_elements:
                element.category = "NarrativeText"

            accuracy = chunker._calculate_boundary_accuracy(
                mock_chunks, original_elements, sample_chunking_parameters
            )

            assert isinstance(accuracy, float)
            assert 0.0 <= accuracy <= 1.0

    def test_calculate_boundary_accuracy_empty_inputs(
        self, mock_settings, sample_chunking_parameters
    ):
        """Test boundary accuracy calculation with empty inputs."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)

            accuracy = chunker._calculate_boundary_accuracy(
                [], [], sample_chunking_parameters
            )
            assert accuracy == 0.0

    def test_convert_unstructured_chunks_to_semantic(
        self, mock_settings, sample_chunking_parameters, mock_unstructured_chunk
    ):
        """Test conversion of unstructured chunks to SemanticChunk objects."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)

            chunks = chunker._convert_unstructured_chunks_to_semantic(
                [mock_unstructured_chunk], sample_chunking_parameters
            )

            assert len(chunks) == 1
            chunk = chunks[0]
            assert isinstance(chunk, SemanticChunk)
            assert chunk.text.startswith("Introduction")
            assert chunk.category == "CompositeElement"
            assert chunk.section_title == "Introduction"
            assert chunk.chunk_index == 0
            assert "boundary_type" in chunk.semantic_boundaries

    def test_chunk_by_title_sync_success(self, mock_settings):
        """Test synchronous chunking with chunk_by_title."""
        with (
            patch(
                "src.processing.chunking.unstructured_chunker.settings", mock_settings
            ),
            patch(
                "src.processing.chunking.unstructured_chunker.chunk_by_title"
            ) as mock_chunk_by_title,
        ):
            chunker = SemanticChunker(mock_settings)

            # Mock chunk_by_title return
            mock_chunks = [Mock(), Mock()]
            mock_chunk_by_title.return_value = mock_chunks

            elements = [Mock()]
            config = {"max_characters": 1500}

            result = chunker._chunk_by_title_sync(elements, config)

            assert result == mock_chunks
            mock_chunk_by_title.assert_called_once_with(elements=elements, **config)

    def test_chunk_by_title_sync_error(self, mock_settings):
        """Test error handling in synchronous chunking."""
        with (
            patch(
                "src.processing.chunking.unstructured_chunker.settings", mock_settings
            ),
            patch(
                "src.processing.chunking.unstructured_chunker.chunk_by_title"
            ) as mock_chunk_by_title,
        ):
            chunker = SemanticChunker(mock_settings)
            mock_chunk_by_title.side_effect = ValueError("Chunking failed")

            with pytest.raises(ChunkingError) as excinfo:
                chunker._chunk_by_title_sync([Mock()], {})

            assert "chunk_by_title failed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_chunk_elements_async_success(
        self, mock_settings, sample_document_elements, mock_unstructured_chunk
    ):
        """Test successful async chunking of elements."""
        with (
            patch(
                "src.processing.chunking.unstructured_chunker.settings", mock_settings
            ),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            chunker = SemanticChunker(mock_settings)

            # Mock the sync chunking method
            mock_to_thread.return_value = [mock_unstructured_chunk]

            result = await chunker.chunk_elements_async(sample_document_elements)

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) == 1
            assert result.total_elements == len(sample_document_elements)
            assert result.processing_time > 0
            assert 0.0 <= result.boundary_accuracy <= 1.0

    @pytest.mark.asyncio
    async def test_chunk_elements_async_with_custom_parameters(
        self, mock_settings, sample_document_elements, mock_unstructured_chunk
    ):
        """Test async chunking with custom parameters."""
        with (
            patch(
                "src.processing.chunking.unstructured_chunker.settings", mock_settings
            ),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            chunker = SemanticChunker(mock_settings)
            mock_to_thread.return_value = [mock_unstructured_chunk]

            custom_params = ChunkingParameters(
                max_characters=2000,
                new_after_n_chars=1500,
                combine_text_under_n_chars=600,
                boundary_detection=BoundaryDetection.HYBRID,
            )

            result = await chunker.chunk_elements_async(
                sample_document_elements, custom_params
            )

            assert result.parameters == custom_params
            assert result.parameters.max_characters == 2000
            assert result.metadata["boundary_strategy"] == "hybrid"

    @pytest.mark.asyncio
    async def test_chunk_elements_async_empty_elements(self, mock_settings):
        """Test async chunking with empty elements list."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)

            result = await chunker.chunk_elements_async([])

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) == 0
            assert result.total_elements == 0
            assert result.boundary_accuracy == 0.0
            assert "warning" in result.metadata

    @pytest.mark.asyncio
    async def test_chunk_elements_async_error_handling(
        self, mock_settings, sample_document_elements
    ):
        """Test error handling in async chunking."""
        with (
            patch(
                "src.processing.chunking.unstructured_chunker.settings", mock_settings
            ),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            chunker = SemanticChunker(mock_settings)
            mock_to_thread.side_effect = ValueError("Async processing failed")

            with pytest.raises(ChunkingError) as excinfo:
                await chunker.chunk_elements_async(sample_document_elements)

            assert "Semantic chunking failed" in str(excinfo.value)

    def test_optimize_parameters_with_target_count(
        self, mock_settings, sample_document_elements
    ):
        """Test parameter optimization with target chunk count."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)

            optimized = chunker.optimize_parameters(
                sample_document_elements, target_chunk_count=3
            )

            assert isinstance(optimized, ChunkingParameters)
            assert optimized.max_characters >= 500
            assert optimized.max_characters <= 3000
            assert optimized.new_after_n_chars < optimized.max_characters
            assert optimized.combine_text_under_n_chars < optimized.new_after_n_chars

    def test_optimize_parameters_without_target(
        self, mock_settings, sample_document_elements
    ):
        """Test parameter optimization without target count."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)

            optimized = chunker.optimize_parameters(sample_document_elements)

            assert isinstance(optimized, ChunkingParameters)
            # Should return default parameters when no target is specified
            assert optimized.max_characters == chunker.default_parameters.max_characters

    def test_optimize_parameters_empty_elements(self, mock_settings):
        """Test parameter optimization with empty elements."""
        with patch(
            "src.processing.chunking.unstructured_chunker.settings", mock_settings
        ):
            chunker = SemanticChunker(mock_settings)

            optimized = chunker.optimize_parameters([])

            assert optimized == chunker.default_parameters


@pytest.mark.unit
class TestChunkingModels:
    """Test chunking-related Pydantic models."""

    def test_chunking_parameters_validation(self):
        """Test ChunkingParameters validation."""
        # Valid parameters
        params = ChunkingParameters(
            max_characters=1500, new_after_n_chars=1200, combine_text_under_n_chars=500
        )
        assert params.max_characters == 1500
        assert params.boundary_detection == BoundaryDetection.TITLE_BASED

    def test_chunking_parameters_validation_errors(self):
        """Test ChunkingParameters validation errors."""
        # Test minimum values
        with pytest.raises(ValueError):
            ChunkingParameters(max_characters=50)  # Below minimum

        with pytest.raises(ValueError):
            ChunkingParameters(max_characters=15000)  # Above maximum

    def test_semantic_chunk_creation(self):
        """Test SemanticChunk model creation."""
        chunk = SemanticChunk(
            text="Test chunk content",
            category="CompositeElement",
            chunk_index=0,
            section_title="Test Section",
            semantic_boundaries={"boundary_type": "title"},
        )

        assert chunk.text == "Test chunk content"
        assert chunk.chunk_index == 0
        assert chunk.section_title == "Test Section"
        assert "boundary_type" in chunk.semantic_boundaries

    def test_chunking_result_creation(self, sample_chunking_parameters):
        """Test ChunkingResult model creation."""
        chunks = [
            SemanticChunk(text="Chunk 1", chunk_index=0),
            SemanticChunk(text="Chunk 2", chunk_index=1),
        ]

        result = ChunkingResult(
            chunks=chunks,
            total_elements=5,
            boundary_accuracy=0.85,
            processing_time=1.5,
            parameters=sample_chunking_parameters,
        )

        assert len(result.chunks) == 2
        assert result.total_elements == 5
        assert result.boundary_accuracy == 0.85
        assert result.processing_time == 1.5


@pytest.mark.integration
class TestSemanticChunkerIntegration:
    """Integration tests for SemanticChunker with realistic data."""

    @pytest.mark.asyncio
    async def test_chunking_workflow_integration(self, tmp_path):
        """Integration test for complete chunking workflow."""
        # Create mock elements similar to real document processing
        elements = []
        for i in range(10):
            if i % 3 == 0:
                # Title elements
                elements.append(
                    DocumentElement(
                        text=f"Section {i // 3 + 1}: Title",
                        category="Title",
                        metadata={
                            "page_number": i // 3 + 1,
                            "element_id": f"title_{i}",
                        },
                    )
                )
            else:
                # Content elements
                content = (
                    f"This is paragraph {i} with substantial content that provides detailed information. "
                    * 3
                )
                elements.append(
                    DocumentElement(
                        text=content,
                        category="NarrativeText",
                        metadata={"page_number": i // 3 + 1, "element_id": f"text_{i}"},
                    )
                )

        with (
            patch(
                "src.processing.chunking.unstructured_chunker.settings"
            ) as mock_settings,
            patch(
                "src.processing.chunking.unstructured_chunker.chunk_by_title"
            ) as mock_chunk_by_title,
        ):
            mock_settings.processing.chunk_size = 1500
            mock_settings.processing.new_after_n_chars = 1200
            mock_settings.processing.combine_text_under_n_chars = 500
            mock_settings.processing.multipage_sections = True

            # Mock chunk_by_title to return combined chunks
            mock_chunks = []
            for i in range(3):
                chunk = Mock()
                chunk.text = f"Combined chunk {i} with multiple elements and substantial content."
                chunk.category = "CompositeElement"
                chunk.metadata = Mock()
                chunk.metadata.section_title = f"Section {i + 1}"
                chunk.metadata.page_number = i + 1
                mock_chunks.append(chunk)

            mock_chunk_by_title.return_value = mock_chunks

            chunker = SemanticChunker()
            result = await chunker.chunk_elements_async(elements)

            assert isinstance(result, ChunkingResult)
            assert len(result.chunks) == 3
            assert result.total_elements == len(elements)
            assert result.boundary_accuracy > 0
            assert result.processing_time > 0
            assert all(isinstance(chunk, SemanticChunk) for chunk in result.chunks)

    def test_boundary_detection_strategies(self):
        """Test different boundary detection strategies."""
        strategies = [
            BoundaryDetection.TITLE_BASED,
            BoundaryDetection.CONTENT_BASED,
            BoundaryDetection.HYBRID,
        ]

        for strategy in strategies:
            params = ChunkingParameters(boundary_detection=strategy)
            assert params.boundary_detection == strategy
            assert params.boundary_detection.value in ["title", "content", "hybrid"]

    @pytest.mark.asyncio
    async def test_performance_with_large_document(self):
        """Test chunking performance with large number of elements."""
        # Create 100 elements
        elements = []
        for i in range(100):
            if i % 10 == 0:
                elements.append(
                    DocumentElement(
                        text=f"Chapter {i // 10 + 1}",
                        category="Title",
                        metadata={"page_number": i // 10 + 1},
                    )
                )
            else:
                elements.append(
                    DocumentElement(
                        text=f"Paragraph content {i} " * 50,  # ~800 chars
                        category="NarrativeText",
                        metadata={"page_number": i // 10 + 1},
                    )
                )

        with (
            patch(
                "src.processing.chunking.unstructured_chunker.settings"
            ) as mock_settings,
            patch(
                "src.processing.chunking.unstructured_chunker.chunk_by_title"
            ) as mock_chunk_by_title,
        ):
            mock_settings.processing.chunk_size = 2000
            mock_settings.processing.new_after_n_chars = 1500
            mock_settings.processing.combine_text_under_n_chars = 800
            mock_settings.processing.multipage_sections = True

            # Mock efficient chunking
            mock_chunks = [
                Mock() for _ in range(15)
            ]  # Simulate 15 chunks from 100 elements
            for i, chunk in enumerate(mock_chunks):
                chunk.text = f"Combined content for chunk {i}"
                chunk.category = "CompositeElement"
                chunk.metadata = Mock()
                chunk.metadata.section_title = (
                    f"Section {i // 3 + 1}" if i % 3 == 0 else None
                )

            mock_chunk_by_title.return_value = mock_chunks

            chunker = SemanticChunker()
            start_time = time.time()
            result = await chunker.chunk_elements_async(elements)
            processing_time = time.time() - start_time

            # Performance assertions
            assert processing_time < 5.0  # Should complete in reasonable time
            assert len(result.chunks) > 0
            assert result.total_elements == 100
            assert result.processing_time > 0
