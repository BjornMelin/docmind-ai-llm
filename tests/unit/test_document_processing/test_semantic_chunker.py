"""Unit tests for SemanticChunker (REQ-0024-v2).

Tests semantic chunking with chunk_by_title functionality, intelligent boundary
detection,
configurable parameters, and multipage section handling.

These are FAILING tests that will pass once the implementation is complete.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest

# These imports will fail until implementation is complete - this is expected
try:
    from src.core.document_processing.direct_unstructured_processor import (
        DocumentElement,
    )
    from src.core.document_processing.semantic_chunker import (
        BoundaryDetection,
        ChunkingParameters,
        ChunkingResult,
        SemanticChunk,
        SemanticChunker,
    )
except ImportError:
    # Create placeholder classes for failing tests
    class SemanticChunker:
        """Placeholder SemanticChunker class for failing tests."""

        pass

    class ChunkingParameters:
        """Placeholder ChunkingParameters class for failing tests."""

        pass

    class SemanticChunk:
        """Placeholder SemanticChunk class for failing tests."""

        pass

    class ChunkingResult:
        """Placeholder ChunkingResult class for failing tests."""

        pass

    class BoundaryDetection:
        """Placeholder BoundaryDetection class for failing tests."""

        TITLE_BASED = "title"
        CONTENT_BASED = "content"
        HYBRID = "hybrid"

    class DocumentElement:
        """Placeholder DocumentElement class for failing tests."""

        pass


@pytest.fixture
def mock_unstructured_chunk_by_title():
    """Mock unstructured.chunking.title.chunk_by_title function."""
    with patch("unstructured.chunking.title.chunk_by_title") as mock_chunk:
        # Mock chunked elements
        mock_chunks = [
            Mock(
                text=(
                    "This is the first semantic chunk containing title and "
                    "related content."
                ),
                category="CompositeElement",
                metadata=Mock(
                    page_number=1,
                    coordinates=Mock(points=[(0, 0), (400, 100)]),
                    parent_id=None,
                    element_id="chunk_1",
                    filename="test.pdf",
                    section_title="Introduction",
                    chunk_index=0,
                ),
            ),
            Mock(
                text=(
                    "This is the second chunk with methodology and analysis "
                    "content that spans multiple paragraphs."
                ),
                category="CompositeElement",
                metadata=Mock(
                    page_number=1,
                    coordinates=Mock(points=[(0, 110), (400, 250)]),
                    parent_id=None,
                    element_id="chunk_2",
                    filename="test.pdf",
                    section_title="Methodology",
                    chunk_index=1,
                ),
            ),
            Mock(
                text=(
                    "Final chunk containing conclusion and references with "
                    "proper semantic boundaries maintained."
                ),
                category="CompositeElement",
                metadata=Mock(
                    page_number=2,
                    coordinates=Mock(points=[(0, 0), (400, 150)]),
                    parent_id=None,
                    element_id="chunk_3",
                    filename="test.pdf",
                    section_title="Conclusion",
                    chunk_index=2,
                ),
            ),
        ]
        mock_chunk.return_value = mock_chunks
        yield mock_chunk


@pytest.fixture
def mock_settings():
    """Mock DocMind settings for chunking."""
    settings = Mock()
    settings.chunk_size = 1500
    settings.chunk_overlap = 200
    settings.new_after_n_chars = 1200
    settings.combine_text_under_n_chars = 500
    settings.multipage_sections = True
    settings.enable_semantic_boundary_detection = True
    return settings


@pytest.fixture
def sample_document_elements():
    """Sample document elements for chunking tests."""
    return [
        Mock(
            text="Introduction to Machine Learning",
            category="Title",
            metadata=Mock(
                page_number=1,
                coordinates=Mock(points=[(0, 0), (300, 30)]),
                element_id="title_1",
                parent_id=None,
                filename="ml_guide.pdf",
            ),
        ),
        Mock(
            text=(
                "Machine learning is a subset of artificial intelligence that "
                "focuses on developing algorithms and statistical models that "
                "enable computer systems to improve their performance on a "
                "specific task through experience."
            ),
            category="NarrativeText",
            metadata=Mock(
                page_number=1,
                coordinates=Mock(points=[(0, 40), (500, 100)]),
                element_id="para_1",
                parent_id="title_1",
                filename="ml_guide.pdf",
            ),
        ),
        Mock(
            text="Key Algorithms",
            category="Title",
            metadata=Mock(
                page_number=1,
                coordinates=Mock(points=[(0, 120), (200, 150)]),
                element_id="title_2",
                parent_id=None,
                filename="ml_guide.pdf",
            ),
        ),
        Mock(
            text=(
                "There are several fundamental algorithms in machine learning "
                "including supervised learning methods like linear regression, "
                "decision trees, and support vector machines. Each algorithm has "
                "specific use cases and performance characteristics."
            ),
            category="NarrativeText",
            metadata=Mock(
                page_number=1,
                coordinates=Mock(points=[(0, 160), (500, 220)]),
                element_id="para_2",
                parent_id="title_2",
                filename="ml_guide.pdf",
            ),
        ),
        Mock(
            text=(
                "<table><tr><th>Algorithm</th><th>Type</th><th>Use Case</th></tr>"
                "<tr><td>Linear Regression</td><td>Supervised</td><td>Prediction</td>"
                "</tr></table>"
            ),
            category="Table",
            metadata=Mock(
                page_number=1,
                coordinates=Mock(points=[(0, 240), (400, 300)]),
                element_id="table_1",
                parent_id="title_2",
                filename="ml_guide.pdf",
            ),
        ),
        Mock(
            text="Conclusions",
            category="Title",
            metadata=Mock(
                page_number=2,
                coordinates=Mock(points=[(0, 0), (150, 30)]),
                element_id="title_3",
                parent_id=None,
                filename="ml_guide.pdf",
            ),
        ),
        Mock(
            text=(
                "Machine learning continues to evolve with new techniques and "
                "applications emerging regularly. Understanding the fundamentals "
                "is crucial for effective implementation."
            ),
            category="NarrativeText",
            metadata=Mock(
                page_number=2,
                coordinates=Mock(points=[(0, 40), (500, 100)]),
                element_id="para_3",
                parent_id="title_3",
                filename="ml_guide.pdf",
            ),
        ),
    ]


@pytest.fixture
def sample_multipage_elements():
    """Sample elements that span multiple pages for testing multipage sections."""
    return [
        Mock(
            text="Large Section Title",
            category="Title",
            metadata=Mock(
                page_number=1, element_id="title_large", filename="multipage.pdf"
            ),
        ),
        Mock(
            text="This is content on page 1 that belongs to a large section.",
            category="NarrativeText",
            metadata=Mock(
                page_number=1, parent_id="title_large", filename="multipage.pdf"
            ),
        ),
        Mock(
            text="Continued content on page 2 still part of the same logical section.",
            category="NarrativeText",
            metadata=Mock(
                page_number=2, parent_id="title_large", filename="multipage.pdf"
            ),
        ),
        Mock(
            text="Final content on page 3 completing the large section.",
            category="NarrativeText",
            metadata=Mock(
                page_number=3, parent_id="title_large", filename="multipage.pdf"
            ),
        ),
    ]


class TestSemanticChunker:
    """Test suite for SemanticChunker implementation.

    Tests REQ-0024-v2: Semantic Chunking with chunk_by_title
    - chunk_by_title() with intelligent boundary detection
    - Semantic coherence preservation (90%+ boundary accuracy)
    - Configurable parameters (max_characters=1500, new_after_n_chars=1200)
    - Multipage section handling with multipage_sections=True
    - Document hierarchy preservation in chunk metadata
    """

    @pytest.mark.unit
    def test_chunker_initialization(self, mock_settings):
        """Test SemanticChunker initializes correctly.

        Should pass after implementation:
        - Creates chunker with proper settings
        - Sets up chunking parameters
        - Configures boundary detection strategy
        """
        chunker = SemanticChunker(mock_settings)

        assert chunker is not None
        assert hasattr(chunker, "settings")
        assert hasattr(chunker, "chunking_parameters")
        assert chunker.settings == mock_settings

        # Verify chunking parameters are configured correctly
        params = chunker.chunking_parameters
        assert params.max_characters == 1500
        assert params.new_after_n_chars == 1200
        assert params.combine_text_under_n_chars == 500
        assert params.multipage_sections is True

    @pytest.mark.unit
    def test_chunking_parameters_configuration(self, mock_settings):
        """Test chunking parameters configuration.

        Should pass after implementation:
        - Configures max_characters=1500 correctly
        - Sets new_after_n_chars=1200 for optimal chunk boundaries
        - Sets combine_text_under_n_chars=500 for small element handling
        - Enables multipage_sections=True for cross-page content
        """
        chunker = SemanticChunker(mock_settings)

        # Test default parameter configuration
        params = chunker.chunking_parameters
        assert isinstance(params, ChunkingParameters)

        # Verify specific parameter values
        assert params.max_characters == 1500
        assert params.new_after_n_chars == 1200
        assert params.combine_text_under_n_chars == 500
        assert params.multipage_sections is True

        # Test parameter validation
        assert params.new_after_n_chars < params.max_characters
        assert params.combine_text_under_n_chars < params.new_after_n_chars

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunk_by_title_integration(
        self, mock_unstructured_chunk_by_title, mock_settings, sample_document_elements
    ):
        """Test direct integration with unstructured.chunking.title.chunk_by_title.

        Should pass after implementation:
        - Calls chunk_by_title() with correct parameters
        - Uses intelligent boundary detection
        - Passes proper configuration parameters
        - Returns properly structured SemanticChunk objects
        """
        chunker = SemanticChunker(mock_settings)

        result = await chunker.chunk_elements(sample_document_elements)

        # Verify chunk_by_title was called with correct parameters
        mock_unstructured_chunk_by_title.assert_called_once_with(
            elements=sample_document_elements,
            max_characters=1500,
            new_after_n_chars=1200,
            combine_text_under_n_chars=500,
            multipage_sections=True,
        )

        # Verify result structure
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 3
        assert result.total_elements == len(sample_document_elements)
        assert result.boundary_accuracy >= 0.9  # 90% accuracy target

        # Verify chunk structure
        for chunk in result.chunks:
            assert isinstance(chunk, SemanticChunk)
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "metadata")
            assert len(chunk.text) <= 1500  # Respects max_characters

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_semantic_boundary_detection(
        self, mock_unstructured_chunk_by_title, mock_settings, sample_document_elements
    ):
        """Test intelligent semantic boundary detection.

        Should pass after implementation:
        - Detects section boundaries at titles/headers
        - Maintains semantic coherence within chunks
        - Achieves 90%+ boundary accuracy target
        - Preserves logical document flow
        """
        chunker = SemanticChunker(mock_settings)

        result = await chunker.chunk_elements(sample_document_elements)

        # Verify boundary accuracy meets target
        assert result.boundary_accuracy >= 0.9

        # Verify semantic coherence in chunks
        for chunk in result.chunks:
            # Each chunk should maintain topic coherence
            assert hasattr(chunk.metadata, "section_title")
            assert chunk.metadata.section_title is not None

            # Chunks should not break mid-sentence or mid-thought
            assert not chunk.text.strip().startswith("and ")
            assert not chunk.text.strip().startswith("but ")
            assert not chunk.text.strip().startswith("however ")

        # Verify logical document flow is preserved
        section_titles = [chunk.metadata.section_title for chunk in result.chunks]
        expected_flow = ["Introduction", "Methodology", "Conclusion"]
        assert section_titles == expected_flow

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multipage_section_handling(
        self, mock_unstructured_chunk_by_title, mock_settings, sample_multipage_elements
    ):
        """Test multipage section handling with multipage_sections=True.

        Should pass after implementation:
        - Handles content spanning multiple pages
        - Maintains section coherence across pages
        - Preserves parent-child relationships across pages
        - Does not break sections at page boundaries artificially
        """
        # Configure mock for multipage scenario
        multipage_chunks = [
            Mock(
                text=(
                    "Large Section Title. This is content on page 1 that belongs "
                    "to a large section. Continued content on page 2 still part of "
                    "the same logical section."
                ),
                category="CompositeElement",
                metadata=Mock(
                    section_title="Large Section Title",
                    spans_pages=[1, 2, 3],
                    chunk_index=0,
                ),
            )
        ]
        mock_unstructured_chunk_by_title.return_value = multipage_chunks

        chunker = SemanticChunker(mock_settings)

        result = await chunker.chunk_elements(sample_multipage_elements)

        # Verify multipage_sections parameter was used
        call_args = mock_unstructured_chunk_by_title.call_args
        assert call_args[1]["multipage_sections"] is True

        # Verify multipage handling
        multipage_chunk = result.chunks[0]
        assert hasattr(multipage_chunk.metadata, "spans_pages")
        assert len(multipage_chunk.metadata.spans_pages) > 1

        # Verify content from multiple pages is preserved together
        assert "page 1" in multipage_chunk.text
        assert "page 2" in multipage_chunk.text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_document_hierarchy_preservation(
        self, mock_unstructured_chunk_by_title, mock_settings, sample_document_elements
    ):
        """Test document hierarchy preservation in chunk metadata.

        Should pass after implementation:
        - Preserves parent-child relationships in chunk metadata
        - Maintains section titles and headers
        - Preserves document structure information
        - Maps chunks back to original document elements
        """
        chunker = SemanticChunker(mock_settings)

        result = await chunker.chunk_elements(sample_document_elements)

        # Verify hierarchy preservation
        for chunk in result.chunks:
            # Each chunk should preserve section information
            assert hasattr(chunk.metadata, "section_title")
            assert hasattr(chunk.metadata, "parent_elements")
            assert hasattr(chunk.metadata, "chunk_index")

            # Verify parent element mapping
            if hasattr(chunk.metadata, "parent_elements"):
                parent_ids = chunk.metadata.parent_elements
                assert isinstance(parent_ids, list)
                assert len(parent_ids) > 0

        # Verify document structure is maintained
        chunk_titles = [chunk.metadata.section_title for chunk in result.chunks]
        original_titles = [
            elem.text for elem in sample_document_elements if elem.category == "Title"
        ]

        # All original titles should be represented in chunks
        for title in original_titles:
            assert any(title in chunk_title for chunk_title in chunk_titles)

    @pytest.mark.unit
    def test_chunk_size_validation(self, mock_settings):
        """Test chunk size parameter validation and enforcement.

        Should pass after implementation:
        - Validates max_characters parameter
        - Validates new_after_n_chars parameter
        - Enforces logical parameter relationships
        - Handles edge cases gracefully
        """
        chunker = SemanticChunker(mock_settings)

        # Test parameter validation
        params = ChunkingParameters(
            max_characters=1500,
            new_after_n_chars=1200,
            combine_text_under_n_chars=500,
            multipage_sections=True,
        )

        assert chunker._validate_parameters(params) is True

        # Test invalid parameter combinations
        invalid_params = ChunkingParameters(
            max_characters=1000,
            new_after_n_chars=1200,  # Greater than max_characters
            combine_text_under_n_chars=500,
            multipage_sections=True,
        )

        assert chunker._validate_parameters(invalid_params) is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_table_and_image_handling(
        self, mock_unstructured_chunk_by_title, mock_settings, sample_document_elements
    ):
        """Test handling of tables and images in semantic chunking.

        Should pass after implementation:
        - Preserves table structure within chunks
        - Maintains image context and OCR content
        - Does not break tables across chunks inappropriately
        - Preserves multimodal content relationships
        """
        chunker = SemanticChunker(mock_settings)

        result = await chunker.chunk_elements(sample_document_elements)

        # Find chunks containing tables
        table_chunks = [chunk for chunk in result.chunks if "table>" in chunk.text]
        assert len(table_chunks) > 0

        # Verify table structure is preserved
        table_chunk = table_chunks[0]
        assert "<table>" in table_chunk.text
        assert "<tr>" in table_chunk.text
        assert "<td>" in table_chunk.text

        # Verify table context is maintained
        assert hasattr(table_chunk.metadata, "contains_table")
        assert table_chunk.metadata.contains_table is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_overlap_handling(
        self, mock_unstructured_chunk_by_title, mock_settings, sample_document_elements
    ):
        """Test chunk overlap handling for context preservation.

        Should pass after implementation:
        - Creates appropriate overlap between adjacent chunks
        - Maintains context at chunk boundaries
        - Preserves important information across chunks
        - Handles overlap configuration parameters
        """
        # Configure settings with overlap
        mock_settings.chunk_overlap = 200
        chunker = SemanticChunker(mock_settings)

        result = await chunker.chunk_elements(sample_document_elements)

        # Verify overlap is handled appropriately
        if len(result.chunks) > 1:
            for i in range(len(result.chunks) - 1):
                current_chunk = result.chunks[i]
                next_chunk = result.chunks[i + 1]

                # Check for semantic overlap (not necessarily text duplication)
                assert hasattr(current_chunk.metadata, "overlaps_with")
                if current_chunk.metadata.overlaps_with:
                    assert (
                        next_chunk.metadata.chunk_index
                        in current_chunk.metadata.overlaps_with
                    )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_empty_elements(self, mock_settings):
        """Test error handling for empty or invalid elements.

        Should pass after implementation:
        - Handles empty element lists gracefully
        - Skips elements with no content
        - Returns appropriate error messages
        - Maintains system stability
        """
        chunker = SemanticChunker(mock_settings)

        # Test empty elements list
        result = await chunker.chunk_elements([])
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 0
        assert result.total_elements == 0

        # Test elements with no text content
        empty_elements = [
            Mock(text="", category="Title", metadata=Mock()),
            Mock(text=None, category="NarrativeText", metadata=Mock()),
        ]

        result = await chunker.chunk_elements(empty_elements)
        assert isinstance(result, ChunkingResult)
        # Should filter out empty elements
        assert result.total_elements == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_performance_chunking_large_documents(
        self, mock_unstructured_chunk_by_title, mock_settings
    ):
        """Test performance with large document processing.

        Should pass after implementation:
        - Handles large numbers of elements efficiently
        - Maintains performance with complex documents
        - Processes documents within reasonable time limits
        - Scales appropriately with document size
        """
        # Create large document with many elements
        large_document = []
        for i in range(100):  # 100 elements
            element = Mock(
                text=(
                    f"This is paragraph {i} with substantial content that needs "
                    f"to be processed efficiently during semantic chunking operations."
                ),
                category="NarrativeText",
                metadata=Mock(
                    page_number=i // 10 + 1,
                    element_id=f"para_{i}",
                    filename="large_doc.pdf",
                ),
            )
            large_document.append(element)

        chunker = SemanticChunker(mock_settings)

        start_time = asyncio.get_event_loop().time()
        result = await chunker.chunk_elements(large_document)
        end_time = asyncio.get_event_loop().time()

        processing_time = end_time - start_time

        # Performance target: should process quickly
        assert processing_time < 2.0  # Less than 2 seconds for 100 elements
        assert isinstance(result, ChunkingResult)
        assert result.total_elements == 100

    @pytest.mark.unit
    def test_boundary_accuracy_calculation(self, mock_settings):
        """Test boundary accuracy calculation methodology.

        Should pass after implementation:
        - Calculates boundary accuracy correctly
        - Uses appropriate metrics for semantic coherence
        - Provides meaningful accuracy scores
        - Handles edge cases in accuracy calculation
        """
        chunker = SemanticChunker(mock_settings)

        # Mock chunk boundaries for testing
        chunk_boundaries = [
            {"start": 0, "end": 500, "semantic_break": True},
            {"start": 500, "end": 1000, "semantic_break": True},
            {"start": 1000, "end": 1400, "semantic_break": False},  # Poor boundary
        ]

        accuracy = chunker._calculate_boundary_accuracy(chunk_boundaries)

        # Should detect 2 good boundaries out of 3 total = 66.7%
        assert 0.6 <= accuracy <= 0.7

        # Test perfect boundaries
        perfect_boundaries = [
            {"start": 0, "end": 500, "semantic_break": True},
            {"start": 500, "end": 1000, "semantic_break": True},
        ]

        perfect_accuracy = chunker._calculate_boundary_accuracy(perfect_boundaries)
        assert perfect_accuracy == 1.0


class TestGherkinScenariosChunking:
    """Test Gherkin scenarios for semantic chunking from ADR-009."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scenario_intelligent_semantic_chunking(
        self, mock_unstructured_chunk_by_title, mock_settings, sample_document_elements
    ):
        """Test Scenario 2: Intelligent Semantic Chunking.

        Given: A document with multiple sections and content types
        When: Using SemanticChunker.chunk_elements() with chunk_by_title
        Then: Chunks are created with 90%+ boundary accuracy
        And: Document hierarchy is preserved in metadata
        And: Multipage sections are handled correctly
        And: Configurable parameters are applied correctly
        """
        chunker = SemanticChunker(mock_settings)

        # When: Chunking elements
        result = await chunker.chunk_elements(sample_document_elements)

        # Then: 90%+ boundary accuracy
        assert result.boundary_accuracy >= 0.9

        # And: Document hierarchy preserved
        for chunk in result.chunks:
            assert hasattr(chunk.metadata, "section_title")
            assert hasattr(chunk.metadata, "parent_elements")

        # And: Multipage handling verified
        call_args = mock_unstructured_chunk_by_title.call_args
        assert call_args[1]["multipage_sections"] is True

        # And: Configurable parameters applied
        assert call_args[1]["max_characters"] == 1500
        assert call_args[1]["new_after_n_chars"] == 1200
        assert call_args[1]["combine_text_under_n_chars"] == 500

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scenario_chunk_coherence_validation(
        self, mock_unstructured_chunk_by_title, mock_settings, sample_document_elements
    ):
        """Test chunk coherence and semantic boundaries.

        Given: Complex document with various content types
        When: Semantic chunking is applied
        Then: Each chunk maintains topical coherence
        And: Boundaries occur at logical section breaks
        And: Related content is kept together
        And: No important context is lost at boundaries
        """
        chunker = SemanticChunker(mock_settings)

        # When: Chunking complex document
        result = await chunker.chunk_elements(sample_document_elements)

        # Then: Topical coherence maintained
        for chunk in result.chunks:
            # Each chunk should focus on one main topic
            section_title = chunk.metadata.section_title
            assert section_title in chunk.text or any(
                keyword in chunk.text.lower()
                for keyword in section_title.lower().split()
            )

        # And: Logical section breaks
        chunk_sections = [chunk.metadata.section_title for chunk in result.chunks]
        # Should not have the same section split unnecessarily
        assert len(set(chunk_sections)) <= len(chunk_sections)

        # And: Related content kept together
        intro_chunks = [
            chunk
            for chunk in result.chunks
            if "Introduction" in chunk.metadata.section_title
        ]
        if intro_chunks:
            intro_chunk = intro_chunks[0]
            assert "machine learning" in intro_chunk.text.lower()
            assert "artificial intelligence" in intro_chunk.text.lower()


class TestChunkingParameterCustomization:
    """Test customization of chunking parameters."""

    @pytest.mark.unit
    def test_custom_parameter_configuration(self, mock_settings):
        """Test custom chunking parameter configuration.

        Should pass after implementation:
        - Allows override of default parameters
        - Validates parameter combinations
        - Maintains parameter constraints
        - Supports different chunking strategies
        """
        # Test custom parameters
        custom_params = ChunkingParameters(
            max_characters=2000,
            new_after_n_chars=1800,
            combine_text_under_n_chars=300,
            multipage_sections=False,
        )

        chunker = SemanticChunker(mock_settings, chunking_parameters=custom_params)

        assert chunker.chunking_parameters.max_characters == 2000
        assert chunker.chunking_parameters.new_after_n_chars == 1800
        assert chunker.chunking_parameters.combine_text_under_n_chars == 300
        assert chunker.chunking_parameters.multipage_sections is False

    @pytest.mark.unit
    def test_adaptive_parameter_adjustment(self, mock_settings):
        """Test adaptive parameter adjustment based on content.

        Should pass after implementation:
        - Adjusts parameters based on document characteristics
        - Optimizes chunk size for different content types
        - Maintains coherence across different document structures
        - Provides feedback on parameter effectiveness
        """
        chunker = SemanticChunker(mock_settings)

        # Test parameter adaptation for different content types
        technical_content = Mock(content_type="technical", avg_sentence_length=150)
        narrative_content = Mock(content_type="narrative", avg_sentence_length=80)

        tech_params = chunker._adapt_parameters_for_content(technical_content)
        narrative_params = chunker._adapt_parameters_for_content(narrative_content)

        # Technical content might need larger chunks
        assert tech_params.max_characters >= narrative_params.max_characters
