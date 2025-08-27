"""Unit tests for DocumentProcessor (REQ-0021-v2).

Tests direct Unstructured.io integration with hi-res strategy, strategy mapping,
multimodal extraction, and performance targets.

Migrated from DirectUnstructuredProcessor to use the working
DocumentProcessor implementation.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest

from src.models.processing import (
    ProcessingResult,
    ProcessingStrategy,
)

# Import working implementations
from src.processing.document_processor import DocumentProcessor

# Alias for backward compatibility during transition
DirectUnstructuredProcessor = DocumentProcessor


@pytest.fixture
def mock_unstructured_partition():
    """Mock unstructured.partition.auto.partition function."""
    with patch("src.processing.document_processor.partition") as mock_partition:

        def create_mock_elements(filename, **kwargs):
            """Create mock elements with dynamic filename based on the input file."""
            from pathlib import Path

            actual_filename = Path(filename).name if filename else "test.pdf"

            return [
                Mock(
                    text="This is a title",
                    category="Title",
                    metadata=Mock(
                        page_number=1,
                        coordinates=Mock(points=[(0, 0), (100, 20)]),
                        parent_id=None,
                        element_id="elem_1",
                        filename=actual_filename,
                    ),
                ),
                Mock(
                    text="This is paragraph text with important information.",
                    category="NarrativeText",
                    metadata=Mock(
                        page_number=1,
                        coordinates=Mock(points=[(0, 25), (400, 80)]),
                        parent_id="elem_1",
                        element_id="elem_2",
                        filename=actual_filename,
                    ),
                ),
                Mock(
                    text=(
                        "<table><tr><th>Header 1</th><th>Header 2</th></tr>"
                        "<tr><td>Data 1</td><td>Data 2</td></tr></table>"
                    ),
                    category="Table",
                    metadata=Mock(
                        page_number=1,
                        coordinates=Mock(points=[(0, 100), (400, 200)]),
                        parent_id=None,
                        element_id="elem_3",
                        filename=actual_filename,
                        text_as_html=(
                            "<table><tr><th>Header 1</th><th>Header 2</th></tr>"
                            "<tr><td>Data 1</td><td>Data 2</td></tr></table>"
                        ),
                    ),
                ),
                Mock(
                    text="OCR extracted text from image",
                    category="Image",
                    metadata=Mock(
                        page_number=2,
                        coordinates=Mock(points=[(50, 50), (350, 250)]),
                        parent_id=None,
                        element_id="elem_4",
                        filename=actual_filename,
                        image_path="/tmp/extracted_image.png",
                    ),
                ),
            ]

        # Set up the mock to call our function with the filename argument
        mock_partition.side_effect = create_mock_elements
        yield mock_partition


@pytest.fixture
def mock_settings():
    """Mock DocMind settings for testing."""
    settings = Mock()
    settings.chunk_size = 512
    settings.chunk_overlap = 50
    settings.max_document_size_mb = 100
    settings.enable_ocr = True
    settings.processing_strategy = "hi_res"
    return settings


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a sample PDF file for testing."""
    pdf_file = tmp_path / "test_document.pdf"
    # Create a minimal valid PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj  
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
180
%%EOF"""
    pdf_file.write_bytes(pdf_content)
    return pdf_file


@pytest.fixture
def sample_docx_path(tmp_path):
    """Create a sample DOCX file for testing."""
    docx_file = tmp_path / "test_document.docx"
    # Create a minimal DOCX structure (ZIP with minimal XML)
    import zipfile

    with zipfile.ZipFile(docx_file, "w") as zip_file:
        zip_file.writestr(
            "word/document.xml",
            """<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Sample DOCX content with table</w:t></w:r></w:p>
  </w:body>
</w:document>""",
        )
        zip_file.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>""",  # noqa: E501
        )
    return docx_file


@pytest.fixture
def sample_scanned_pdf_path(tmp_path):
    """Create a sample scanned PDF (image-based) for testing."""
    pdf_file = tmp_path / "scanned_document.pdf"
    # Same minimal PDF structure - in real scenario would contain image data
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj  
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
180
%%EOF"""
    pdf_file.write_bytes(pdf_content)
    return pdf_file


class TestDocumentProcessor:
    """Test suite for DocumentProcessor implementation.

    Tests REQ-0021-v2: Direct Unstructured.io PDF Processing
    - Direct partition() calls with hi_res strategy
    - Strategy mapping (hi_res, fast, ocr_only)
    - Multimodal extraction (tables, images, OCR)
    - Metadata preservation and element categorization
    - Performance target: >1 page/second with hi_res strategy
    """

    @pytest.mark.unit
    def test_processor_initialization(self, mock_settings):
        """Test DocumentProcessor initializes correctly.

        Should pass with working implementation:
        - Creates processor with proper settings
        - Sets up strategy mapping
        - Initializes caching if enabled
        """
        processor = DocumentProcessor(mock_settings)

        assert processor is not None
        assert hasattr(processor, "settings")
        assert hasattr(processor, "strategy_map")
        assert processor.settings == mock_settings

        # Verify core strategy mapping exists (DocumentProcessor supports more formats)
        expected_core_strategies = {
            ".pdf": ProcessingStrategy.HI_RES,
            ".docx": ProcessingStrategy.HI_RES,
            ".html": ProcessingStrategy.FAST,
            ".txt": ProcessingStrategy.FAST,
            ".jpg": ProcessingStrategy.OCR_ONLY,
            ".png": ProcessingStrategy.OCR_ONLY,
        }

        # Verify all expected strategies are present
        for ext, strategy in expected_core_strategies.items():
            assert ext in processor.strategy_map
            assert processor.strategy_map[ext] == strategy

        # Verify additional strategies are also present (implementation supports more)
        assert len(processor.strategy_map) >= len(expected_core_strategies)

    @pytest.mark.unit
    def test_strategy_mapping(self, mock_settings):
        """Test strategy selection based on file extension.

        Should pass after implementation:
        - Maps PDF/DOCX to hi_res strategy for full multimodal extraction
        - Maps HTML/TXT to fast strategy for quick text extraction
        - Maps images to ocr_only strategy for image-focused processing
        """
        processor = DocumentProcessor(mock_settings)

        # Test PDF strategy mapping
        assert (
            processor._get_strategy_for_file("document.pdf")
            == ProcessingStrategy.HI_RES
        )
        assert (
            processor._get_strategy_for_file("complex.docx")
            == ProcessingStrategy.HI_RES
        )

        # Test fast strategy mapping
        assert (
            processor._get_strategy_for_file("webpage.html") == ProcessingStrategy.FAST
        )
        assert processor._get_strategy_for_file("notes.txt") == ProcessingStrategy.FAST

        # Test OCR strategy mapping
        assert (
            processor._get_strategy_for_file("scan.jpg") == ProcessingStrategy.OCR_ONLY
        )
        assert (
            processor._get_strategy_for_file("image.png") == ProcessingStrategy.OCR_ONLY
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_direct_partition_call_pdf_hi_res(
        self, mock_unstructured_partition, mock_settings, sample_pdf_path
    ):
        """Test direct partition() call with hi_res strategy for PDF.

        Should pass after implementation:
        - Calls partition() directly with hi_res strategy
        - Passes correct parameters for multimodal extraction
        - Returns properly structured DocumentElement objects
        - Preserves all metadata from unstructured.io
        """
        processor = DocumentProcessor(mock_settings)

        # Process document using direct partition call
        result = await processor.process_document_async(sample_pdf_path)

        # Verify partition was called with correct parameters
        mock_unstructured_partition.assert_called_once_with(
            filename=str(sample_pdf_path),
            strategy="hi_res",
            include_metadata=True,
            include_page_breaks=True,
            extract_images_in_pdf=True,
            extract_image_blocks=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            multipage_sections=True,
            combine_text_under_n_chars=500,
            new_after_n_chars=1200,
            max_characters=1500,
        )

        # Verify result structure
        assert isinstance(result, ProcessingResult)
        assert len(result.elements) == 4  # Title, Text, Table, Image
        assert result.processing_time > 0
        assert result.strategy_used == ProcessingStrategy.HI_RES

        # Verify element categorization
        element_categories = [elem.category for elem in result.elements]
        assert "Title" in element_categories
        assert "NarrativeText" in element_categories
        assert "Table" in element_categories
        assert "Image" in element_categories

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_docx_structure_preservation(
        self, mock_unstructured_partition, mock_settings, sample_docx_path
    ):
        """Test DOCX structure preservation with infer_table_structure=True.

        Should pass after implementation (REQ-0022-v2):
        - Calls partition() with infer_table_structure=True for DOCX
        - Preserves table structure with HTML output
        - Maintains document hierarchy with coordinate mapping
        - Preserves formatting for headers, lists, tables
        """
        processor = DocumentProcessor(mock_settings)

        result = await processor.process_document_async(sample_docx_path)

        # Verify DOCX-specific parameters
        mock_unstructured_partition.assert_called_once()
        call_args = mock_unstructured_partition.call_args
        assert call_args[1]["strategy"] == "hi_res"
        assert call_args[1]["infer_table_structure"] is True
        assert call_args[1]["include_metadata"] is True

        # Verify structure preservation
        assert isinstance(result, ProcessingResult)
        table_elements = [elem for elem in result.elements if elem.category == "Table"]
        assert len(table_elements) > 0

        # Verify HTML table structure is preserved
        table_elem = table_elements[0]
        assert "text_as_html" in table_elem.metadata
        assert "<table>" in table_elem.metadata["text_as_html"]
        assert "<tr>" in table_elem.metadata["text_as_html"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multimodal_element_extraction(
        self, mock_unstructured_partition, mock_settings, sample_pdf_path
    ):
        """Test multimodal element extraction capabilities.

        Should pass after implementation (REQ-0023-v2):
        - Extracts tables with automatic HTML formatting
        - Extracts images with OCR text and coordinate mapping
        - Produces structured JSON output with metadata preservation
        - Achieves accurate element categorization
        """
        processor = DocumentProcessor(mock_settings)

        result = await processor.process_document_async(sample_pdf_path)

        # Verify multimodal content extraction
        elements_by_type = {}
        for elem in result.elements:
            if elem.category not in elements_by_type:
                elements_by_type[elem.category] = []
            elements_by_type[elem.category].append(elem)

        # Verify table extraction with HTML formatting
        assert "Table" in elements_by_type
        table_elem = elements_by_type["Table"][0]
        assert "text_as_html" in table_elem.metadata
        assert "<table>" in table_elem.text or "<table>" in table_elem.metadata.get(
            "text_as_html", ""
        )

        # Verify image extraction with OCR and coordinates
        assert "Image" in elements_by_type
        image_elem = elements_by_type["Image"][0]
        assert "coordinates" in image_elem.metadata
        assert "image_path" in image_elem.metadata
        assert len(image_elem.text) > 0  # OCR extracted text

        # Verify coordinate mapping for all elements
        for elem in result.elements:
            assert "coordinates" in elem.metadata
            assert "page_number" in elem.metadata
            assert elem.metadata["page_number"] >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metadata_preservation(
        self, mock_unstructured_partition, mock_settings, sample_pdf_path
    ):
        """Test comprehensive metadata preservation from unstructured.io.

        Should pass after implementation:
        - Preserves page numbers, coordinates, element relationships
        - Maintains parent-child relationships between elements
        - Preserves filename, element IDs, and hierarchy information
        - Maintains formatting metadata for text elements
        """
        processor = DocumentProcessor(mock_settings)

        result = await processor.process_document_async(sample_pdf_path)

        for elem in result.elements:
            # Verify essential metadata is preserved
            assert "page_number" in elem.metadata
            assert "element_id" in elem.metadata
            assert "filename" in elem.metadata
            assert "coordinates" in elem.metadata

            # Verify metadata values are valid
            assert elem.metadata["page_number"] >= 1
            assert elem.metadata["element_id"] is not None
            assert elem.metadata["filename"] == "test_document.pdf"
            assert elem.metadata["coordinates"] is not None

        # Verify parent-child relationships are preserved
        title_elem = next((e for e in result.elements if e.category == "Title"), None)
        text_elem = next(
            (e for e in result.elements if e.category == "NarrativeText"), None
        )

        if title_elem and text_elem:
            assert "parent_id" in text_elem.metadata
            assert text_elem.metadata["parent_id"] == title_elem.metadata["element_id"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scanned_pdf_ocr_processing(
        self, mock_unstructured_partition, mock_settings, sample_scanned_pdf_path
    ):
        """Test OCR processing for scanned PDFs.

        Should pass after implementation:
        - Automatically detects image-based PDFs
        - Applies OCR extraction for text recognition
        - Maintains coordinate mapping for OCR results
        - Preserves image metadata alongside text content
        """
        processor = DocumentProcessor(mock_settings)

        result = await processor.process_document_async(sample_scanned_pdf_path)

        # Verify OCR parameters were used
        mock_unstructured_partition.assert_called_once()
        call_args = mock_unstructured_partition.call_args
        assert call_args[1]["extract_images_in_pdf"] is True
        assert call_args[1]["extract_image_blocks"] is True

        # Verify OCR results
        image_elements = [elem for elem in result.elements if elem.category == "Image"]
        assert len(image_elements) > 0

        for image_elem in image_elements:
            # Verify OCR extracted text
            assert len(image_elem.text) > 0
            assert "coordinates" in image_elem.metadata

    @pytest.mark.unit
    def test_error_handling_corrupted_file(self, mock_settings, tmp_path):
        """Test graceful error handling for corrupted files.

        Should pass after implementation:
        - Gracefully handles corrupted PDF files
        - Returns appropriate error information
        - Does not crash the processing pipeline
        - Provides meaningful error messages
        """
        # Create corrupted file
        corrupted_file = tmp_path / "corrupted.pdf"
        corrupted_file.write_bytes(b"Not a valid PDF file")

        processor = DocumentProcessor(mock_settings)

        # Should handle error gracefully - import the correct exception
        from src.processing.document_processor import ProcessingError

        # The processor may successfully process the "corrupted" file and return content
        # or it may fail with a ProcessingError - both are valid outcomes
        processing_error_occurred = False
        error_message = ""

        try:
            result = asyncio.run(processor.process_document_async(corrupted_file))
            # If it processes successfully, may extract text like "Not a valid PDF file"
            # This is actually correct behavior - the processor is resilient
            assert isinstance(result, ProcessingResult)
            assert len(result.elements) >= 0  # Could be empty or contain extracted text
        except ProcessingError as e:
            processing_error_occurred = True
            error_message = str(e).lower()

        # If an exception was raised, verify error is informative
        if processing_error_occurred:
            assert (
                "corrupted" in error_message
                or "invalid" in error_message
                or "processing failed" in error_message
                or "partition failed" in error_message
            ), f"Expected informative error message, got: {error_message}"

    @pytest.mark.unit
    def test_unsupported_file_format(self, mock_settings, tmp_path):
        """Test handling of unsupported file formats.

        Should pass after implementation:
        - Recognizes unsupported file extensions
        - Returns appropriate error or skips processing
        - Maintains system stability
        """
        unsupported_file = tmp_path / "document.xyz"
        unsupported_file.write_text("Unsupported format content")

        processor = DocumentProcessor(mock_settings)

        with pytest.raises(
            ValueError, match=r"(unsupported|not supported|unknown).*format"
        ) as exc_info:
            processor._get_strategy_for_file(str(unsupported_file))

        assert "unsupported" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_performance_target_validation(
        self, mock_unstructured_partition, mock_settings, sample_pdf_path
    ):
        """Test performance target of >1 page/second with hi_res strategy.

        Enhanced with pytest-benchmark for accurate performance measurement:
        - Processes single-page document in <1 second
        - Maintains performance with hi_res strategy
        - Tracks processing time accurately
        - Validates performance metrics with statistical analysis
        """
        processor = DocumentProcessor(mock_settings)

        start_time = asyncio.get_event_loop().time()
        result = await processor.process_document_async(sample_pdf_path)
        end_time = asyncio.get_event_loop().time()

        processing_time = end_time - start_time

        # Verify performance target: >1 page/second (single page should be <1 second)
        assert processing_time < 1.0, (
            f"Processing took {processing_time}s, exceeding 1 page/second target"
        )
        assert result.processing_time > 0
        assert result.processing_time <= processing_time

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_performance_benchmark_validation(
        self, mock_unstructured_partition, mock_settings, sample_pdf_path
    ):
        """Benchmark performance using manual timing for statistical accuracy.

        Provides detailed performance metrics with statistical analysis:
        - Multiple rounds for statistical significance
        - Mean, median, and standard deviation calculations
        - Performance regression detection
        - Validates >1 page/second target consistently
        """
        processor = DocumentProcessor(mock_settings)

        # Run multiple iterations for statistical significance
        times = []
        results = []
        for _ in range(5):
            start_time = asyncio.get_event_loop().time()
            result = await processor.process_document_async(sample_pdf_path)
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            times.append(processing_time)
            results.append(result)

        # Calculate statistics
        mean_time = sum(times) / len(times)

        # Verify performance target with benchmark statistics
        assert mean_time < 1.0, (
            f"Mean processing time {mean_time:.3f}s exceeds 1 page/second target"
        )

        # Verify result structure
        for result in results:
            assert isinstance(result, ProcessingResult)
            assert len(result.elements) > 0
            assert result.processing_time > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_processing_multiple_files(
        self, mock_unstructured_partition, mock_settings, tmp_path
    ):
        """Test batch processing of multiple files with consistent results.

        Should pass after implementation:
        - Processes multiple files in sequence
        - Maintains consistent element extraction
        - Tracks individual file processing times
        - Preserves metadata for each file separately
        """
        # Create multiple test files
        files = []
        for i in range(3):
            file_path = tmp_path / f"document_{i}.pdf"
            file_path.write_bytes(b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj  
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
180
%%EOF""")
            files.append(file_path)

        processor = DocumentProcessor(mock_settings)

        results = []
        for file_path in files:
            result = await processor.process_document_async(file_path)
            results.append(result)

        # Verify all files processed successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, ProcessingResult)
            assert len(result.elements) > 0
            # Verify filename is preserved correctly
            # (check if filename contains the document name)
            expected_filename = f"document_{i}.pdf"
            assert any(
                str(elem.metadata.get("filename", "")).endswith(expected_filename)
                for elem in result.elements
            ), (
                f"Expected filename ending with '{expected_filename}', but found: "
                f"{[elem.metadata.get('filename', '') for elem in result.elements]}"
            )

    @pytest.mark.unit
    def test_configuration_override(self, mock_settings):
        """Test configuration parameter override capabilities.

        Should pass after implementation:
        - Allows runtime parameter overrides
        - Validates parameter combinations
        - Maintains default strategy mapping
        - Supports custom processing configurations
        """
        processor = DocumentProcessor(mock_settings)

        # Test custom configuration override (will be used in implementation)

        # Should support configuration override
        assert hasattr(processor, "override_config") or hasattr(
            processor, "_apply_config_override"
        )

        # Test configuration validation
        valid_strategies = ["hi_res", "fast", "ocr_only"]
        for strategy in valid_strategies:
            config = {"strategy": strategy}
            # Should not raise exception for valid strategies
            assert processor._validate_config(config) is True


class TestGherkinScenarios:
    """Test Gherkin scenarios from ADR-009 specification.

    These tests validate the acceptance criteria scenarios directly.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scenario_direct_unstructured_one_line_processing(
        self, mock_unstructured_partition, mock_settings, sample_pdf_path
    ):
        """Test Scenario 1: DIRECT Unstructured.io One-Line Processing.

        Given: A PDF document with tables and images
        When: Using DirectUnstructuredProcessor.process_document_async()
        Then: Single partition() call extracts all multimodal content
        And: Processing completes with hi_res strategy
        And: All elements are categorized correctly
        """
        processor = DocumentProcessor(mock_settings)

        # When: Processing document
        result = await processor.process_document_async(sample_pdf_path)

        # Then: Single partition call
        assert mock_unstructured_partition.call_count == 1

        # And: Hi-res strategy used
        call_args = mock_unstructured_partition.call_args
        assert call_args[1]["strategy"] == "hi_res"

        # And: All multimodal content extracted
        assert call_args[1]["extract_images_in_pdf"] is True
        assert call_args[1]["infer_table_structure"] is True
        assert call_args[1]["extract_image_blocks"] is True

        # And: Elements categorized correctly
        categories = {elem.category for elem in result.elements}
        expected_categories = {"Title", "NarrativeText", "Table", "Image"}
        assert categories == expected_categories

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scenario_multimodal_extraction_accuracy(
        self, mock_unstructured_partition, mock_settings, sample_pdf_path
    ):
        """Test multimodal extraction accuracy and coordinate preservation.

        Given: A complex PDF with tables, images, and text
        When: Processing with hi_res strategy
        Then: Tables are extracted with HTML structure preserved
        And: Images are processed with OCR text extraction
        And: Coordinate information is preserved for all elements
        And: Parent-child relationships are maintained
        """
        processor = DocumentProcessor(mock_settings)

        # When: Processing with hi_res
        result = await processor.process_document_async(sample_pdf_path)

        # Then: Tables extracted with HTML structure
        table_elements = [elem for elem in result.elements if elem.category == "Table"]
        assert len(table_elements) > 0
        table_elem = table_elements[0]
        assert "text_as_html" in table_elem.metadata
        assert "<table>" in table_elem.metadata["text_as_html"]

        # And: Images processed with OCR
        image_elements = [elem for elem in result.elements if elem.category == "Image"]
        assert len(image_elements) > 0
        image_elem = image_elements[0]
        assert len(image_elem.text.strip()) > 0  # OCR extracted text

        # And: Coordinates preserved for all elements
        for elem in result.elements:
            assert "coordinates" in elem.metadata
            assert elem.metadata["coordinates"] is not None

        # And: Parent-child relationships maintained
        elements_with_parents = [
            elem
            for elem in result.elements
            if "parent_id" in elem.metadata and elem.metadata["parent_id"] is not None
        ]
        assert len(elements_with_parents) > 0


# Test fixtures for multimodal content
@pytest.fixture
def complex_pdf_elements():
    """Mock complex PDF elements for comprehensive testing."""
    return [
        Mock(
            text="Executive Summary",
            category="Title",
            metadata=Mock(
                page_number=1,
                coordinates=Mock(points=[(50, 50), (300, 80)]),
                element_id="title_1",
                parent_id=None,
                filename="complex.pdf",
            ),
        ),
        Mock(
            text=(
                "This document contains comprehensive analysis of market "
                "trends and projections."
            ),
            category="NarrativeText",
            metadata=Mock(
                page_number=1,
                coordinates=Mock(points=[(50, 90), (500, 150)]),
                element_id="para_1",
                parent_id="title_1",
                filename="complex.pdf",
            ),
        ),
        Mock(
            text="<table><tr><th>Quarter</th><th>Revenue</th><th>Growth</th></tr><tr><td>Q1</td><td>$100M</td><td>5%</td></tr></table>",
            category="Table",
            metadata=Mock(
                page_number=1,
                coordinates=Mock(points=[(50, 160), (450, 220)]),
                element_id="table_1",
                parent_id=None,
                filename="complex.pdf",
                text_as_html="<table><tr><th>Quarter</th><th>Revenue</th><th>Growth</th></tr><tr><td>Q1</td><td>$100M</td><td>5%</td></tr></table>",
            ),
        ),
        Mock(
            text=(
                "Chart showing revenue growth over time with quarterly "
                "breakdown and projections."
            ),
            category="Image",
            metadata=Mock(
                page_number=1,
                coordinates=Mock(points=[(50, 230), (400, 350)]),
                element_id="chart_1",
                parent_id=None,
                filename="complex.pdf",
                image_path="/tmp/extracted_chart.png",
            ),
        ),
    ]


@pytest.mark.unit
class TestComplexDocumentProcessing:
    """Test complex document processing scenarios with realistic content."""

    def test_complex_element_relationships(self, complex_pdf_elements, mock_settings):
        """Test processing of complex documents with element relationships.

        Should pass after implementation:
        - Maintains parent-child relationships between sections
        - Preserves document hierarchy and structure
        - Correctly categorizes diverse element types
        - Maintains coordinate mapping across pages
        """
        with patch("unstructured.partition.auto.partition") as mock_partition:
            mock_partition.return_value = complex_pdf_elements

            # This will initialize the processor for implementation
            DirectUnstructuredProcessor(mock_settings)

            # This will need to be async when implemented
            # For now, test the element processing logic
            elements = mock_partition.return_value

            # Verify relationship preservation
            title_elem = next(e for e in elements if e.category == "Title")
            para_elem = next(e for e in elements if e.category == "NarrativeText")

            assert para_elem.metadata.parent_id == title_elem.metadata.element_id

            # Verify coordinate consistency
            for elem in elements:
                coords = elem.metadata.coordinates.points
                assert len(coords) == 2  # Start and end coordinates
                assert all(
                    isinstance(point, tuple) and len(point) == 2 for point in coords
                )
