"""Unit tests for DocumentProcessor.

This module provides comprehensive unit tests for the new DocumentProcessor
that combines Unstructured.io with LlamaIndex IngestionPipeline orchestration.

Test Coverage:
- DocumentProcessor initialization and configuration
- UnstructuredTransformation component behavior
- Strategy selection based on file types
- LlamaIndex IngestionPipeline integration
- Caching functionality (LlamaIndex cache + SimpleCache)
- Async processing with retry logic
- Error handling and recovery
- API compatibility with DocumentProcessor

Following 3-tier testing strategy:
- Tier 1 (Unit): Fast tests with mocks (<5s each)
- Use mocks for external dependencies (unstructured.partition, file system)
- Focus on logic validation and component integration
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.processing import DocumentElement, ProcessingResult, ProcessingStrategy
from src.processing.document_processor import (
    DocumentProcessor,
    ProcessingError,
    UnstructuredTransformation,
)


@pytest.fixture
def mock_settings():
    """Mock DocMind settings for testing."""
    settings = Mock()
    settings.chunk_size = 512
    settings.chunk_overlap = 50
    settings.max_document_size_mb = 100
    settings.cache_dir = "./cache"
    settings.bge_m3_model_name = "BAAI/bge-m3"
    return settings


@pytest.fixture
def mock_unstructured_elements():
    """Mock unstructured.io elements for testing."""
    return [
        Mock(
            text="This is a title element",
            category="Title",
            metadata=Mock(
                page_number=1,
                element_id="elem_1",
                parent_id=None,
                filename="test.pdf",
                coordinates=[(0, 0), (100, 20)],
                text_as_html=None,
                image_path=None,
            ),
        ),
        Mock(
            text="This is narrative text with important information.",
            category="NarrativeText",
            metadata=Mock(
                page_number=1,
                element_id="elem_2",
                parent_id="elem_1",
                filename="test.pdf",
                coordinates=[(0, 25), (400, 80)],
                text_as_html=None,
                image_path=None,
            ),
        ),
        Mock(
            text="<table><tr><th>Header 1</th><th>Header 2</th></tr></table>",
            category="Table",
            metadata=Mock(
                page_number=1,
                element_id="elem_3",
                parent_id=None,
                filename="test.pdf",
                coordinates=[(0, 100), (400, 200)],
                text_as_html=(
                    "<table><tr><th>Header 1</th><th>Header 2</th></tr></table>"
                ),
                image_path=None,
            ),
        ),
    ]


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a sample PDF file for testing."""
    pdf_file = tmp_path / "test_document.pdf"
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
def sample_image_path(tmp_path):
    """Create a sample image file for testing."""
    image_file = tmp_path / "test_image.jpg"
    # Create a minimal JPEG header
    jpeg_content = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00"
        b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12"
        b"\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444"
        b"\x1f'9=82<."
        b"342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4"
        b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02"
        b"\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05"
        b'\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142'
        b"\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:"
        b"CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97"
        b"\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3"
        b"\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7"
        b"\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xd9"
    )
    image_file.write_bytes(jpeg_content)
    return image_file


class TestUnstructuredTransformation:
    """Test UnstructuredTransformation component."""

    @pytest.mark.unit
    def test_initialization(self, mock_settings):
        """Test UnstructuredTransformation initialization."""
        transformation = UnstructuredTransformation(
            ProcessingStrategy.HI_RES, mock_settings
        )

        assert transformation.strategy == ProcessingStrategy.HI_RES
        assert transformation.settings == mock_settings

    @pytest.mark.unit
    def test_initialization_with_default_settings(self):
        """Test UnstructuredTransformation initialization with default settings."""
        with patch(
            "src.processing.document_processor.app_settings"
        ) as mock_app_settings:
            transformation = UnstructuredTransformation(ProcessingStrategy.FAST)

            assert transformation.strategy == ProcessingStrategy.FAST
            assert transformation.settings == mock_app_settings

    @pytest.mark.unit
    def test_build_partition_config_hi_res(self, mock_settings):
        """Test partition configuration for hi_res strategy."""
        transformation = UnstructuredTransformation(
            ProcessingStrategy.HI_RES, mock_settings
        )

        config = transformation._build_partition_config(ProcessingStrategy.HI_RES)

        # Verify base configuration
        assert config["strategy"] == "hi_res"
        assert config["include_metadata"] is True
        assert config["include_page_breaks"] is True

        # Verify hi_res specific configuration
        assert config["extract_images_in_pdf"] is True
        assert config["extract_image_blocks"] is True
        assert config["infer_table_structure"] is True
        assert config["chunking_strategy"] == "by_title"
        assert config["multipage_sections"] is True
        assert config["combine_text_under_n_chars"] == 500
        assert config["new_after_n_chars"] == 1200
        assert config["max_characters"] == 1500

    @pytest.mark.unit
    def test_build_partition_config_fast(self, mock_settings):
        """Test partition configuration for fast strategy."""
        transformation = UnstructuredTransformation(
            ProcessingStrategy.FAST, mock_settings
        )

        config = transformation._build_partition_config(ProcessingStrategy.FAST)

        # Verify base configuration
        assert config["strategy"] == "fast"
        assert config["include_metadata"] is True
        assert config["include_page_breaks"] is True

        # Verify fast specific configuration
        assert config["extract_images_in_pdf"] is False
        assert config["infer_table_structure"] is False
        assert config["chunking_strategy"] == "basic"
        assert config["max_characters"] == 1000

    @pytest.mark.unit
    def test_build_partition_config_ocr_only(self, mock_settings):
        """Test partition configuration for ocr_only strategy."""
        transformation = UnstructuredTransformation(
            ProcessingStrategy.OCR_ONLY, mock_settings
        )

        config = transformation._build_partition_config(ProcessingStrategy.OCR_ONLY)

        # Verify base configuration
        assert config["strategy"] == "ocr_only"
        assert config["include_metadata"] is True
        assert config["include_page_breaks"] is True

        # Verify OCR specific configuration
        assert config["extract_images_in_pdf"] is True
        assert config["extract_image_blocks"] is True
        assert config["infer_table_structure"] is False
        assert config["ocr_languages"] == ["eng"]

    @pytest.mark.unit
    @patch("src.processing.document_processor.partition")
    def test_transform_with_document_nodes(
        self, mock_partition, mock_settings, mock_unstructured_elements, sample_pdf_path
    ):
        """Test transformation of Document nodes."""
        from llama_index.core import Document

        mock_partition.return_value = mock_unstructured_elements

        transformation = UnstructuredTransformation(
            ProcessingStrategy.HI_RES, mock_settings
        )

        # Create Document node with file path metadata
        document_node = Document(text="", metadata={"file_path": str(sample_pdf_path)})

        result = transformation([document_node])

        # Verify partition was called
        mock_partition.assert_called_once_with(
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
        assert len(result) == 3  # Same as mock_unstructured_elements

        for i, node in enumerate(result):
            assert hasattr(node, "text")
            assert hasattr(node, "metadata")
            assert str(mock_unstructured_elements[i].text) in str(node.text)

    @pytest.mark.unit
    def test_transform_non_document_nodes(self, mock_settings):
        """Test transformation passes through non-Document nodes unchanged."""
        from llama_index.core.schema import TextNode

        transformation = UnstructuredTransformation(
            ProcessingStrategy.HI_RES, mock_settings
        )

        # Create non-Document node
        text_node = TextNode(text="This is a text node")

        result = transformation([text_node])

        # Should pass through unchanged
        assert len(result) == 1
        assert result[0] == text_node

    @pytest.mark.unit
    def test_extract_file_path_from_metadata(self, mock_settings, sample_pdf_path):
        """Test file path extraction from Document metadata."""
        from llama_index.core import Document

        transformation = UnstructuredTransformation(
            ProcessingStrategy.HI_RES, mock_settings
        )

        # Test various metadata field names
        test_cases = [
            {"file_path": str(sample_pdf_path)},
            {"filename": str(sample_pdf_path)},
            {"source": str(sample_pdf_path)},
            {"file_name": str(sample_pdf_path)},
            {"path": str(sample_pdf_path)},
        ]

        for metadata in test_cases:
            document = Document(text="", metadata=metadata)
            result = transformation._extract_file_path(document)
            assert result == sample_pdf_path

    @pytest.mark.unit
    def test_extract_file_path_not_found(self, mock_settings):
        """Test file path extraction when no valid path exists."""
        from llama_index.core import Document

        transformation = UnstructuredTransformation(
            ProcessingStrategy.HI_RES, mock_settings
        )

        # Document with no file path
        document = Document(text="", metadata={})
        result = transformation._extract_file_path(document)
        assert result is None

        # Document with non-existent file path
        document = Document(text="", metadata={"file_path": "/non/existent/path.pdf"})
        result = transformation._extract_file_path(document)
        assert result is None

    @pytest.mark.unit
    def test_convert_elements_to_nodes(
        self, mock_settings, mock_unstructured_elements, sample_pdf_path
    ):
        """Test conversion of unstructured elements to Document nodes."""
        from llama_index.core import Document

        transformation = UnstructuredTransformation(
            ProcessingStrategy.HI_RES, mock_settings
        )

        original_node = Document(text="", metadata={"file_path": str(sample_pdf_path)})

        result = transformation._convert_elements_to_nodes(
            mock_unstructured_elements, original_node, sample_pdf_path
        )

        assert len(result) == 3

        for i, node in enumerate(result):
            element = mock_unstructured_elements[i]

            # Verify text content
            assert node.text == str(element.text)

            # Verify metadata preservation and enhancement
            assert node.metadata["element_index"] == i
            assert node.metadata["element_category"] == str(element.category)
            assert node.metadata["processing_strategy"] == "hi_res"
            assert node.metadata["source_file"] == str(sample_pdf_path)

            # Verify element-specific metadata
            assert node.metadata["page_number"] == element.metadata.page_number
            assert node.metadata["element_id"] == element.metadata.element_id
            assert node.metadata["filename"] == element.metadata.filename

    @pytest.mark.unit
    @patch("src.processing.document_processor.partition")
    def test_transform_error_handling(
        self, mock_partition, mock_settings, sample_pdf_path
    ):
        """Test error handling in transformation."""
        from llama_index.core import Document

        mock_partition.side_effect = Exception("Partition failed")

        transformation = UnstructuredTransformation(
            ProcessingStrategy.HI_RES, mock_settings
        )

        document_node = Document(text="", metadata={"file_path": str(sample_pdf_path)})

        # Should not raise exception but return original node
        result = transformation([document_node])

        assert len(result) == 1
        assert result[0] == document_node


class TestDocumentProcessor:
    """Test DocumentProcessor class."""

    @pytest.mark.unit
    def test_initialization(self, mock_settings):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(mock_settings)

        assert processor.settings == mock_settings
        assert hasattr(processor, "strategy_map")
        assert hasattr(processor, "cache")
        assert hasattr(processor, "docstore")
        assert hasattr(processor, "simple_cache")

        # Verify strategy mapping
        expected_strategies = {
            ".pdf": ProcessingStrategy.HI_RES,
            ".docx": ProcessingStrategy.HI_RES,
            ".doc": ProcessingStrategy.HI_RES,
            ".pptx": ProcessingStrategy.HI_RES,
            ".html": ProcessingStrategy.FAST,
            ".txt": ProcessingStrategy.FAST,
            ".md": ProcessingStrategy.FAST,
            ".jpg": ProcessingStrategy.OCR_ONLY,
            ".png": ProcessingStrategy.OCR_ONLY,
        }

        for ext, strategy in expected_strategies.items():
            assert ext in processor.strategy_map
            assert processor.strategy_map[ext] == strategy

    @pytest.mark.unit
    @patch("src.processing.document_processor.app_settings")
    def test_initialization_with_default_settings(self, mock_app_settings):
        """Test initialization with default settings."""
        processor = DocumentProcessor()

        assert processor.settings == mock_app_settings

    @pytest.mark.unit
    def test_get_strategy_for_file(self, mock_settings):
        """Test strategy selection based on file extension."""
        processor = DocumentProcessor(mock_settings)

        # Test PDF mapping
        assert (
            processor._get_strategy_for_file("document.pdf")
            == ProcessingStrategy.HI_RES
        )
        assert (
            processor._get_strategy_for_file("document.PDF")
            == ProcessingStrategy.HI_RES
        )

        # Test DOCX mapping
        assert (
            processor._get_strategy_for_file("document.docx")
            == ProcessingStrategy.HI_RES
        )

        # Test HTML mapping
        assert processor._get_strategy_for_file("page.html") == ProcessingStrategy.FAST

        # Test image mapping
        assert (
            processor._get_strategy_for_file("image.jpg") == ProcessingStrategy.OCR_ONLY
        )
        assert (
            processor._get_strategy_for_file("scan.png") == ProcessingStrategy.OCR_ONLY
        )

    @pytest.mark.unit
    def test_get_strategy_for_file_unsupported(self, mock_settings):
        """Test error handling for unsupported file formats."""
        processor = DocumentProcessor(mock_settings)

        with pytest.raises(
            ValueError, match=r"(unsupported|not supported|unknown).*format"
        ) as exc_info:
            processor._get_strategy_for_file("document.xyz")

        assert "unsupported" in str(exc_info.value).lower()
        assert ".xyz" in str(exc_info.value)

    @pytest.mark.unit
    def test_calculate_document_hash(self, mock_settings, sample_pdf_path):
        """Test document hash calculation."""
        processor = DocumentProcessor(mock_settings)

        hash_result = processor._calculate_document_hash(sample_pdf_path)

        # Verify hash is a valid SHA-256 hex string
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result.lower())

        # Verify hash is consistent
        hash_result2 = processor._calculate_document_hash(sample_pdf_path)
        assert hash_result == hash_result2

        # Verify hash changes with file content
        sample_pdf_path.write_bytes(b"different content")
        hash_result3 = processor._calculate_document_hash(sample_pdf_path)
        assert hash_result != hash_result3

    @pytest.mark.unit
    def test_create_pipeline(self, mock_settings):
        """Test LlamaIndex IngestionPipeline creation."""
        processor = DocumentProcessor(mock_settings)

        pipeline = processor._create_pipeline(ProcessingStrategy.HI_RES)

        # Verify pipeline structure
        assert hasattr(pipeline, "transformations")
        assert (
            len(pipeline.transformations) == 2
        )  # UnstructuredTransformation + SentenceSplitter

        # Verify first transformation is UnstructuredTransformation
        assert isinstance(pipeline.transformations[0], UnstructuredTransformation)
        assert pipeline.transformations[0].strategy == ProcessingStrategy.HI_RES

        # Verify second transformation is SentenceSplitter
        from llama_index.core.node_parser import SentenceSplitter

        assert isinstance(pipeline.transformations[1], SentenceSplitter)

        # Verify pipeline configuration
        assert pipeline.cache is not None
        assert pipeline.docstore is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("src.processing.document_processor.partition")
    async def test_process_document_async_success(
        self, mock_partition, mock_settings, mock_unstructured_elements, sample_pdf_path
    ):
        """Test successful document processing."""
        mock_partition.return_value = mock_unstructured_elements

        processor = DocumentProcessor(mock_settings)

        # Mock cache miss
        with (
            patch.object(processor.simple_cache, "get_document", return_value=None),
            patch.object(processor.simple_cache, "store_document", new=AsyncMock()),
        ):
            result = await processor.process_document_async(sample_pdf_path)

        # Verify result structure
        assert isinstance(result, ProcessingResult)
        assert len(result.elements) > 0
        assert result.processing_time > 0
        assert result.strategy_used == ProcessingStrategy.HI_RES
        assert result.document_hash is not None
        assert "file_path" in result.metadata
        assert "element_count" in result.metadata
        assert "pipeline_config" in result.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_document_async_cache_hit(
        self, mock_settings, sample_pdf_path
    ):
        """Test document processing with cache hit."""
        processor = DocumentProcessor(mock_settings)

        # Create mock cached result
        cached_result = ProcessingResult(
            elements=[
                DocumentElement(text="Cached content", category="Text", metadata={})
            ],
            processing_time=0.1,
            strategy_used=ProcessingStrategy.HI_RES,
            metadata={},
            document_hash="cached_hash",
        )

        # Mock cache hit
        with patch.object(
            processor.simple_cache, "get_document", return_value=cached_result
        ):
            result = await processor.process_document_async(sample_pdf_path)

        # Should return cached result
        assert result == cached_result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_document_async_file_not_found(self, mock_settings):
        """Test error handling for non-existent files."""
        processor = DocumentProcessor(mock_settings)

        with pytest.raises(ProcessingError) as exc_info:
            await processor.process_document_async("/non/existent/file.pdf")

        assert "file not found" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_document_async_file_too_large(self, mock_settings, tmp_path):
        """Test error handling for files exceeding size limit."""
        processor = DocumentProcessor(mock_settings)

        # Create large file
        large_file = tmp_path / "large.pdf"
        large_content = b"a" * (101 * 1024 * 1024)  # 101MB
        large_file.write_bytes(large_content)

        with pytest.raises(ProcessingError) as exc_info:
            await processor.process_document_async(large_file)

        assert "exceeds limit" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("src.processing.document_processor.partition")
    async def test_process_document_async_processing_error(
        self, mock_partition, mock_settings, sample_pdf_path
    ):
        """Test error handling during document processing."""
        mock_partition.side_effect = Exception("Processing failed")

        processor = DocumentProcessor(mock_settings)

        # Mock cache miss
        with (
            patch.object(processor.simple_cache, "get_document", return_value=None),
            pytest.raises(ProcessingError) as exc_info,
        ):
            await processor.process_document_async(sample_pdf_path)

        assert "processing failed" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("src.processing.document_processor.partition")
    async def test_process_document_async_corrupted_file_detection(
        self, mock_partition, mock_settings, sample_pdf_path
    ):
        """Test corrupted file detection and specific error handling."""
        mock_partition.side_effect = Exception("File appears to be corrupted")

        processor = DocumentProcessor(mock_settings)

        # Mock cache miss
        with (
            patch.object(processor.simple_cache, "get_document", return_value=None),
            pytest.raises(ProcessingError) as exc_info,
        ):
            await processor.process_document_async(sample_pdf_path)

        assert "corrupted" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_convert_nodes_to_elements(self, mock_settings):
        """Test conversion of LlamaIndex nodes to DocumentElement objects."""
        from llama_index.core.schema import TextNode

        processor = DocumentProcessor(mock_settings)

        # Create mock nodes
        nodes = [
            TextNode(
                text="Sample text content",
                metadata={
                    "element_category": "Title",
                    "page_number": 1,
                    "element_id": "elem_1",
                },
            ),
            TextNode(
                text="More text content",
                metadata={
                    "element_category": "NarrativeText",
                    "page_number": 1,
                    "element_id": "elem_2",
                },
            ),
        ]

        elements = processor._convert_nodes_to_elements(nodes)

        assert len(elements) == 2

        for i, element in enumerate(elements):
            assert isinstance(element, DocumentElement)
            assert element.text == nodes[i].text
            assert element.category == nodes[i].metadata["element_category"]
            assert element.metadata == nodes[i].metadata

    @pytest.mark.unit
    def test_override_config(self, mock_settings):
        """Test configuration override functionality."""
        processor = DocumentProcessor(mock_settings)

        config = {"strategy": "fast", "extract_images": False}
        processor.override_config(config)

        assert hasattr(processor, "_config_override")
        assert processor._config_override == config

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_settings):
        """Test cache clearing functionality."""
        processor = DocumentProcessor(mock_settings)

        # Mock cache objects
        with (
            patch.object(processor.cache, "clear", return_value=None),
            patch.object(
                processor.simple_cache, "clear_cache", new=AsyncMock(return_value=None)
            ),
        ):
            result = await processor.clear_cache()
            assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_cache_error(self, mock_settings):
        """Test cache clearing error handling."""
        processor = DocumentProcessor(mock_settings)

        # Mock cache error
        with patch.object(
            processor.simple_cache,
            "clear_cache",
            new=AsyncMock(side_effect=Exception("Cache error")),
        ):
            result = await processor.clear_cache()
            assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, mock_settings):
        """Test cache statistics retrieval."""
        processor = DocumentProcessor(mock_settings)

        mock_simple_stats = {"hits": 10, "misses": 5, "size_mb": 12.5}

        with patch.object(
            processor.simple_cache,
            "get_cache_stats",
            new=AsyncMock(return_value=mock_simple_stats),
        ):
            stats = await processor.get_cache_stats()

            assert "processor_type" in stats
            assert stats["processor_type"] == "hybrid"
            assert "simple_cache" in stats
            assert stats["simple_cache"] == mock_simple_stats
            assert "llamaindex_cache" in stats
            assert "strategy_mappings" in stats

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_cache_stats_error(self, mock_settings):
        """Test cache statistics error handling."""
        processor = DocumentProcessor(mock_settings)

        with patch.object(
            processor.simple_cache,
            "get_cache_stats",
            new=AsyncMock(side_effect=Exception("Stats error")),
        ):
            stats = await processor.get_cache_stats()

            assert "error" in stats
            assert "processor_type" in stats


class TestFactoryFunctions:
    """Test factory functions for DocumentProcessor."""

    @pytest.mark.unit
    def test_document_processor_constructor(self, mock_settings):
        """Test DocumentProcessor constructor."""
        processor = DocumentProcessor(mock_settings)

        assert isinstance(processor, DocumentProcessor)
        assert processor.settings == mock_settings

    @pytest.mark.unit
    @patch("src.processing.document_processor.app_settings")
    def test_document_processor_constructor_default_settings(self, mock_app_settings):
        """Test constructor with default settings."""
        processor = DocumentProcessor()

        assert isinstance(processor, DocumentProcessor)
        assert processor.settings == mock_app_settings

    @pytest.mark.unit
    def test_document_processor_constructor_compatibility(self, mock_settings):
        """Test DocumentProcessor constructor compatibility."""
        processor = DocumentProcessor(mock_settings)

        assert isinstance(processor, DocumentProcessor)
        assert processor.settings == mock_settings


class TestPerformanceAndBenchmarks:
    """Performance and benchmark tests for DocumentProcessor."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("src.processing.document_processor.partition")
    async def test_performance_target_validation(
        self, mock_partition, mock_settings, mock_unstructured_elements, sample_pdf_path
    ):
        """Test performance target of >1 page/second with hi_res strategy."""
        mock_partition.return_value = mock_unstructured_elements

        processor = DocumentProcessor(mock_settings)

        # Mock cache miss
        with (
            patch.object(processor.simple_cache, "get_document", return_value=None),
            patch.object(processor.simple_cache, "store_document", new=AsyncMock()),
        ):
            start_time = time.time()
            result = await processor.process_document_async(sample_pdf_path)
            processing_time = time.time() - start_time

        # Verify performance target: single page should be processed in <1 second
        assert processing_time < 1.0, (
            f"Processing took {processing_time}s, exceeding 1 page/second target"
        )
        assert result.processing_time > 0
        assert result.processing_time <= processing_time

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("src.processing.document_processor.partition")
    async def test_batch_processing_consistency(
        self, mock_partition, mock_settings, mock_unstructured_elements, tmp_path
    ):
        """Test consistent processing across multiple files."""
        mock_partition.return_value = mock_unstructured_elements

        processor = DocumentProcessor(mock_settings)

        # Create multiple test files
        files = []
        for i in range(3):
            file_path = tmp_path / f"doc_{i}.pdf"
            file_path.write_bytes(b"PDF content")
            files.append(file_path)

        results = []

        # Mock cache miss for all files
        with (
            patch.object(processor.simple_cache, "get_document", return_value=None),
            patch.object(processor.simple_cache, "store_document", new=AsyncMock()),
        ):
            for file_path in files:
                result = await processor.process_document_async(file_path)
                results.append(result)

        # Verify all files processed successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ProcessingResult)
            assert len(result.elements) > 0
            assert result.strategy_used == ProcessingStrategy.HI_RES
