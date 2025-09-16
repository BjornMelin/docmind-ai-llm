"""Comprehensive test suite for DocumentProcessor class.

This test suite covers the hybrid document processor that combines unstructured.io
with LlamaIndex IngestionPipeline, focusing on strategy-based processing,
caching, error handling, and format support.

Note: Minimal private helper usage is intentional to validate conversion and
hashing behavior where no public seam exists yet.
"""

# pylint: disable=protected-access

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from llama_index.core import Document

from src.config.settings import HashingConfig
from src.models.processing import (
    DocumentElement,
    ProcessingError,
    ProcessingResult,
    ProcessingStrategy,
)
from src.processing.document_processor import (
    DocumentProcessor,
    UnstructuredTransformation,
)


@pytest.fixture
def mock_settings():
    """Mock DocMind settings for testing."""
    mock_settings = Mock()
    mock_settings.processing = Mock()
    mock_settings.processing.chunk_size = 1000
    mock_settings.processing.chunk_overlap = 100
    mock_settings.processing.pipeline_version = "test"
    mock_settings.cache = SimpleNamespace(
        backend="memory",
        dir=Path("./test_cache"),
        filename="cache.duckdb",
    )
    mock_settings.cache_version = "1"
    mock_settings.max_document_size_mb = 50
    mock_settings.hashing = HashingConfig().model_copy(
        update={
            "metadata_keys": [
                "content_type",
                "language",
                "source",
                "source_path",
                "tenant_id",
                "size_bytes",
            ]
        }
    )
    mock_settings.tenant_id = None
    return mock_settings


@pytest.fixture
def sample_document_path(tmp_path):
    """Create a sample document file for testing."""
    doc_path = tmp_path / "test_document.pdf"
    doc_path.write_text("Sample PDF content for testing")
    return doc_path


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    txt_path = tmp_path / "test_document.txt"
    txt_path.write_text(
        "This is sample text content for testing the document processor."
    )
    return txt_path


@pytest.fixture
def mock_unstructured_element():
    """Mock unstructured element for testing."""
    element = Mock()
    element.text = "Sample element text content"
    element.category = "NarrativeText"
    element.metadata = Mock()
    element.metadata.page_number = 1
    element.metadata.element_id = "elem_1"
    element.metadata.filename = "test.pdf"
    return element


@pytest.mark.unit
class TestDocumentProcessor:
    """Test DocumentProcessor functionality."""

    def test_initialization_with_settings(self, mock_settings):
        """Test processor initialization with provided settings."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)
            assert processor.settings == mock_settings
            assert processor.strategy_map is not None
            assert len(processor.strategy_map) > 0

    def test_initialization_without_settings(self):
        """Test processor initialization without settings uses defaults."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            # When no settings passed, should create with whatever is available
            processor = DocumentProcessor(None)
            assert hasattr(processor, "strategy_map")
            assert len(processor.strategy_map) > 0

    def test_strategy_mapping_coverage(self, mock_settings):
        """Test that all common document formats have strategy mappings."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)

            # Test key file extensions are mapped
            expected_extensions = [
                ".pdf",
                ".docx",
                ".doc",
                ".pptx",
                ".txt",
                ".html",
                ".jpg",
                ".png",
            ]

            for ext in expected_extensions:
                assert ext in processor.strategy_map
                assert isinstance(processor.strategy_map[ext], ProcessingStrategy)

    def test_get_strategy_for_file_pdf(self, mock_settings):
        """Test strategy selection for PDF files."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)
            strategy = processor.get_strategy_for_file("document.pdf")
            assert strategy == ProcessingStrategy.HI_RES

    def test_get_strategy_for_file_text(self, mock_settings):
        """Test strategy selection for text files."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)
            strategy = processor.get_strategy_for_file("document.txt")
            assert strategy == ProcessingStrategy.FAST

    def test_get_strategy_for_file_image(self, mock_settings):
        """Test strategy selection for image files."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)
            strategy = processor.get_strategy_for_file("image.jpg")
            assert strategy == ProcessingStrategy.OCR_ONLY

    def test_get_strategy_for_unsupported_file(self, mock_settings):
        """Test error handling for unsupported file formats."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)

            with pytest.raises(ValueError, match="Unsupported file format") as excinfo:
                processor.get_strategy_for_file("document.xyz")

            assert "Unsupported file format" in str(excinfo.value)
            assert "xyz" in str(excinfo.value)

    def test_compute_document_hashes(self, mock_settings, sample_text_file):
        """Test document hash bundle generation."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)
            bundle = processor._compute_document_hashes(sample_text_file)

            assert bundle.raw_sha256
            assert len(bundle.raw_sha256) == 64
            assert bundle.canonical_hmac_sha256
            assert len(bundle.canonical_hmac_sha256) == 64
            assert (
                bundle.canonicalization_version
                == mock_settings.hashing.canonicalization_version
            )
            assert (
                bundle.hmac_secret_version == mock_settings.hashing.hmac_secret_version
            )

            # Hash bundle should be consistent between invocations
            bundle_repeat = processor._compute_document_hashes(sample_text_file)
            assert bundle == bundle_repeat

    def test_convert_nodes_to_elements(self, mock_settings):
        """Test conversion of LlamaIndex nodes to DocumentElements."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)

            # Create mock nodes
            mock_node = Mock()
            mock_node.get_content.return_value = "Test content"
            mock_node.text = "Test content"
            mock_node.metadata = {
                "element_category": "NarrativeText",
                "page_number": 1,
                "source_file": "test.pdf",
            }

            elements = processor._convert_nodes_to_elements([mock_node])

            assert len(elements) == 1
            assert isinstance(elements[0], DocumentElement)
            assert elements[0].text == "Test content"
            assert elements[0].category == "NarrativeText"

    @pytest.mark.asyncio
    async def test_process_document_async_file_not_found(self, mock_settings):
        """Test error handling for non-existent files."""
        from src.processing.document_processor import (
            ProcessingError as DPProcessingError,
        )

        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
            patch("asyncio.sleep", new=AsyncMock(return_value=None)),
        ):
            # Mock the getattr calls for settings access
            mock_settings.max_document_size_mb = 50

            processor = DocumentProcessor(mock_settings)

            with pytest.raises((ProcessingError, DPProcessingError)) as excinfo:
                await processor.process_document_async("non_existent_file.pdf")
            assert "File not found" in str(excinfo.value)

            assert "File not found" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_process_document_async_file_too_large(self, mock_settings, tmp_path):
        """Test error handling for files exceeding size limit."""
        # Create a file that appears large based on settings
        large_file = tmp_path / "large_file.pdf"
        large_file.write_text("A" * 2000)  # 2KB file with very small limit

        from src.processing.document_processor import (
            ProcessingError as DPProcessingError,
        )

        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
            patch("asyncio.sleep", new=AsyncMock(return_value=None)),
        ):
            # Set a very small size limit to trigger the error
            mock_settings.max_document_size_mb = 0.0001
            processor = DocumentProcessor(mock_settings)

            with pytest.raises((ProcessingError, DPProcessingError)) as excinfo:
                await processor.process_document_async(large_file)
            assert "exceeds limit" in str(excinfo.value)

            assert "exceeds limit" in str(excinfo.value)

    # Direct cache-hit path removed per ADR-030; pipeline caching covered elsewhere.

    @pytest.mark.asyncio
    async def test_process_document_async_success(
        self, mock_settings, sample_text_file
    ):
        """Test successful document processing without cache."""
        [
            DocumentElement(
                text="Processed content", category="NarrativeText", metadata={"page": 1}
            )
        ]

        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            # Mock pipeline processing
            mock_nodes = [Mock()]
            mock_nodes[0].get_content.return_value = "Processed content"
            mock_nodes[0].metadata = {"element_category": "NarrativeText", "page": 1}
            mock_to_thread.return_value = mock_nodes

            processor = DocumentProcessor(mock_settings)

            # Mock the pipeline creation
            with patch.object(processor, "_create_pipeline") as mock_create_pipeline:
                mock_pipeline = Mock()
                mock_create_pipeline.return_value = mock_pipeline
                mock_pipeline.run.return_value = mock_nodes
                # Provide transformations list so len() works in processor
                mock_pipeline.transformations = [Mock(), Mock()]

                result = await processor.process_document_async(sample_text_file)

                assert isinstance(result, ProcessingResult)
                assert len(result.elements) == 1
                assert result.elements[0].text == "Processed content"
                assert result.strategy_used == ProcessingStrategy.FAST

    @pytest.mark.asyncio
    async def test_clear_cache_success(self, mock_settings):
        """Test cache clearing success."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)
            result = await processor.clear_cache()
            assert result is True

    @pytest.mark.asyncio
    async def test_get_cache_stats_success(self, mock_settings):
        """Test cache statistics retrieval."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)
            stats = await processor.get_cache_stats()

            assert isinstance(stats, dict)
            assert stats.get("processor_type") == "hybrid"
            assert "llamaindex_cache" in stats
            assert stats["llamaindex_cache"].get("cache_type") == "duckdb_kvstore"

    def test_override_config(self, mock_settings):
        """Test configuration override functionality."""
        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor(mock_settings)
            config_override = {"test_param": "test_value"}

            processor.override_config(config_override)
            assert processor._config_override["test_param"] == "test_value"


@pytest.mark.unit
class TestUnstructuredTransformation:
    """Test UnstructuredTransformation component."""

    @pytest.fixture
    def mock_transformation(self, mock_settings):
        """Create a mock UnstructuredTransformation for testing."""
        return UnstructuredTransformation(ProcessingStrategy.FAST, mock_settings)

    def test_transformation_initialization(self, mock_settings):
        """Test UnstructuredTransformation initialization."""
        transform = UnstructuredTransformation(ProcessingStrategy.HI_RES, mock_settings)
        assert transform.strategy == ProcessingStrategy.HI_RES
        assert transform.settings == mock_settings

    def test_extract_file_path_from_document(
        self, mock_transformation, sample_text_file
    ):
        """Test file path extraction from Document metadata."""
        # Create document with file_path in metadata
        doc = Document(text="test", metadata={"file_path": str(sample_text_file)})

        file_path = mock_transformation._extract_file_path(doc)
        assert file_path == sample_text_file

    def test_extract_file_path_no_metadata(self, mock_transformation):
        """Test file path extraction when no metadata exists."""
        doc = Document(text="test", metadata={})

        file_path = mock_transformation._extract_file_path(doc)
        assert file_path is None

    def test_build_partition_config_hi_res(self, mock_transformation):
        """Test partition config for HI_RES strategy."""
        transform = UnstructuredTransformation(
            ProcessingStrategy.HI_RES, mock_transformation.settings
        )
        config = transform._build_partition_config(ProcessingStrategy.HI_RES)

        assert config["strategy"] == "hi_res"
        assert config["include_metadata"] is True
        assert config["extract_images_in_pdf"] is True
        assert config["infer_table_structure"] is True

    def test_build_partition_config_fast(self, mock_transformation):
        """Test partition config for FAST strategy."""
        transform = UnstructuredTransformation(
            ProcessingStrategy.FAST, mock_transformation.settings
        )
        config = transform._build_partition_config(ProcessingStrategy.FAST)

        assert config["strategy"] == "fast"
        assert config["extract_images_in_pdf"] is False
        assert config["infer_table_structure"] is False

    def test_build_partition_config_ocr_only(self, mock_transformation):
        """Test partition config for OCR_ONLY strategy."""
        transform = UnstructuredTransformation(
            ProcessingStrategy.OCR_ONLY, mock_transformation.settings
        )
        config = transform._build_partition_config(ProcessingStrategy.OCR_ONLY)

        assert config["strategy"] == "ocr_only"
        assert config["extract_images_in_pdf"] is True
        assert "ocr_languages" in config

    def test_convert_elements_to_nodes(
        self, mock_transformation, mock_unstructured_element, sample_text_file
    ):
        """Test conversion of unstructured elements to Document nodes."""
        original_doc = Document(
            text="original", metadata={"source": str(sample_text_file)}
        )

        nodes = mock_transformation._convert_elements_to_nodes(
            [mock_unstructured_element], original_doc, sample_text_file
        )

        assert len(nodes) == 1
        assert isinstance(nodes[0], Document)
        assert nodes[0].text == "Sample element text content"
        assert "element_category" in nodes[0].metadata
        assert "processing_strategy" in nodes[0].metadata

    def test_transform_nodes_with_non_document(self, mock_transformation):
        """Test transformation passes through non-Document nodes unchanged."""
        mock_node = Mock()
        mock_node.__class__.__name__ = "TextNode"  # Not a Document

        result = mock_transformation([mock_node])
        assert len(result) == 1
        assert result[0] == mock_node

    @patch("src.processing.document_processor.partition")
    def test_transform_nodes_with_document(
        self, mock_partition, mock_transformation, sample_text_file
    ):
        """Test transformation of Document nodes."""
        # Mock unstructured partition
        mock_partition.return_value = [mock_unstructured_element]

        # Create document with valid file path
        doc = Document(text="test", metadata={"file_path": str(sample_text_file)})

        result = mock_transformation([doc])

        # Should return transformed nodes
        assert len(result) >= 1
        mock_partition.assert_called_once()

    def test_transform_nodes_error_handling(self, mock_transformation):
        """Test error handling in transformation."""
        # Create document with invalid file path
        doc = Document(text="test", metadata={"file_path": "/nonexistent/file.pdf"})

        result = mock_transformation([doc])

        # Should return original node on error
        assert len(result) == 1
        assert result[0] == doc


@pytest.mark.integration
class TestDocumentProcessorIntegration:
    """Integration tests for DocumentProcessor with real file operations."""

    @pytest.mark.asyncio
    async def test_process_text_file_integration(self, tmp_path):
        """Integration test for processing a real text file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = (
            "This is a test document.\n\n"
            "It has multiple paragraphs.\n\n"
            "Each paragraph contains test content."
        )
        test_file.write_text(test_content)

        # Mock heavy dependencies but test real file processing
        with (
            patch("src.processing.document_processor.partition") as mock_partition,
        ):
            # Mock unstructured partition
            mock_element = Mock()
            mock_element.text = test_content
            mock_element.category = "NarrativeText"
            mock_element.metadata = Mock()
            mock_element.metadata.page_number = 1
            mock_partition.return_value = [mock_element]

            from src.config.settings import DocMindSettings

            processor = DocumentProcessor(
                DocMindSettings(debug=True, enable_gpu_acceleration=False)
            )
            result = await processor.process_document_async(test_file)

            assert isinstance(result, ProcessingResult)
            assert result.strategy_used == ProcessingStrategy.FAST
            assert len(result.elements) > 0
            assert result.processing_time > 0

    def test_batch_strategy_detection(self, tmp_path):
        """Test strategy detection for multiple file types."""
        # Create various file types
        files = {
            "document.pdf": ProcessingStrategy.HI_RES,
            "document.docx": ProcessingStrategy.HI_RES,
            "document.txt": ProcessingStrategy.FAST,
            "document.html": ProcessingStrategy.FAST,
            "image.jpg": ProcessingStrategy.OCR_ONLY,
            "image.png": ProcessingStrategy.OCR_ONLY,
        }

        with (
            patch("src.processing.document_processor.IngestionCache"),
            patch("src.processing.document_processor.SimpleDocumentStore"),
        ):
            processor = DocumentProcessor()

            for filename, expected_strategy in files.items():
                file_path = tmp_path / filename
                file_path.write_text("test content")

                strategy = processor.get_strategy_for_file(file_path)
                assert strategy == expected_strategy, f"Wrong strategy for {filename}"
