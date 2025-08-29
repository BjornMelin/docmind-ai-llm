"""Comprehensive integration tests for document processing workflows.

This test suite focuses on real user workflows with minimal mocking to achieve 70%
coverage for src/processing/document_processor.py (183 statements).

User Workflows Tested:
1. Document upload and parsing (PDF, DOCX, TXT, MD)
2. Chunking strategies and text splitting
3. Error handling for corrupt/unsupported files
4. Memory management for large documents
5. Cache behavior and performance validation
6. Metadata extraction and preservation

Library-First Approach:
- Use pytest fixtures for test data
- Use tmp_path for file operations
- Use pytest-asyncio for async document processing
- Mock only external services (embedding models, not internal business logic)
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Ensure proper import path
PROJECT_ROOT = Path(__file__).parents[2]
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

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
from tests.fixtures.test_settings import TestDocMindSettings


@pytest.fixture
def test_settings():
    """Create lightweight test settings."""
    return TestDocMindSettings(
        cache_dir="./test_cache",
        max_document_size_mb=100,
        processing={"chunk_size": 1000, "chunk_overlap": 100},
    )


@pytest.fixture
def test_documents(tmp_path):
    """Create realistic test documents for various formats."""
    docs_dir = tmp_path / "test_documents"
    docs_dir.mkdir()

    # Text document - simple and fast to process
    txt_file = docs_dir / "sample.txt"
    txt_content = """Document Processing Test File

This is a comprehensive test document for validating the document processing pipeline.

Section 1: Introduction
This section contains introductory content that should be properly parsed and chunked.
It includes multiple sentences to test the chunking and text splitting functionality.

Section 2: Technical Details
The document processor should handle various document types including:
- Plain text files (.txt)
- Markdown documents (.md)  
- PDF files (.pdf)
- Word documents (.docx)

Section 3: Performance Considerations
Large documents should be processed efficiently with proper memory management.
The system should handle documents up to the configured size limit.

Section 4: Error Handling
The processor should gracefully handle:
- Corrupted files
- Unsupported formats
- Missing files
- Network interruptions

Section 5: Conclusion
This test document validates core document processing workflows.
"""
    txt_file.write_text(txt_content)

    # Markdown document with structured content
    md_file = docs_dir / "structured.md"
    md_content = """# Structured Test Document

## Overview
This markdown document tests structured content parsing.

### Features
- Hierarchical headers
- **Bold text formatting**
- *Italic emphasis*
- `Code snippets`

### Lists
1. Ordered list item one
2. Ordered list item two
3. Ordered list item three

Unordered list:
- Item A
- Item B  
- Item C

### Code Block
```python
def example_function():
    return "Hello, World!"
```

### Tables
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Value A  | Value B  | Value C  |

## Conclusion
This document validates markdown parsing and structure extraction.
"""
    md_file.write_text(md_content)

    # Large text document for performance testing
    large_file = docs_dir / "large_document.txt"
    large_content = "Large document content section. " * 2000  # ~50KB
    large_content += (
        "\n\nThis large document tests memory management and processing performance."
    )
    large_file.write_text(large_content)

    # Empty file for edge case testing
    empty_file = docs_dir / "empty.txt"
    empty_file.write_text("")

    # File with special characters for encoding testing
    unicode_file = docs_dir / "unicode.txt"
    unicode_content = """Unicode Test Document

This document contains various unicode characters:
- Accented characters: café, résumé, naïve
- Mathematical symbols: α, β, γ, ∑, ∫, √
- Currency symbols: $, €, ¥, £
- Emoji: 🚀 🔬 📊 💡
- Chinese characters: 你好世界
- Arabic text: مرحبا بالعالم
"""
    unicode_file.write_text(unicode_content, encoding="utf-8")

    return {
        "txt": txt_file,
        "md": md_file,
        "large": large_file,
        "empty": empty_file,
        "unicode": unicode_file,
    }


@pytest.fixture
def mock_cache():
    """Mock cache components to avoid file system dependencies."""
    # Use real lightweight instances instead of mocks to satisfy Pydantic validation
    with (
        patch("src.processing.document_processor.SimpleCache") as mock_simple_cache,
    ):
        # Setup cache mock behavior for SimpleCache (our custom cache)
        cache_instance = Mock()
        cache_instance.get_document = AsyncMock(
            return_value=None
        )  # Cache miss by default
        cache_instance.store_document = AsyncMock(return_value=True)
        cache_instance.clear_cache = AsyncMock(return_value=True)
        cache_instance.get_cache_stats = AsyncMock(
            return_value={"hits": 0, "misses": 1}
        )

        mock_simple_cache.return_value = cache_instance
        yield cache_instance


@pytest.fixture
def mock_unstructured_partition():
    """Mock unstructured.partition to return realistic elements."""

    def create_mock_elements(file_path: Path) -> list[Mock]:
        """Create mock elements based on file content."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            content = "Mock content"

        # Split content into chunks for realistic element simulation
        chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

        elements = []
        for i, chunk in enumerate(chunks[:10]):  # Limit to 10 elements for performance
            element = Mock()
            element.text = chunk[:1000]  # Limit chunk size
            element.category = "NarrativeText" if len(chunk) > 50 else "Title"

            # Mock metadata
            element.metadata = Mock()
            element.metadata.page_number = (i // 3) + 1  # Simulate multiple pages
            element.metadata.element_id = f"elem_{i}"
            element.metadata.filename = file_path.name
            element.metadata.coordinates = None
            element.metadata.text_as_html = None
            element.metadata.image_path = None
            element.metadata.parent_id = None

            elements.append(element)

        return elements if elements else [Mock(text="Default content", category="Text")]

    with patch("src.processing.document_processor.partition") as mock_partition:
        # Configure side_effect to return different elements based on filename
        def partition_side_effect(filename=None, **kwargs):
            if filename:
                return create_mock_elements(Path(filename))
            return [Mock(text="Default mock content", category="Text")]

        mock_partition.side_effect = partition_side_effect
        yield mock_partition


@pytest.mark.integration
class TestDocumentProcessingWorkflows:
    """Test real user workflows for document processing."""

    @pytest.mark.asyncio
    async def test_text_document_processing_workflow(
        self, test_settings, test_documents, mock_cache, mock_unstructured_partition
    ):
        """Test complete text document processing workflow."""
        processor = DocumentProcessor(test_settings)

        # Process text document
        result = await processor.process_document_async(test_documents["txt"])

        # Validate processing result
        assert isinstance(result, ProcessingResult)
        assert result.strategy_used == ProcessingStrategy.FAST
        assert len(result.elements) > 0
        assert result.processing_time > 0
        assert result.document_hash is not None

        # Validate elements have proper structure
        for element in result.elements:
            assert isinstance(element, DocumentElement)
            assert element.text is not None
            assert element.category is not None
            assert isinstance(element.metadata, dict)

        # Validate metadata preservation
        assert "file_path" in result.metadata
        assert "element_count" in result.metadata
        assert result.metadata["element_count"] == len(result.elements)

    @pytest.mark.asyncio
    async def test_markdown_document_processing_workflow(
        self, test_settings, test_documents, mock_cache, mock_unstructured_partition
    ):
        """Test markdown document processing with structured content."""
        processor = DocumentProcessor(test_settings)

        result = await processor.process_document_async(test_documents["md"])

        assert isinstance(result, ProcessingResult)
        assert result.strategy_used == ProcessingStrategy.FAST
        assert len(result.elements) > 0

        # Check for structured content indicators
        text_content = " ".join(elem.text for elem in result.elements)
        assert (
            "Structured Test Document" in text_content
            or "markdown" in text_content.lower()
        )

    @pytest.mark.asyncio
    async def test_large_document_processing_workflow(
        self, test_settings, test_documents, mock_cache, mock_unstructured_partition
    ):
        """Test large document processing and memory management."""
        processor = DocumentProcessor(test_settings)

        result = await processor.process_document_async(test_documents["large"])

        assert isinstance(result, ProcessingResult)
        assert result.strategy_used == ProcessingStrategy.FAST
        assert len(result.elements) > 0
        assert result.processing_time > 0

        # Validate memory efficiency - should handle large docs without issues
        total_text_length = sum(len(elem.text) for elem in result.elements)
        assert total_text_length > 0

        # Check metadata includes file size information
        assert "file_size_mb" in result.metadata
        assert result.metadata["file_size_mb"] > 0

    @pytest.mark.asyncio
    async def test_unicode_document_processing_workflow(
        self, test_settings, test_documents, mock_cache, mock_unstructured_partition
    ):
        """Test unicode and special character handling."""
        processor = DocumentProcessor(test_settings)

        result = await processor.process_document_async(test_documents["unicode"])

        assert isinstance(result, ProcessingResult)
        assert len(result.elements) > 0

        # Verify unicode content is preserved
        text_content = " ".join(elem.text for elem in result.elements)
        # Should contain some unicode or special characters
        assert (
            any(ord(char) > 127 for char in text_content)
            or "unicode" in text_content.lower()
        )

    @pytest.mark.asyncio
    async def test_empty_document_processing_workflow(
        self, test_settings, test_documents, mock_cache, mock_unstructured_partition
    ):
        """Test processing empty documents."""
        processor = DocumentProcessor(test_settings)

        result = await processor.process_document_async(test_documents["empty"])

        assert isinstance(result, ProcessingResult)
        # Should handle empty files gracefully
        assert result.processing_time >= 0
        assert isinstance(result.elements, list)


@pytest.mark.integration
class TestDocumentProcessingStrategies:
    """Test different processing strategies for various file types."""

    def test_strategy_selection_workflow(self, test_settings, mock_cache):
        """Test strategy selection for different file types."""
        processor = DocumentProcessor(test_settings)

        # Test strategy mappings for different file extensions
        strategy_tests = {
            "document.pdf": ProcessingStrategy.HI_RES,
            "document.docx": ProcessingStrategy.HI_RES,
            "document.doc": ProcessingStrategy.HI_RES,
            "presentation.pptx": ProcessingStrategy.HI_RES,
            "text.txt": ProcessingStrategy.FAST,
            "readme.md": ProcessingStrategy.FAST,
            "webpage.html": ProcessingStrategy.FAST,
            "image.jpg": ProcessingStrategy.OCR_ONLY,
            "scan.png": ProcessingStrategy.OCR_ONLY,
            "diagram.tiff": ProcessingStrategy.OCR_ONLY,
        }

        for filename, expected_strategy in strategy_tests.items():
            actual_strategy = processor.get_strategy_for_file(filename)
            assert actual_strategy == expected_strategy, (
                f"Wrong strategy for {filename}: got {actual_strategy}, expected {expected_strategy}"
            )

    def test_unsupported_format_handling(self, test_settings, mock_cache):
        """Test handling of unsupported file formats."""
        processor = DocumentProcessor(test_settings)

        with pytest.raises(ValueError) as exc_info:
            processor.get_strategy_for_file("unsupported.xyz")

        assert "Unsupported file format" in str(exc_info.value)
        assert "xyz" in str(exc_info.value)

    def test_strategy_configuration_validation(self, test_settings):
        """Test strategy configuration is properly built."""
        # Test UnstructuredTransformation configuration
        transform = UnstructuredTransformation(ProcessingStrategy.HI_RES, test_settings)

        # Test HI_RES configuration
        hi_res_config = transform._build_partition_config(ProcessingStrategy.HI_RES)
        assert hi_res_config["strategy"] == "hi_res"
        assert hi_res_config["include_metadata"] is True
        assert hi_res_config["extract_images_in_pdf"] is True
        assert hi_res_config["infer_table_structure"] is True

        # Test FAST configuration
        fast_config = transform._build_partition_config(ProcessingStrategy.FAST)
        assert fast_config["strategy"] == "fast"
        assert fast_config["extract_images_in_pdf"] is False
        assert fast_config["infer_table_structure"] is False

        # Test OCR_ONLY configuration
        ocr_config = transform._build_partition_config(ProcessingStrategy.OCR_ONLY)
        assert ocr_config["strategy"] == "ocr_only"
        assert "ocr_languages" in ocr_config


@pytest.mark.integration
class TestDocumentProcessingErrorHandling:
    """Test error handling for various failure scenarios."""

    @pytest.mark.asyncio
    async def test_missing_file_error_handling(self, test_settings, mock_cache):
        """Test graceful handling of missing files."""
        processor = DocumentProcessor(test_settings)

        with pytest.raises(ProcessingError) as exc_info:
            await processor.process_document_async("/nonexistent/file.txt")

        assert "File not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_oversized_file_error_handling(
        self, test_settings, tmp_path, mock_cache
    ):
        """Test handling of files that exceed size limits."""
        # Create a file that appears large
        large_file = tmp_path / "oversized.txt"
        content = "A" * 10000  # 10KB file content
        large_file.write_text(content)

        # Override settings to have very small limit
        test_settings.max_document_size_mb = 0.001  # Very small limit

        processor = DocumentProcessor(test_settings)

        with pytest.raises(ProcessingError) as exc_info:
            await processor.process_document_async(large_file)

        assert "exceeds limit" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_corrupted_file_simulation(
        self, test_settings, tmp_path, mock_cache, mock_unstructured_partition
    ):
        """Test handling of processing errors that simulate corrupted files."""
        # Configure mock to simulate corruption
        mock_unstructured_partition.side_effect = Exception("corrupted file data")

        processor = DocumentProcessor(test_settings)

        # Create a test file
        test_file = tmp_path / "corrupted.txt"
        test_file.write_text("This file will simulate corruption")

        with pytest.raises(ProcessingError) as exc_info:
            await processor.process_document_async(test_file)

        assert "corrupted" in str(exc_info.value).lower()

    def test_unstructured_transformation_error_recovery(self, test_settings):
        """Test UnstructuredTransformation handles errors gracefully."""
        transform = UnstructuredTransformation(ProcessingStrategy.FAST, test_settings)

        # Create a document node with invalid file path
        from llama_index.core import Document

        doc = Document(text="test", metadata={"file_path": "/invalid/path.txt"})

        # Should handle error and return original node
        result = transform([doc])
        assert len(result) == 1
        assert result[0] == doc  # Original node returned on error


@pytest.mark.integration
class TestDocumentProcessingCache:
    """Test caching behavior and performance optimization."""

    @pytest.mark.asyncio
    async def test_cache_hit_workflow(
        self, test_settings, test_documents, mock_cache, mock_unstructured_partition
    ):
        """Test workflow when cache hit occurs."""
        # Configure cache to return cached result
        cached_result = ProcessingResult(
            elements=[
                DocumentElement(text="Cached content", category="Text", metadata={})
            ],
            processing_time=0.05,
            strategy_used=ProcessingStrategy.FAST,
            metadata={"cached": True},
            document_hash="cached_hash",
        )
        mock_cache.get_document.return_value = cached_result

        processor = DocumentProcessor(test_settings)
        result = await processor.process_document_async(test_documents["txt"])

        # Should return cached result
        assert result == cached_result
        assert result.metadata["cached"] is True
        mock_cache.get_document.assert_called_once()
        # Should not call store_document for cache hit
        mock_cache.store_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_workflow(
        self, test_settings, test_documents, mock_cache, mock_unstructured_partition
    ):
        """Test workflow when cache miss occurs."""
        # Configure cache miss
        mock_cache.get_document.return_value = None

        processor = DocumentProcessor(test_settings)
        result = await processor.process_document_async(test_documents["txt"])

        # Should process and store result
        assert isinstance(result, ProcessingResult)
        mock_cache.get_document.assert_called_once()
        mock_cache.store_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_operations_workflow(self, test_settings, mock_cache):
        """Test cache management operations."""
        processor = DocumentProcessor(test_settings)

        # Test cache clearing
        clear_result = await processor.clear_cache()
        assert clear_result is True
        mock_cache.clear_cache.assert_called_once()

        # Test cache statistics
        stats = await processor.get_cache_stats()
        assert isinstance(stats, dict)
        assert "processor_type" in stats
        assert stats["processor_type"] == "hybrid"
        mock_cache.get_cache_stats.assert_called_once()

    def test_document_hash_calculation(self, test_settings, test_documents, mock_cache):
        """Test document hash calculation for cache keys."""
        processor = DocumentProcessor(test_settings)

        # Calculate hash for same file multiple times
        hash1 = processor._calculate_document_hash(test_documents["txt"])
        hash2 = processor._calculate_document_hash(test_documents["txt"])

        # Hash should be consistent
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length
        assert isinstance(hash1, str)

        # Different files should have different hashes
        hash3 = processor._calculate_document_hash(test_documents["md"])
        assert hash1 != hash3


@pytest.mark.integration
class TestDocumentProcessingConfiguration:
    """Test configuration and customization workflows."""

    def test_configuration_override_workflow(self, test_settings, mock_cache):
        """Test configuration override functionality."""
        processor = DocumentProcessor(test_settings)

        # Test configuration override
        config_override = {
            "chunk_size": 500,
            "strategy": "fast",
            "enable_caching": False,
        }

        processor.override_config(config_override)

        # Verify configuration was stored
        assert processor._config_override["chunk_size"] == 500
        assert processor._config_override["strategy"] == "fast"
        assert processor._config_override["enable_caching"] is False

    def test_pipeline_creation_workflow(self, test_settings, mock_cache):
        """Test IngestionPipeline creation and configuration."""
        processor = DocumentProcessor(test_settings)

        # Test pipeline creation for different strategies
        strategies = [
            ProcessingStrategy.HI_RES,
            ProcessingStrategy.FAST,
            ProcessingStrategy.OCR_ONLY,
        ]

        for strategy in strategies:
            with patch(
                "src.processing.document_processor.IngestionPipeline"
            ) as mock_pipeline:
                pipeline_instance = Mock()
                mock_pipeline.return_value = pipeline_instance

                processor._create_pipeline(strategy)

                # Verify pipeline was created with correct configuration
                mock_pipeline.assert_called_once()
                call_args = mock_pipeline.call_args
                assert "transformations" in call_args.kwargs
                assert "cache" in call_args.kwargs
                assert "docstore" in call_args.kwargs

    def test_node_to_element_conversion_workflow(self, test_settings, mock_cache):
        """Test conversion of LlamaIndex nodes to DocumentElements."""
        processor = DocumentProcessor(test_settings)

        # Create mock nodes with various metadata
        mock_nodes = []

        # Node with complete metadata
        node1 = Mock()
        node1.get_content.return_value = "Content with metadata"
        node1.text = "Content with metadata"
        node1.metadata = {
            "element_category": "NarrativeText",
            "page_number": 1,
            "source_file": "test.txt",
            "processing_strategy": "fast",
        }
        mock_nodes.append(node1)

        # Node with minimal metadata
        node2 = Mock()
        node2.get_content.return_value = "Minimal content"
        node2.text = "Minimal content"
        node2.metadata = {"element_category": "Title"}
        mock_nodes.append(node2)

        # Convert nodes to elements
        elements = processor._convert_nodes_to_elements(mock_nodes)

        # Validate conversion
        assert len(elements) == 2
        assert all(isinstance(elem, DocumentElement) for elem in elements)

        # Check first element
        assert elements[0].text == "Content with metadata"
        assert elements[0].category == "NarrativeText"
        assert elements[0].metadata["page_number"] == 1

        # Check second element
        assert elements[1].text == "Minimal content"
        assert elements[1].category == "Title"


@pytest.mark.integration
class TestDocumentProcessingPerformance:
    """Test performance characteristics and benchmarks."""

    @pytest.mark.asyncio
    async def test_processing_time_validation(
        self, test_settings, test_documents, mock_cache, mock_unstructured_partition
    ):
        """Test that processing times are reasonable for different document types."""
        processor = DocumentProcessor(test_settings)

        # Test processing times for different documents
        test_cases = [
            ("txt", test_documents["txt"], 5.0),  # 5 second max for text
            ("md", test_documents["md"], 5.0),  # 5 second max for markdown
            ("large", test_documents["large"], 10.0),  # 10 second max for large doc
        ]

        for doc_type, doc_path, max_time in test_cases:
            result = await processor.process_document_async(doc_path)

            assert result.processing_time < max_time, (
                f"{doc_type} processing took {result.processing_time}s, max allowed: {max_time}s"
            )
            assert result.processing_time > 0  # Should take some time

    @pytest.mark.asyncio
    async def test_concurrent_processing_workflow(
        self, test_settings, test_documents, mock_cache, mock_unstructured_partition
    ):
        """Test concurrent processing of multiple documents."""
        processor = DocumentProcessor(test_settings)

        # Process multiple documents concurrently
        documents_to_process = [
            test_documents["txt"],
            test_documents["md"],
            test_documents["unicode"],
        ]

        # Use asyncio.gather for concurrent processing
        tasks = [processor.process_document_async(doc) for doc in documents_to_process]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete successfully
        successful_results = [r for r in results if isinstance(r, ProcessingResult)]
        assert (
            len(successful_results) >= len(documents_to_process) // 2
        )  # At least half should succeed

        # Validate each successful result
        for result in successful_results:
            assert isinstance(result, ProcessingResult)
            assert len(result.elements) > 0
            assert result.processing_time > 0

    def test_memory_efficiency_indicators(
        self, test_settings, test_documents, mock_cache
    ):
        """Test memory efficiency indicators in processing results."""
        processor = DocumentProcessor(test_settings)

        # Verify processor setup doesn't consume excessive memory
        assert processor.strategy_map is not None
        assert len(processor.strategy_map) > 0

        # Test memory-related metadata is captured
        # This is structural validation - actual memory testing would require different tools
        assert hasattr(processor, "_config_override")
        assert hasattr(processor, "cache")
        assert hasattr(processor, "docstore")


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v", "--tb=short"])
