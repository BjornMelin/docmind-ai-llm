"""Document processing integration tests (consolidated).

This file consolidates previous document processing suites into a single,
deterministic, offline integration test module. It contains the content
from test_document_processing_workflows.py (primary coverage) and will be
the definitive location for document processing integration tests.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.processing import (
    ProcessingResult,
)
from src.processing.document_processor import (
    DocumentProcessor,
)
from tests.fixtures.test_settings import MockDocMindSettings as TestDocMindSettings


@pytest.fixture
def test_settings():
    """Provide test settings for document processor."""
    return TestDocMindSettings(
        cache_dir=Path("./test_cache"),
        processing={
            "max_document_size_mb": 100,
            "chunk_size": 1000,
            "chunk_overlap": 100,
        },
    )


@pytest.fixture
def test_documents(tmp_path: Path):
    """Create basic test documents of various types."""
    docs_dir = tmp_path / "test_documents"
    docs_dir.mkdir()

    txt_file = docs_dir / "sample.txt"
    txt_file.write_text(
        "Document Processing Test File\n\nSection A.\n"
        "This is a comprehensive test document for validating the processing "
        "pipeline."
    )

    md_file = docs_dir / "structured.md"
    md_file.write_text("# Structured Test Document\n\n## Overview\nContent")

    large_file = docs_dir / "large_document.txt"
    large_file.write_text("Large section. " * 2000)

    empty_file = docs_dir / "empty.txt"
    empty_file.write_text("")

    unicode_file = docs_dir / "unicode.txt"
    unicode_file.write_text(
        "café résumé naïve α β γ 🚀 你好世界 مرحبا بالعالم", encoding="utf-8"
    )

    return {
        "txt": txt_file,
        "md": md_file,
        "large": large_file,
        "empty": empty_file,
        "unicode": unicode_file,
    }


@pytest.fixture
def mock_cache():
    """Patch SimpleCache to avoid disk IO and ensure determinism."""
    with patch("src.processing.document_processor.SimpleCache") as mock_simple_cache:
        cache_instance = Mock()
        cache_instance.get_document = AsyncMock(return_value=None)
        cache_instance.store_document = AsyncMock(return_value=True)
        cache_instance.clear_cache = AsyncMock(return_value=True)
        cache_instance.get_cache_stats = AsyncMock(
            return_value={"hits": 0, "misses": 1}
        )
        mock_simple_cache.return_value = cache_instance
        yield cache_instance


@pytest.fixture
def mock_unstructured_partition():
    """Patch unstructured partition to return deterministic chunks."""

    def create_mock_elements(file_path: Path) -> list[Mock]:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            content = "Mock content"
        chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
        elements: list[Mock] = []
        for i, chunk in enumerate(chunks[:10]):
            el = Mock()
            el.text = chunk[:1000]
            el.category = "NarrativeText" if len(chunk) > 50 else "Title"
            el.metadata = Mock()
            el.metadata.page_number = (i // 3) + 1
            el.metadata.element_id = f"elem_{i}"
            el.metadata.filename = file_path.name
            elements.append(el)
        return elements or [Mock(text="Default content", category="Text")]

    with patch("src.processing.document_processor.partition") as mock_partition:
        mock_partition.side_effect = create_mock_elements
        yield mock_partition


@pytest.mark.integration
class TestDocumentProcessorCore:
    """Test core DocumentProcessor functionality."""

    def test_init_with_settings(self, test_settings):
        """Test DocumentProcessor initialization with settings."""
        proc = DocumentProcessor(test_settings)
        assert isinstance(proc, DocumentProcessor)

    @pytest.mark.asyncio
    async def test_process_txt_document(
        self, test_settings, test_documents, mock_unstructured_partition, mock_cache
    ):
        """Test processing of TXT documents with mocking."""
        proc = DocumentProcessor(test_settings)
        result = await proc.process_document_async(test_documents["txt"])
        assert isinstance(result, ProcessingResult)
        assert result.elements

    @pytest.mark.asyncio
    async def test_process_md_document(
        self, test_settings, test_documents, mock_unstructured_partition
    ):
        """Test processing of Markdown documents."""
        proc = DocumentProcessor(test_settings)
        result = await proc.process_document_async(test_documents["md"])
        assert isinstance(result, ProcessingResult)
        assert result.elements

    @pytest.mark.asyncio
    async def test_empty_document_handling(self, test_settings, test_documents):
        """Test handling of empty documents gracefully."""
        proc = DocumentProcessor(test_settings)
        # For an empty document, processor should not crash; it returns a result
        # with zero elements when unstructured finds nothing.
        with patch("src.processing.document_processor.partition", return_value=[]):
            result = await proc.process_document_async(test_documents["empty"])
        assert isinstance(result, ProcessingResult)
        assert isinstance(result.elements, list)
        assert len(result.elements) == 0

    @pytest.mark.asyncio
    async def test_large_document_performance(
        self, test_settings, test_documents, mock_unstructured_partition
    ):
        """Test performance handling of large documents."""
        proc = DocumentProcessor(test_settings)
        result = await proc.process_document_async(test_documents["large"])
        assert isinstance(result, ProcessingResult)
        assert result.processing_time >= 0
