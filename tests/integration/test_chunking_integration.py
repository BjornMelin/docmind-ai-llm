"""Integration tests for Unstructured-first chunking in DocumentProcessor.

These tests validate that:
- Title-rich inputs trigger by_title chunking and respect section boundaries.
- Heading-sparse inputs fall back to basic chunking.
- Tables are isolated and never merged with narrative text.
- Multipage sections flag is passed through.

All heavy library calls are patched; we assert behavior via calls/metadata.
"""

from unittest.mock import Mock, patch

import pytest

from src.models.processing import ProcessingResult, ProcessingStrategy
from src.processing.document_processor import DocumentProcessor


def _mk_elem(text: str, category: str = "NarrativeText", **md):
    """Create a mock document element for testing."""
    e = Mock()
    e.text = text
    e.category = category
    e.metadata = Mock()
    for k, v in md.items():
        setattr(e.metadata, k, v)
    return e


@pytest.mark.asyncio
async def test_by_title_section_boundaries(tmp_path):
    """Test that by_title chunking respects section boundaries."""
    test_file = tmp_path / "doc.pdf"
    test_file.write_text("x")

    # Partition returns elements with titles
    parts = [
        _mk_elem("Intro", "Title", page_number=1),
        _mk_elem("para1", "NarrativeText", page_number=1),
        _mk_elem("Section A", "Title", page_number=1),
        _mk_elem("para2", "NarrativeText", page_number=1),
        _mk_elem("Section B", "Title", page_number=2),
        _mk_elem("para3", "NarrativeText", page_number=2),
    ]

    # Chunker returns one chunk per title section
    def _mk_chunk(
        text: str, section_title: str, page: int, category: str = "CompositeElement"
    ):
        """Create a mock chunk element for testing."""
        c = Mock()
        c.text = text
        c.category = category
        c.metadata = Mock()
        c.metadata.page_number = page
        c.metadata.section_title = section_title
        return c

    chunks = [
        _mk_chunk("Intro\n\npara1", "Intro", 1),
        _mk_chunk("Section A\n\npara2", "Section A", 1),
        _mk_chunk("Section B\n\npara3", "Section B", 2),
    ]

    class _FakePipeline:
        """Mock pipeline class for testing document processing."""

        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = type("C", (), {"hits": 0, "misses": 0})()

        def run(self, documents=None, show_progress=False):
            nodes = documents or []
            for t in self.transformations:
                nodes = t(nodes)
            return nodes

    with (
        patch("src.processing.document_processor.partition", return_value=parts),
        patch(
            "src.processing.document_processor.is_unstructured_like",
            return_value=True,
        ),
        patch(
            "src.processing.document_processor.chunk_by_title", return_value=chunks
        ) as mock_chunk,
        patch("src.processing.document_processor.chunk_by_basic") as mock_basic,
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
    ):
        processor = DocumentProcessor()
        result = await processor.process_document_async(test_file)

        assert isinstance(result, ProcessingResult)
        assert result.strategy_used == ProcessingStrategy.HI_RES
        # We expect 3 chunks back as elements
        assert len(result.elements) == 3
        # Ensure by_title used, not basic fallback
        mock_chunk.assert_called_once()
        mock_basic.assert_not_called()
        # Ensure multipage flag propagated
        kwargs = mock_chunk.call_args.kwargs
        assert "multipage_sections" in kwargs
        assert kwargs["multipage_sections"] is True


@pytest.mark.asyncio
async def test_basic_fallback_heading_sparse(tmp_path):
    """Test that heading-sparse documents fall back to basic chunking."""
    test_file = tmp_path / "doc.txt"
    test_file.write_text("x")

    # No Title elements -> fallback to basic
    parts = [_mk_elem(f"para{i}") for i in range(5)]
    basic_chunks = [
        _mk_elem("para0 para1", "CompositeElement"),
        _mk_elem("para2 para3 para4", "CompositeElement"),
    ]

    class _FakePipeline:
        """Mock pipeline class for testing document processing."""

        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = type("C", (), {"hits": 0, "misses": 0})()

        def run(self, documents=None, show_progress=False):
            nodes = documents or []
            for t in self.transformations:
                nodes = t(nodes)
            return nodes

    with (
        patch("src.processing.document_processor.partition", return_value=parts),
        patch(
            "src.processing.document_processor.is_unstructured_like",
            return_value=True,
        ),
        patch("src.processing.document_processor.chunk_by_title") as mock_title,
        patch(
            "src.processing.document_processor.chunk_by_basic",
            return_value=basic_chunks,
        ) as mock_basic,
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
    ):
        processor = DocumentProcessor()
        result = await processor.process_document_async(test_file)

        assert isinstance(result, ProcessingResult)
        assert result.strategy_used == ProcessingStrategy.FAST
        mock_basic.assert_called_once()
        mock_title.assert_not_called()
        assert len(result.elements) == 2


@pytest.mark.asyncio
async def test_table_isolation(tmp_path):
    """Test that tables are isolated and never merged with narrative text."""
    test_file = tmp_path / "doc.pdf"
    test_file.write_text("x")

    # Partition returns a table and some text; by_title should keep table isolated
    parts = [
        _mk_elem("Intro", "Title", page_number=1),
        _mk_elem("table data", "Table", page_number=1),
        _mk_elem("para1", "NarrativeText", page_number=1),
    ]

    # Simulate chunker yielding a Table chunk as-is and a text chunk
    table_chunk = _mk_elem("table data", "Table", page_number=1)
    text_chunk = _mk_elem("Intro\n\npara1", "CompositeElement", page_number=1)

    class _FakePipeline:
        """Mock pipeline class for testing document processing."""

        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = type("C", (), {"hits": 0, "misses": 0})()

        def run(self, documents=None, show_progress=False):
            nodes = documents or []
            for t in self.transformations:
                nodes = t(nodes)
            return nodes

    with (
        patch("src.processing.document_processor.partition", return_value=parts),
        patch(
            "src.processing.document_processor.is_unstructured_like",
            return_value=True,
        ),
        patch(
            "src.processing.document_processor.chunk_by_title",
            return_value=[table_chunk, text_chunk],
        ),
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
    ):
        processor = DocumentProcessor()
        result = await processor.process_document_async(test_file)

        # Expect both chunks back; table preserved
        assert any(e.category == "Table" for e in result.elements)
        assert any(e.category == "CompositeElement" for e in result.elements)


@pytest.mark.asyncio
async def test_clustered_titles_form_separate_sections(tmp_path):
    """Heading cluster still yields distinct chunks by title."""
    test_file = tmp_path / "doc.pdf"
    test_file.write_text("x")

    parts = [
        _mk_elem("A", "Title", page_number=1),
        _mk_elem("B", "Title", page_number=1),
        _mk_elem("C", "Title", page_number=1),
        _mk_elem("para", "NarrativeText", page_number=1),
    ]

    chunks = [
        _mk_elem("A", "CompositeElement", page_number=1),
        _mk_elem("B", "CompositeElement", page_number=1),
        _mk_elem("C\n\npara", "CompositeElement", page_number=1),
    ]

    class _FakePipeline:
        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = type("C", (), {"hits": 0, "misses": 0})()

        def run(self, documents=None, show_progress=False):
            nodes = documents or []
            for t in self.transformations:
                nodes = t(nodes)
            return nodes

    with (
        patch("src.processing.document_processor.partition", return_value=parts),
        patch(
            "src.processing.document_processor.is_unstructured_like",
            return_value=True,
        ),
        patch("src.processing.document_processor.chunk_by_title", return_value=chunks),
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
    ):
        processor = DocumentProcessor()
        result = await processor.process_document_async(test_file)
        assert len(result.elements) == 3


@pytest.mark.asyncio
async def test_frequent_small_headings_fallbacks_to_basic(tmp_path):
    """Many tiny headings should still allow basic fallback by heuristic."""
    test_file = tmp_path / "doc.txt"
    test_file.write_text("x")

    parts = sum(
        (
            [[_mk_elem(f"H{i}", "Title")], [_mk_elem("x", "NarrativeText")]]
            for i in range(2)
        ),
        [],
    )

    basic_chunks = [
        _mk_elem("H0 x", "CompositeElement"),
        _mk_elem("H1 x", "CompositeElement"),
    ]

    class _FakePipeline:
        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = type("C", (), {"hits": 0, "misses": 0})()

        def run(self, documents=None, show_progress=False):
            nodes = documents or []
            for t in self.transformations:
                nodes = t(nodes)
            return nodes

    with (
        patch("src.processing.document_processor.partition", return_value=parts),
        patch(
            "src.processing.document_processor.is_unstructured_like",
            return_value=True,
        ),
        patch("src.processing.document_processor.chunk_by_title"),
        patch(
            "src.processing.document_processor.chunk_by_basic",
            return_value=basic_chunks,
        ) as mock_basic,
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
    ):
        processor = DocumentProcessor()
        result = await processor.process_document_async(test_file)
        assert mock_basic.called
        assert len(result.elements) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("multipage", [True, False])
async def test_multipage_sections_propagation(tmp_path, multipage):
    """Ensure multipage_sections setting is forwarded to chunk_by_title."""
    test_file = tmp_path / "doc.pdf"
    test_file.write_text("x")

    parts = [
        _mk_elem("Heading", "Title", page_number=1),
        _mk_elem("para", "NarrativeText", page_number=1),
        _mk_elem("Heading 2", "Title", page_number=2),
    ]

    class _FakePipeline:
        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = type("C", (), {"hits": 0, "misses": 0})()

        def run(self, documents=None, show_progress=False):
            nodes = documents or []
            for t in self.transformations:
                nodes = t(nodes)
            return nodes

    # Prepare custom settings with desired multipage flag
    settings = Mock()
    settings.processing.chunk_size = 1500
    settings.processing.new_after_n_chars = 1200
    settings.processing.combine_text_under_n_chars = 500
    settings.processing.multipage_sections = multipage
    settings.max_document_size_mb = 100
    settings.cache_dir = tmp_path / "cache"

    with (
        patch("src.processing.document_processor.partition", return_value=parts),
        patch(
            "src.processing.document_processor.is_unstructured_like",
            return_value=True,
        ),
        patch("src.processing.document_processor.chunk_by_title") as mock_title,
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
    ):
        processor = DocumentProcessor(settings=settings)
        await processor.process_document_async(test_file)

        assert mock_title.called
        kwargs = mock_title.call_args.kwargs
        assert kwargs.get("multipage_sections") is multipage


@pytest.mark.asyncio
@pytest.mark.parametrize("combine_under", [200, 800])
async def test_combine_text_under_n_chars_forwarded(tmp_path, combine_under):
    """Ensure combine_text_under_n_chars is forwarded to chunk_by_title."""
    test_file = tmp_path / "doc.pdf"
    test_file.write_text("x")

    parts = [
        _mk_elem("Heading", "Title", page_number=1),
        _mk_elem("para1", "NarrativeText", page_number=1),
        _mk_elem("para2", "NarrativeText", page_number=1),
    ]

    class _FakePipeline:
        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = type("C", (), {"hits": 0, "misses": 0})()

        def run(self, documents=None, show_progress=False):
            nodes = documents or []
            for t in self.transformations:
                nodes = t(nodes)
            return nodes

    settings = Mock()
    settings.processing.chunk_size = 1500
    settings.processing.new_after_n_chars = 1200
    settings.processing.combine_text_under_n_chars = combine_under
    settings.processing.multipage_sections = True
    settings.max_document_size_mb = 100

    with (
        patch("src.processing.document_processor.partition", return_value=parts),
        patch(
            "src.processing.document_processor.is_unstructured_like",
            return_value=True,
        ),
        patch("src.processing.document_processor.chunk_by_title") as mock_title,
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
    ):
        processor = DocumentProcessor(settings=settings)
        await processor.process_document_async(test_file)

        assert mock_title.called
        kwargs = mock_title.call_args.kwargs
        assert kwargs.get("combine_text_under_n_chars") == combine_under
