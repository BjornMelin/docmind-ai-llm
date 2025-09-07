"""Focused unit tests for DocumentProcessor (library-first, KISS).

Covers boundary validations, strategy selection, metadata propagation, and a
Hypothesis property test that validates forwarding of chunking parameters to
Unstructured chunkers with deterministic settings.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from hypothesis import HealthCheck, assume, given
from hypothesis import settings as hy_settings
from hypothesis import strategies as st

from src.models.processing import ProcessingStrategy
from src.processing.document_processor import DocumentProcessor


def _mk_settings(
    *,
    chunk_size: int = 1500,
    new_after: int = 1200,
    combine_under: int = 500,
    multipage: bool = True,
    max_mb: float = 50.0,
):
    """Create mock settings object for testing."""
    s = Mock()
    s.processing.chunk_size = chunk_size
    s.processing.new_after_n_chars = new_after
    s.processing.combine_text_under_n_chars = combine_under
    s.processing.multipage_sections = multipage
    s.max_document_size_mb = max_mb
    s.cache_dir = "./cache"
    return s


@pytest.mark.unit
def test_strategy_selection_table():
    """Test strategy selection for different file extensions."""
    proc = DocumentProcessor(_mk_settings())
    mapping = {
        ".pdf": ProcessingStrategy.HI_RES,
        ".docx": ProcessingStrategy.HI_RES,
        ".txt": ProcessingStrategy.FAST,
        ".html": ProcessingStrategy.FAST,
        ".jpg": ProcessingStrategy.OCR_ONLY,
    }
    for ext, expected in mapping.items():
        assert proc.get_strategy_for_file(f"sample{ext}") == expected


@pytest.mark.unit
def test_unsupported_extension_raises():
    """Test that unsupported file extensions raise ValueError."""
    proc = DocumentProcessor(_mk_settings())
    with pytest.raises(ValueError, match="Unsupported file format"):
        proc.get_strategy_for_file("file.unsupported")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_oversized_input_error(tmp_path):
    """Test that oversized documents raise ProcessingError."""
    p = tmp_path / "big.pdf"
    p.write_bytes(b"x" * 2048)  # 2KB
    settings = _mk_settings(max_mb=0.0001)
    proc = DocumentProcessor(settings)
    from src.processing.document_processor import ProcessingError

    with (
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
        pytest.raises(ProcessingError),
    ):
        await proc.process_document_async(p)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_propagation_minimal(tmp_path):
    """Test that metadata is properly propagated through processing."""
    p = tmp_path / "doc.txt"
    p.write_text("hello world")
    settings = _mk_settings()
    proc = DocumentProcessor(settings)

    # Fake transformation pipeline: partition returns one element, chunker echoes
    elem = Mock()
    elem.text = "hello world"
    elem.category = "NarrativeText"
    elem.metadata = Mock()
    elem.metadata.page_number = 1

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
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
        patch("src.processing.document_processor.partition", return_value=[elem]),
        patch("src.processing.document_processor.chunk_by_basic", return_value=[elem]),
    ):
        result = await proc.process_document_async(p)
        assert result.elements
        assert result.elements[0].metadata.get("page_number") == 1


@pytest.mark.unit
@pytest.mark.asyncio
@hy_settings(
    deadline=None,
    max_examples=50,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    max_chars=st.integers(min_value=600, max_value=3000),
    new_after=st.integers(min_value=400, max_value=2500),
    combine_under=st.integers(min_value=100, max_value=1200),
    multipage=st.booleans(),
)
async def test_property_chunking_param_forwarding(
    tmp_path, max_chars: int, new_after: int, combine_under: int, multipage: bool
):
    """Property test that validates chunking parameters are forwarded correctly."""
    # Ensure ordering: combine_under < new_after < max_chars
    assume(combine_under < new_after < max_chars)

    # Use a TXT file instead of PDF to avoid heavy I/O operations (like PDF
    # parsing/rendering) during unit testing, while still validating chunking
    # parameter forwarding behavior.
    test_file = tmp_path / "doc.txt"
    test_file.write_text("x")

    settings = _mk_settings(
        chunk_size=max_chars,
        new_after=new_after,
        combine_under=combine_under,
        multipage=multipage,
    )
    proc = DocumentProcessor(settings)

    parts = []

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
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
        patch("src.processing.document_processor.partition", return_value=parts),
        patch("src.processing.document_processor.chunk_by_basic") as mock_basic,
    ):
        await proc.process_document_async(test_file)
        # Heading-sparse -> basic fallback path
        assert mock_basic.called
        kwargs = mock_basic.call_args.kwargs
        assert kwargs.get("max_characters") == max_chars
        assert kwargs.get("new_after_n_chars") == new_after
