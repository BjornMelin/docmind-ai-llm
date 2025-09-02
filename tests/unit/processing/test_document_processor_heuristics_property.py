"""Property-based tests for DocumentProcessor heuristics.

Verifies by_title vs basic fallback based on synthetic title densities.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hy_settings
from hypothesis import strategies as st

from src.processing.document_processor import DocumentProcessor


def _mk_elem(text: str, category: str) -> object:
    """Create a minimal element-like object with metadata."""
    e = type("E", (), {})()
    e.text = text
    e.category = category
    e.metadata = type("M", (), {"page_number": 1})()
    return e


@pytest.mark.unit
@pytest.mark.asyncio
@hy_settings(
    deadline=None,
    max_examples=50,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    titles=st.integers(min_value=0, max_value=20),
    paras=st.integers(min_value=0, max_value=80),
)
async def test_by_title_heuristic_titles_trigger_chunk_by_title(
    titles: int, paras: int, tmp_path
) -> None:
    """When title density is high enough, by_title should be used over basic.

    We synthesize elements with a specified count of Title and NarrativeText
    and assert which chunker is invoked.
    """
    test_file = tmp_path / "doc.pdf"
    test_file.write_text("x")

    # Build synthetic parts
    parts = [_mk_elem(f"Title {i}", "Title") for i in range(titles)] + [
        _mk_elem(f"para {j}", "NarrativeText") for j in range(paras)
    ]

    # Guard: ensure at least one element to avoid degenerate cases
    added_title = False
    if not parts:
        parts = [_mk_elem("Title 0", "Title")]
        added_title = True

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
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
        patch("src.processing.document_processor.SimpleCache") as mock_simple_cache,
        patch("src.processing.document_processor.chunk_by_title") as mock_title,
        patch("src.processing.document_processor.chunk_by_basic") as mock_basic,
    ):
        mock_simple_cache.return_value.get_document = AsyncMock(return_value=None)
        mock_simple_cache.return_value.store_document = AsyncMock()

        processor = DocumentProcessor()
        await processor.process_document_async(test_file)

        # Heuristic in processor uses title count and density; compute effective counts
        eff_titles = titles if not added_title else 1
        eff_paras = paras if not added_title else 0
        if eff_titles >= 3 or (eff_titles / max(1, eff_titles + eff_paras)) >= 0.05:
            assert mock_title.called
            assert not mock_basic.called
        else:
            assert mock_basic.called
