"""Ensure deterministic IDs remain unique for blank-text elements on a page.

Creates two unstructured-like elements with empty text and the same page number
and verifies that the processor assigns distinct node_ids to avoid collisions.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.processing.document_processor import DocumentProcessor


@pytest.mark.unit
@pytest.mark.asyncio
async def test_blank_text_elements_receive_unique_ids(tmp_path: Path) -> None:
    """Two blank-text elements on the same page get distinct deterministic IDs."""
    # Two elements with empty text on the same page and no element_id
    e1 = Mock()
    e1.text = ""
    e1.category = "Image"
    e1.metadata = Mock()
    e1.metadata.page_number = 2
    e1.metadata.element_id = None
    e1.metadata.coordinates = None

    e2 = Mock()
    e2.text = ""
    e2.category = "Image"
    e2.metadata = Mock()
    e2.metadata.page_number = 2
    e2.metadata.element_id = None
    e2.metadata.coordinates = None

    # Minimal IngestionPipeline stub that passes nodes through transformations
    class _FakePipeline:
        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = type("C", (), {"hits": 0, "misses": 0})()

        def run(self, nodes=None, documents=None, show_progress=False):  # type: ignore[override]
            nodes = nodes or documents or []
            for t in self.transformations:
                nodes = t(nodes)
            return nodes

    fpath = tmp_path / "file.txt"
    fpath.write_text("dummy")

    with (
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
        patch("src.processing.document_processor.partition", return_value=[e1, e2]),
        patch(
            "src.processing.document_processor.chunk_by_basic",
            side_effect=lambda els: els,
        ),
    ):
        proc = DocumentProcessor()
        result = await proc.process_document_async(fpath)

    assert len(result.elements) >= 2
    ids = [el.metadata.get("node_id") for el in result.elements[:2]]
    assert all(isinstance(x, str) and len(x) == 64 for x in ids)
    assert ids[0] != ids[1]
