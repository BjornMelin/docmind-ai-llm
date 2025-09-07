"""Unit tests for deterministic IDs and lineage in processing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.processing.document_processor import DocumentProcessor
from src.processing.utils import sha256_id


@pytest.mark.unit
def test_sha256_id_determinism() -> None:
    """sha256_id returns stable values for equivalent inputs."""
    a = sha256_id("/path/to/file.pdf", "1", " Hello\nWorld  ")
    b = sha256_id("/path/to/file.pdf", "1", "Hello World")
    assert a == b
    assert len(a) == 64


@pytest.mark.unit
@pytest.mark.asyncio
async def test_element_node_id_and_parent_id_present(tmp_path: Path) -> None:
    """Element nodes carry deterministic node_id and lineage to parent hash."""
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    # Unstructured partition returns one element with page_number
    elem = Mock()
    elem.text = " test  text "
    elem.category = "NarrativeText"
    elem.metadata = Mock()
    elem.metadata.page_number = 1

    class _FakePipeline:
        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = type("C", (), {"hits": 0, "misses": 0})()

        def run(self, nodes=None, documents=None, show_progress=False):
            nodes = nodes or documents or []
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
        proc = DocumentProcessor()
        result = await proc.process_document_async(f)
        assert result.elements
        md = result.elements[0].metadata
        assert "node_id" in md
        assert isinstance(md["node_id"], str)
        assert len(md["node_id"]) == 64
        assert "parent_id" in md
        assert isinstance(md["parent_id"], str)
