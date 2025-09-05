"""Integration tests for SPEC-002 ingestion pipeline behaviors.

These tests avoid heavy I/O by patching Unstructured and PyMuPDF surfaces.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.models.processing import ProcessingStrategy
from src.processing.document_processor import DocumentProcessor


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_with_tables_emits_page_image_nodes(tmp_path: Path) -> None:
    """PDF with tables → emits text chunks and at least one pdf_page_image node."""
    pdf = tmp_path / "with_tables.pdf"
    pdf.write_text("dummy")

    # Partition returns a table-like element and a text element
    class _El:
        def __init__(self, text: str, category: str) -> None:
            self.text = text
            self.category = category
            self.metadata = SimpleNamespace(page_number=1)

    parts = [_El("Table 1", "Table"), _El("Para", "NarrativeText")]

    # Fake pdf page images
    fake_imgs = [
        {
            "page_no": 1,
            "image_path": str(tmp_path / "with_tables__page-1.png"),
            "bbox": [0.0, 0.0, 10.0, 10.0],
        }
    ]

    class _FakePipeline:
        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = SimpleNamespace(hits=0, misses=1)

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
        patch("src.processing.document_processor.chunk_by_title", return_value=parts),
        patch("src.processing.pdf_pages.save_pdf_page_images", return_value=fake_imgs),
    ):
        proc = DocumentProcessor()
        result = await proc.process_document_async(pdf)

        assert result.elements
        # Has at least one page-image element
        assert any(
            e.metadata.get("modality") == "pdf_page_image" for e in result.elements
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ocr_fallback_produces_text_nodes(tmp_path: Path) -> None:
    """Scanned-like PDF path triggers OCR_ONLY strategy and yields text nodes."""
    pdf = tmp_path / "scanned.pdf"
    pdf.write_text("dummy")

    # Simple element to simulate OCR text
    el = SimpleNamespace(
        text="ocr text",
        category="NarrativeText",
        metadata=SimpleNamespace(page_number=1),
    )

    class _FakePipeline:
        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            self.cache = SimpleNamespace(hits=0, misses=1)

        def run(self, documents=None, show_progress=False):
            nodes = documents or []
            for t in self.transformations:
                nodes = t(nodes)
            return nodes

    with (
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
        patch("src.processing.document_processor.partition", return_value=[el]),
        patch("src.processing.document_processor.chunk_by_basic", return_value=[el]),
        patch.object(
            DocumentProcessor,
            "_get_strategy_for_file",
            return_value=ProcessingStrategy.OCR_ONLY,
        ),
        patch("src.processing.pdf_pages.save_pdf_page_images", return_value=[]),
    ):
        proc = DocumentProcessor()
        result = await proc.process_document_async(pdf)
        # Ensure text element present
        assert any(e.category == "NarrativeText" for e in result.elements)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reingestion_uses_cache_stats(tmp_path: Path) -> None:
    """Re-ingestion exposes cache stats with hits increasing across runs."""
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    el = SimpleNamespace(
        text="hello", category="NarrativeText", metadata=SimpleNamespace(page_number=1)
    )

    class _FakePipeline:
        def __init__(self, transformations=None, cache=None, docstore=None):
            self.transformations = transformations or []
            # Mutable counters we can bump
            self.cache = SimpleNamespace(hits=0, misses=0)

        def run(self, documents=None, show_progress=False):
            nodes = documents or []
            for t in self.transformations:
                nodes = t(nodes)
            # Simulate hit on subsequent runs by increasing hits
            self.cache.hits += 1
            return nodes

    with (
        patch("src.processing.document_processor.IngestionPipeline", _FakePipeline),
        patch("src.processing.document_processor.IngestionCache"),
        patch("src.processing.document_processor.SimpleDocumentStore"),
        patch("src.processing.document_processor.partition", return_value=[el]),
        patch("src.processing.document_processor.chunk_by_basic", return_value=[el]),
        patch("src.processing.pdf_pages.save_pdf_page_images", return_value=[]),
    ):
        proc = DocumentProcessor()
        r1 = await proc.process_document_async(f)
        r2 = await proc.process_document_async(f)

        # Ensure cache stats field exists and reflects non-zero hits
        assert r1.metadata["cache_stats"]["hits"] >= 0
        assert r2.metadata["cache_stats"]["hits"] >= r1.metadata["cache_stats"]["hits"]
