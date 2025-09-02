"""Performance smoke for document processing.

Informational only; marked with performance and kept small/fast.
"""

import time

import pytest

from src.processing.document_processor import DocumentProcessor


@pytest.mark.performance
@pytest.mark.asyncio
async def test_pages_per_second_smoke(tmp_path):
    file_path = tmp_path / "multi.txt"
    # Create a small multi-page-like input (simulated with separators)
    content = "\n\n".join([f"Title {i}\n\npara {i}" for i in range(10)])
    file_path.write_text(content)

    proc = DocumentProcessor()
    start = time.perf_counter()
    res = await proc.process_document_async(file_path)
    elapsed = time.perf_counter() - start

    # Treat each Title as a page surrogate for this smoke
    pages = max(1, sum(1 for e in res.elements if getattr(e, "category", "") == "Title"))
    pps = pages / max(1e-6, elapsed)
    # Loose bound; this is informational, not enforced in CI (-m not performance)
    assert pps > 0.5

