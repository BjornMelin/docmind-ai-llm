"""Unit tests for ingest adapter minimal return shape.

Verifies that calling `ingest_files` with an empty file list returns the
expected dict with count=0 and pg_index=None without performing heavy work.
"""

from __future__ import annotations

from src.ui.ingest_adapter import ingest_files


def test_ingest_files_empty_returns_zero_and_no_pg() -> None:
    """When no files are provided, returns count=0 and no pg_index."""
    out = ingest_files([], enable_graphrag=True)
    assert isinstance(out, dict)
    assert out.get("count") == 0
    assert out.get("pg_index") is None
