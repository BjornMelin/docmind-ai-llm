"""Lightweight tests for DocumentProcessor simple methods.

Avoids running heavy pipelines; focuses on strategy selection, size limit,
and hash calculation.
"""

from __future__ import annotations

import importlib


def test_strategy_selection_by_extension(tmp_path):  # type: ignore[no-untyped-def]
    dmod = importlib.import_module("src.processing.document_processor")
    proc = dmod.DocumentProcessor()

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert proc.get_strategy_for_file(pdf).name in {"HI_RES", "FAST", "OCR_ONLY"}

    txt = tmp_path / "a.txt"
    txt.write_text("hello", encoding="utf-8")
    assert proc.get_strategy_for_file(txt).name == "FAST"


def test_max_size_resolution_with_overrides(monkeypatch, tmp_path):  # type: ignore[no-untyped-def]
    dmod = importlib.import_module("src.processing.document_processor")

    class _S:
        class processing:  # noqa: N801 - match attribute name
            max_document_size_mb = 7

    proc = dmod.DocumentProcessor(settings=_S)
    assert proc._get_max_document_size_mb() == 7  # pylint: disable=protected-access

    class _S2:
        max_document_size_mb = 9

    proc2 = dmod.DocumentProcessor(settings=_S2)
    assert proc2._get_max_document_size_mb() == 9  # pylint: disable=protected-access


def test_calculate_document_hash(tmp_path):  # type: ignore[no-untyped-def]
    dmod = importlib.import_module("src.processing.document_processor")
    proc = dmod.DocumentProcessor()
    f = tmp_path / "file.txt"
    f.write_text("abc", encoding="utf-8")
    h = proc._calculate_document_hash(f)  # pylint: disable=protected-access
    assert isinstance(h, str)
    assert len(h) == 64
