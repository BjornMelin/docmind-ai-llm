from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.utils import document as doc_utils

pytestmark = pytest.mark.unit


def test_sanitize_doc_metadata_drops_path_keys_and_normalizes_source() -> None:
    meta = {
        "source_path": "/abs/path/file.pdf",
        "file_path": "/abs/path/file.pdf",
        "path": "/abs/path/file.pdf",
        "source": "/abs/path/file.pdf",
        "keep": "x",
    }
    out = doc_utils._sanitize_doc_metadata(meta, source_filename="file.pdf")  # type: ignore[attr-defined]
    assert "source_path" not in out
    assert "file_path" not in out
    assert "path" not in out
    assert out["source"] == "file.pdf"
    assert out["source_filename"] == "file.pdf"
    assert out["keep"] == "x"


@pytest.mark.asyncio
async def test_load_documents_unstructured_falls_back_without_reader(
    monkeypatch, tmp_path: Path
) -> None:
    # Force the UnstructuredReader import to fail without affecting llama_index.core.
    monkeypatch.setitem(sys.modules, "llama_index.readers.file", ModuleType("x"))

    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    docs = await doc_utils.load_documents_unstructured([p, tmp_path / "missing.txt"])
    assert len(docs) == 1

    doc = docs[0]
    assert doc.doc_id.startswith("doc-")
    assert doc.metadata["source_filename"] == "a.txt"
    assert "source_path" not in doc.metadata


@pytest.mark.asyncio
async def test_load_documents_unstructured_sanitizes_reader_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    class _Reader:
        def load_data(self, *, file: Path, unstructured_kwargs: dict) -> list[object]:
            _ = unstructured_kwargs
            return [
                SimpleNamespace(metadata={"source": str(file), "path": str(file)}),
                SimpleNamespace(metadata={"source": f"file:{file}"}),
            ]

    mod = ModuleType("llama_index.readers.file")
    mod.UnstructuredReader = _Reader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.readers.file", mod)

    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    docs = await doc_utils.load_documents_unstructured([p])
    assert len(docs) == 2
    assert docs[0].doc_id.endswith("-0")
    assert docs[1].doc_id.endswith("-1")
    assert docs[0].metadata["source"] == "a.txt"
    assert docs[1].metadata["source"] == "a.txt"
    assert docs[0].metadata["source_filename"] == "a.txt"
    assert "path" not in docs[0].metadata


@pytest.mark.asyncio
async def test_load_documents_from_directory_filters_extensions(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("y", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"nope")

    seen: list[Path] = []

    async def _stub(paths):  # type: ignore[no-untyped-def]
        seen.extend([Path(p) for p in paths])
        return ["ok"]

    monkeypatch.setattr(doc_utils, "load_documents_unstructured", _stub)
    docs = await doc_utils.load_documents_from_directory(tmp_path, recursive=False)
    assert docs == ["ok"]
    assert {p.name for p in seen} == {"a.txt", "b.md"}


def test_get_document_info_includes_sha_and_size(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    info = doc_utils.get_document_info(p)
    assert info["exists"] is True
    assert info["source_filename"] == "a.txt"
    assert info["suffix"] == ".txt"
    assert info["size_bytes"] == 5
    assert isinstance(info["sha256"], str)
    assert len(info["sha256"]) == 64


def test_cache_stats_and_clear(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "a").write_text("x", encoding="utf-8")
    (cache_dir / "b").write_text("yz", encoding="utf-8")

    stats = doc_utils.get_cache_stats(cache_dir=cache_dir)
    assert stats["exists"] is True
    assert stats["files"] == 2
    assert stats["bytes"] == 3

    doc_utils.clear_document_cache(cache_dir=cache_dir)
    stats2 = doc_utils.get_cache_stats(cache_dir=cache_dir)
    assert stats2 == {"exists": False, "files": 0, "bytes": 0}


def test_ensure_spacy_model_raises_clear_error() -> None:
    # This project treats spaCy as optional. In minimal environments the import
    # can fail due to missing CLI deps (e.g., typer). Ensure we raise a clear
    # RuntimeError either way.
    with pytest.raises(RuntimeError, match=r"spaCy|spacy"):
        doc_utils.ensure_spacy_model("en_core_web_sm")
