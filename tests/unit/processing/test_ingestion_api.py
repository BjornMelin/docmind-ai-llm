from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.models.processing import IngestionInput
from src.processing import ingestion_api

pytestmark = pytest.mark.unit


def test_sanitize_doc_metadata_drops_path_keys_and_normalizes_source() -> None:
    """Test that metadata is sanitized."""
    meta = {
        "source_path": "/abs/path/file.pdf",
        "file_path": "/abs/path/file.pdf",
        "path": "/abs/path/file.pdf",
        "source": "/abs/path/file.pdf",
        "keep": "x",
    }
    out = ingestion_api.sanitize_document_metadata(meta, source_filename="file.pdf")
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
    """Test that unstructured reader falls back without reader."""
    # Force the UnstructuredReader import to fail without affecting llama_index.core.
    monkeypatch.setitem(sys.modules, "llama_index.readers.file", ModuleType("x"))

    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    docs = await ingestion_api.load_documents([p, tmp_path / "missing.txt"])
    assert len(docs) == 1

    doc = docs[0]
    assert doc.doc_id.startswith("doc-")
    assert doc.metadata["source_filename"] == "a.txt"
    assert "source_path" not in doc.metadata


@pytest.mark.asyncio
async def test_load_documents_unstructured_sanitizes_reader_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    """Test that unstructured reader sanitizes metadata."""

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
    docs = await ingestion_api.load_documents([p])
    assert len(docs) == 2
    assert docs[0].doc_id.endswith("-0")
    assert docs[1].doc_id.endswith("-1")
    assert docs[0].metadata["source"] == "a.txt"
    assert docs[1].metadata["source"] == "a.txt"
    assert docs[0].metadata["source_filename"] == "a.txt"
    assert "path" not in docs[0].metadata


def test_collect_paths_filters_extensions_and_is_deterministic(tmp_path: Path) -> None:
    """Test that collect_paths filters extensions and is deterministic."""
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("y", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"nope")

    collected = ingestion_api.collect_paths(tmp_path, recursive=False)
    assert [p.name for p in collected] == ["a.txt", "b.md"]


def test_collect_paths_root_not_dir_raises(tmp_path: Path) -> None:
    """Test that collect_paths rejects non-directory roots."""
    root = tmp_path / "file.txt"
    root.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        ingestion_api.collect_paths(root, recursive=False)


def test_collect_paths_root_symlink_rejected(tmp_path: Path) -> None:
    """Test that collect_paths rejects symlink roots."""
    real_root = tmp_path / "real"
    real_root.mkdir()
    link_root = tmp_path / "link"
    try:
        link_root.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlinks unsupported in this environment: {exc}")
    with pytest.raises(ValueError, match="Refusing symlink ingestion root"):
        ingestion_api.collect_paths(link_root, recursive=False)


def test_collect_paths_rejects_symlinks(tmp_path: Path) -> None:
    """Test that collect_paths rejects symlinks."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope", encoding="utf-8")

    root = tmp_path / "root"
    root.mkdir()

    link = root / "link.txt"
    try:
        link.symlink_to(outside / "secret.txt")
    except OSError as exc:
        pytest.skip(f"Symlinks unsupported in this environment: {exc}")

    collected = ingestion_api.collect_paths(root, recursive=False)
    assert collected == []


def test_load_documents_skips_symlink_components(tmp_path: Path) -> None:
    """Test that load_documents skips paths that include symlink components."""
    root = tmp_path / "root"
    root.mkdir()
    target = root / "a.txt"
    target.write_text("hello", encoding="utf-8")

    link_dir = tmp_path / "link_dir"
    try:
        link_dir.symlink_to(root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlinks unsupported in this environment: {exc}")

    docs = asyncio.run(ingestion_api.load_documents([link_dir / "a.txt"]))
    assert docs == []


def test_normalize_extensions_handles_dotless_inputs() -> None:
    """Test that_normalize_extensions accepts dotless and mixed-case extensions."""
    out = ingestion_api._normalize_extensions({"TXT", ".Md", ""})
    assert ".txt" in out
    assert ".md" in out


@pytest.mark.asyncio
async def test_load_documents_from_inputs_sanitizes_input_metadata_for_files(
    monkeypatch, tmp_path: Path
) -> None:
    """Test that load_documents_from_inputs sanitizes input metadata for files."""
    monkeypatch.setitem(sys.modules, "llama_index.readers.file", ModuleType("x"))

    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")

    inputs = [
        IngestionInput(
            document_id="doc-1",
            source_path=p,
            metadata={
                "keep": "ok",
                "path": str(p),
                "source_path": str(p),
                "source": str(p),
                "source_filename": str(p),
            },
        )
    ]
    docs = await ingestion_api.load_documents_from_inputs(inputs)
    assert len(docs) == 1

    meta = docs[0].metadata
    assert meta["document_id"] == "doc-1"
    assert meta["keep"] == "ok"
    assert meta["source"] == "a.txt"
    assert meta["source_filename"] == "a.txt"
    assert "path" not in meta
    assert "source_path" not in meta


@pytest.mark.asyncio
async def test_load_documents_from_inputs_bytes_source_filename_is_authoritative() -> (
    None
):
    """Test that load_documents_from_inputs bytes source filename is authoritative."""
    inputs = [
        IngestionInput(
            document_id="doc-2",
            payload_bytes=b"hello",
            metadata={
                "source_filename": "/tmp/evil.txt",
                "source": "file:/tmp/evil.txt",
                "keep": "ok",
            },
        )
    ]
    docs = await ingestion_api.load_documents_from_inputs(inputs)
    assert len(docs) == 1

    meta = docs[0].metadata
    assert meta["document_id"] == "doc-2"
    assert meta["keep"] == "ok"
    assert meta["source"] == "evil.txt"
    assert meta["source_filename"] == "<bytes>"


def test_clear_ingestion_cache_is_best_effort_on_symlinked_cache_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """Test that clear_ingestion_cache is best effort on symlinked cache dir."""
    from src.config.settings import settings

    real_base = tmp_path / "cache-real"
    real_base.mkdir()
    ingestion_dir = real_base / "ingestion"
    ingestion_dir.mkdir()
    (ingestion_dir / "x").write_text("x", encoding="utf-8")

    link_base = tmp_path / "cache-link"
    try:
        link_base.symlink_to(real_base, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlinks unsupported in this environment: {exc}")

    monkeypatch.setattr(settings, "cache_dir", link_base)

    ingestion_api.clear_ingestion_cache()
    assert not ingestion_dir.exists()


def test_clear_ingestion_cache_refuses_symlink_target(
    monkeypatch, tmp_path: Path
) -> None:
    """clear_ingestion_cache refuses symlinked ingestion targets."""
    from src.config.settings import settings

    base = tmp_path / "cache"
    base.mkdir()
    real_ingestion = tmp_path / "real_ingestion"
    real_ingestion.mkdir()
    (real_ingestion / "x").write_text("x", encoding="utf-8")

    target = base / "ingestion"
    try:
        target.symlink_to(real_ingestion, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlinks unsupported in this environment: {exc}")

    monkeypatch.setattr(settings, "cache_dir", base)

    ingestion_api.clear_ingestion_cache()
    assert real_ingestion.exists()


def test_assert_resolves_within_root_rejects_escape(tmp_path: Path) -> None:
    """_assert_resolves_within_root rejects paths escaping the root."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing path that resolves outside"):
        ingestion_api._assert_resolves_within_root(root.resolve(strict=True), outside)
