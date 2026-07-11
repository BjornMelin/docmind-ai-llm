from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from src.models.processing import CANONICAL_DOCUMENT_ID_KEY, IngestionInput
from src.processing import ingestion_api
from src.processing.parsing.canonical_types import (
    DocumentParseResult,
    PageParseResult,
    ParserFramework,
    ParserProfile,
)
from src.processing.parsing.errors import DocumentParseError

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
async def test_load_documents_text_parser_failure_is_propagated(
    monkeypatch, tmp_path: Path
) -> None:
    """Direct-text parser failures remain parser-boundary failures."""

    async def _explode(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("parser unavailable")

    monkeypatch.setattr(ingestion_api, "_parse_path", _explode)

    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")

    with pytest.raises(DocumentParseError) as raised:
        await ingestion_api.load_documents([p])

    assert raised.value.stage == "ingestion_facade"
    assert raised.value.reason == "parser_service_failed"
    assert raised.value.source_suffix == ".txt"


def test_generate_stable_id_uses_full_sha256(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_bytes(b"canonical payload")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert ingestion_api.generate_stable_id(path) == f"doc-{digest}"


def test_generate_stable_id_wraps_hash_io_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "unreadable.txt"
    path.write_text("content", encoding="utf-8")
    monkeypatch.setattr(
        ingestion_api,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(DocumentParseError) as raised:
        ingestion_api.generate_stable_id(path)

    assert raised.value.stage == "source_hash"
    assert raised.value.reason == "source_hash_failed"
    assert raised.value.cause_type == "PermissionError"
    assert "denied" not in str(raised.value)


@pytest.mark.asyncio
async def test_load_documents_local_model_errors_do_not_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Local-model parser failures are surfaced instead of hidden."""

    async def _fail_fast(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("RapidOCR requires verified local ONNX models")

    monkeypatch.setattr(ingestion_api, "_parse_path", _fail_fast)

    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF-1.4\n%%EOF\n")

    with pytest.raises(DocumentParseError) as raised:
        await ingestion_api.load_documents([p])

    assert raised.value.stage == "ingestion_facade"
    assert raised.value.reason == "parser_service_failed"
    assert raised.value.source_suffix == ".pdf"


@pytest.mark.asyncio
async def test_pdf_parser_failure_publishes_no_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed PDF parse must not become a decoded LlamaIndex Document."""
    import llama_index.core as llama_core

    published: list[object] = []

    class _TrackingDocument:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            published.append(self)

    async def _explode(*_args: object, **_kwargs: object) -> None:
        raise DocumentParseError(
            pdf,
            stage="docling_conversion",
            reason="conversion_status_partial_success",
        )

    monkeypatch.setattr(llama_core, "Document", _TrackingDocument)
    monkeypatch.setattr(ingestion_api, "_parse_path", _explode)
    pdf = tmp_path / "malformed.pdf"
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n")

    with pytest.raises(DocumentParseError) as raised:
        await ingestion_api.load_documents([pdf])

    assert raised.value.stage == "docling_conversion"
    assert raised.value.reason == "conversion_status_partial_success"
    assert raised.value.source_suffix == ".pdf"
    assert published == []


@pytest.mark.asyncio
async def test_load_documents_preserves_direct_text_parse_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The facade preserves canonical direct-text parser failures."""
    source = tmp_path / "invalid.txt"
    source.write_bytes(b"\xff")
    parse_error = DocumentParseError(
        source,
        stage="direct_text",
        reason="invalid_utf8_text",
    )

    async def _explode(*_args: object, **_kwargs: object) -> None:
        raise parse_error

    monkeypatch.setattr(ingestion_api, "_parse_path", _explode)

    with pytest.raises(DocumentParseError) as raised:
        await ingestion_api.load_documents([source])

    assert raised.value is parse_error


@pytest.mark.asyncio
async def test_load_documents_sanitizes_parser_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    """Parser page metadata is sanitized and page IDs are preserved."""
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")

    async def _parse(
        path: Path,
        *,
        document_id: str,
        parsing_overrides: dict | None = None,
    ) -> DocumentParseResult:
        _ = parsing_overrides
        return DocumentParseResult(
            document_id=document_id,
            source_filename=path.name,
            source_hash="abc",
            profile=ParserProfile.CPU_SAFE,
            parser_framework=ParserFramework.DIRECT_TEXT,
            page_count=2,
            pages=[
                PageParseResult(
                    page_id=f"{document_id}::page::1",
                    page_index=0,
                    text_markdown="one",
                    routing_reason="test",
                ),
                PageParseResult(
                    page_id=f"{document_id}::page::2",
                    page_index=1,
                    text_markdown="two",
                    routing_reason="test",
                ),
            ],
        )

    monkeypatch.setattr(ingestion_api, "_parse_path", _parse)

    docs = await ingestion_api.load_documents([p])
    assert len(docs) == 2
    assert docs[0].doc_id.endswith("-0")
    assert docs[1].doc_id.endswith("-1")
    assert docs[0].metadata["source_filename"] == "a.txt"
    assert docs[0].metadata[CANONICAL_DOCUMENT_ID_KEY].startswith("doc-")
    assert docs[0].metadata["page_id"].endswith("::page::1")
    assert docs[0].metadata["parsing"]["framework"] == "direct_text"
    assert "path" not in docs[0].metadata

    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.vector_stores.utils import node_to_metadata_dict

    node = TokenTextSplitter(chunk_size=256, chunk_overlap=0).get_nodes_from_documents(
        docs
    )[0]
    payload = node_to_metadata_dict(node)
    assert (
        payload[CANONICAL_DOCUMENT_ID_KEY]
        == docs[0].metadata[CANONICAL_DOCUMENT_ID_KEY]
    )
    assert payload["document_id"] == docs[0].doc_id
    assert CANONICAL_DOCUMENT_ID_KEY in node.excluded_embed_metadata_keys


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
    """Test that _normalize_extensions accepts dotless and mixed-case extensions."""
    out = ingestion_api._normalize_extensions({"TXT", ".Md", ""})
    assert ".txt" in out
    assert ".md" in out


@pytest.mark.asyncio
async def test_load_documents_from_inputs_sanitizes_input_metadata_for_files(
    monkeypatch, tmp_path: Path
) -> None:
    """Test that load_documents_from_inputs sanitizes input metadata for files."""

    async def _parse(
        path: Path,
        *,
        document_id: str,
        parsing_overrides: dict | None = None,
    ) -> DocumentParseResult:
        _ = parsing_overrides
        return DocumentParseResult(
            document_id=document_id,
            source_filename=path.name,
            source_hash="a" * 64,
            profile=ParserProfile.CPU_SAFE,
            parser_framework=ParserFramework.DIRECT_TEXT,
            page_count=1,
            pages=[
                PageParseResult(
                    page_id=f"{document_id}::page::1",
                    page_index=0,
                    text_markdown="hello",
                    routing_reason="test",
                )
            ],
        )

    monkeypatch.setattr(ingestion_api, "_parse_path", _parse)

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
            },
        )
    ]
    docs = await ingestion_api.load_documents_from_inputs(inputs)
    assert len(docs) == 1

    meta = docs[0].metadata
    assert meta["document_id"] == "doc-1"
    assert meta[CANONICAL_DOCUMENT_ID_KEY] == "doc-1"
    assert meta["keep"] == "ok"
    assert meta["source"] == "a.txt"
    assert meta["source_filename"] == "a.txt"
    assert "path" not in meta
    assert "source_path" not in meta


@pytest.mark.asyncio
async def test_load_documents_from_inputs_text_source_filename_is_authoritative() -> (
    None
):
    """Test that canonical text source metadata overrides user path metadata."""
    inputs = [
        IngestionInput(
            document_id="doc-2",
            payload_text="hello",
            metadata={
                "source": "file:/tmp/evil.txt",
                "keep": "ok",
            },
        )
    ]
    docs = await ingestion_api.load_documents_from_inputs(inputs)
    assert len(docs) == 1

    meta = docs[0].metadata
    assert meta["document_id"] == "doc-2"
    assert meta[CANONICAL_DOCUMENT_ID_KEY] == "doc-2"
    assert meta["keep"] == "ok"
    assert meta["source"] == "evil.txt"
    assert meta["source_filename"] == "<text>"


def test_ingestion_input_rejects_parser_owned_metadata() -> None:
    """Test that callers cannot forge canonical parser provenance."""
    with pytest.raises(ValueError, match="parser-owned: source_filename"):
        IngestionInput(
            document_id="doc-forgery",
            payload_text="hello",
            metadata={"source_filename": "/tmp/evil.txt"},
        )

    with pytest.raises(ValueError, match=CANONICAL_DOCUMENT_ID_KEY):
        IngestionInput(
            document_id="doc-forgery",
            payload_text="hello",
            metadata={CANONICAL_DOCUMENT_ID_KEY: "forged"},
        )


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

    monkeypatch.setattr(settings.cache, "dir", link_base)

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

    monkeypatch.setattr(settings.cache, "dir", base)

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
