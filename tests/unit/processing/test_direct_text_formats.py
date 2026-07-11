"""Direct-text parser boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import DocMindSettings
from src.processing.parsing import service
from src.processing.parsing.errors import DocumentParseError
from src.processing.parsing.formats import read_direct_text

pytestmark = pytest.mark.unit


def test_read_direct_text_decodes_bounded_utf8(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("Local café notes", encoding="utf-8")

    assert read_direct_text(source, max_bytes=100, probe_bytes=32) == (
        "Local café notes"
    )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"%PDF-1.7\n", "binary_magic_rejected"),
        (b"text\x00hidden", "binary_content_rejected"),
        (b"\xff\xfe", "invalid_utf8_text"),
    ],
)
def test_read_direct_text_rejects_binary_or_invalid_content(
    tmp_path: Path,
    payload: bytes,
    reason: str,
) -> None:
    source = tmp_path / "disguised.txt"
    source.write_bytes(payload)

    with pytest.raises(DocumentParseError) as raised:
        read_direct_text(source, max_bytes=100, probe_bytes=32)

    assert raised.value.stage == "direct_text"
    assert raised.value.reason == reason


def test_read_direct_text_enforces_byte_limit_during_read(tmp_path: Path) -> None:
    source = tmp_path / "large.txt"
    source.write_bytes(b"a" * 11)

    with pytest.raises(DocumentParseError) as raised:
        read_direct_text(source, max_bytes=10, probe_bytes=10)

    assert raised.value.reason == "document_size_limit_exceeded"


def test_parser_wraps_source_hash_io_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "unreadable.md"
    source.write_text("content", encoding="utf-8")
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    def _fail_hash(_path: Path) -> str:
        raise PermissionError("private path detail")

    monkeypatch.setattr(service, "sha256_file", _fail_hash)

    with pytest.raises(DocumentParseError) as raised:
        service.parse_document_sync(source, settings=cfg)

    assert raised.value.stage == "source_hash"
    assert raised.value.reason == "source_hash_failed"
    assert raised.value.cause_type == "PermissionError"
    assert "private path detail" not in str(raised.value)


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_parser_rejects_empty_direct_text_output(
    tmp_path: Path,
    content: str,
) -> None:
    source = tmp_path / "empty.md"
    source.write_text(content, encoding="utf-8")
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    with pytest.raises(DocumentParseError) as raised:
        service.parse_document_sync(source, settings=cfg)

    assert raised.value.stage == "resource_validation"
    assert raised.value.reason == "empty_text_output"
