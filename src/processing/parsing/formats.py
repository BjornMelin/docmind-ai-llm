"""Canonical document-format policy for the parsing boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.processing.parsing.errors import DocumentParseError

DIRECT_TEXT_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".txt", ".md", ".markdown", ".rst"}
)
_BINARY_SIGNATURES: Final[tuple[bytes, ...]] = (
    b"%PDF-",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"PK\x03\x04",
    b"\x1f\x8b",
    b"\x7fELF",
)
_ALLOWED_CONTROL_BYTES: Final[frozenset[int]] = frozenset({8, 9, 10, 12, 13})


def is_direct_text_path(path: Path) -> bool:
    """Return whether a source may be decoded directly as UTF-8 text."""
    return Path(path).suffix.lower() in DIRECT_TEXT_EXTENSIONS


def read_direct_text(
    path: Path,
    *,
    max_bytes: int,
    probe_bytes: int,
) -> str:
    """Read strict UTF-8 text after bounded binary-content validation."""
    source = Path(path)
    try:
        with source.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise DocumentParseError(
                source,
                stage="direct_text",
                reason="document_size_limit_exceeded",
            )
        _reject_binary_probe(source, payload[:probe_bytes])
        return payload.decode("utf-8", errors="strict")
    except DocumentParseError:
        raise
    except (OSError, UnicodeError) as exc:
        raise DocumentParseError(
            source,
            stage="direct_text",
            reason="invalid_utf8_text",
            cause=exc,
        ) from exc


def _reject_binary_probe(path: Path, probe: bytes) -> None:
    if any(probe.startswith(signature) for signature in _BINARY_SIGNATURES):
        raise DocumentParseError(
            path,
            stage="direct_text",
            reason="binary_magic_rejected",
        )
    if b"\x00" in probe:
        raise DocumentParseError(
            path,
            stage="direct_text",
            reason="binary_content_rejected",
        )
    controls = sum(byte < 32 and byte not in _ALLOWED_CONTROL_BYTES for byte in probe)
    if probe and controls / len(probe) > 0.05:
        raise DocumentParseError(
            path,
            stage="direct_text",
            reason="binary_content_rejected",
        )


__all__ = ["DIRECT_TEXT_EXTENSIONS", "is_direct_text_path", "read_direct_text"]
