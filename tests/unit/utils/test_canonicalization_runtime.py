"""Unit tests for canonicalization helper functions."""

from __future__ import annotations

from pathlib import Path

from src.utils.canonicalization import (
    CanonicalizationConfig,
    HashBundle,
    _filter_metadata,
    _normalise_text,
    canonicalize_document,
    compute_hashes,
)


def test_normalise_text_collapses_whitespace() -> None:
    raw = "\ufeffHello\u200b  World\r\n\nSecond\tLine  "
    assert _normalise_text(raw) == "Hello WorldSecondLine"


def test_filter_metadata_keeps_sorted_keys(tmp_path: Path) -> None:
    metadata = {
        "source": "web",
        "source_path": tmp_path / "doc.pdf",
        "language": None,
        "extra": "ignored",
        "tags": {"b", "a"},
    }
    filtered = _filter_metadata(metadata, ("source", "tags", "source_path"))
    assert filtered == {
        "source": "web",
        "tags": ["a", "b"],
        "source_path": metadata["source_path"],
    }


def _config() -> CanonicalizationConfig:
    return CanonicalizationConfig(
        version="v1",
        hmac_secret=b"secret",
        hmac_secret_version="s1",  # noqa: S106
    )


def test_canonicalize_document_stable(tmp_path: Path) -> None:
    content = b"Hello\nWorld"
    metadata_one = {"source": "web", "source_path": str(tmp_path / "a.txt")}
    metadata_two = {"source_path": str(tmp_path / "a.txt"), "source": "web"}

    payload_one = canonicalize_document(content, metadata_one, _config())
    payload_two = canonicalize_document(content, metadata_two, _config())
    assert payload_one == payload_two


def test_compute_hashes_returns_bundle(tmp_path: Path) -> None:
    content = b"Hello"
    metadata = {"source": "web", "source_path": str(tmp_path / "doc.txt")}
    bundle = compute_hashes(content, metadata, _config(), {"parser": "1.0"})
    assert isinstance(bundle, HashBundle)
    expected_raw = "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969"
    assert bundle.raw_sha256 == expected_raw
    assert "canonical_hmac_sha256" in bundle.as_dict()
