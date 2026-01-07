"""Tests for canonicalization helpers."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from src.utils.canonicalization import (
    CanonicalizationConfig,
    canonicalize_document,
    compute_hashes,
)


@pytest.fixture
def canonical_config() -> CanonicalizationConfig:
    """Return a canonicalization config used across tests."""
    return CanonicalizationConfig(
        version="v1",
        hmac_secret=b"unit-secret",
        hmac_secret_version="secret-v1",  # noqa: S106 - benign test fixture
        metadata_keys=("content_type", "extra"),
    )


def test_canonicalize_document_normalizes_text_and_metadata(
    canonical_config: CanonicalizationConfig,
) -> None:
    """Canonical payload should normalise text, metadata, and model versions."""
    content = "H\u200bello  \r\n\nworld \u0007".encode()
    metadata = {
        "content_type": "text/plain",
        "extra": {"b": 2, "a": 1},
        "ignored": "value",
    }
    model_versions = {"parser": "1.0.0", "ocr": "0.2"}

    payload = canonicalize_document(
        content=content,
        metadata=metadata,
        config=canonical_config,
        model_versions=model_versions,
    )
    envelope = json.loads(payload.decode("utf-8"))

    assert envelope["v"] == canonical_config.version
    assert envelope["text"] == "Hello world"
    assert envelope["metadata"] == {
        "content_type": "text/plain",
        "extra": {"a": 1, "b": 2},
    }
    assert envelope["models"] == {"ocr": "0.2", "parser": "1.0.0"}


def test_compute_hashes_returns_expected_digests(
    canonical_config: CanonicalizationConfig,
) -> None:
    """Hash bundle should include raw SHA256 and canonical HMAC digests."""
    content = b"DocMind canonicalization test"
    metadata = {"content_type": "text/plain", "extra": {"flag": True}}

    bundle = compute_hashes(content, metadata, canonical_config)

    assert bundle.raw_sha256 == hashlib.sha256(content).hexdigest()

    payload = canonicalize_document(content, metadata, canonical_config)
    expected_hmac = hashlib.sha256
    # Manual HMAC check to guard against accidental regressions
    computed = hmac.new(
        canonical_config.hmac_secret,
        payload,
        expected_hmac,
    ).hexdigest()
    assert bundle.canonical_hmac_sha256 == computed

    assert bundle.as_dict() == {
        "raw_sha256": bundle.raw_sha256,
        "canonical_hmac_sha256": bundle.canonical_hmac_sha256,
        "canonicalization_version": canonical_config.version,
        "hmac_secret_version": canonical_config.hmac_secret_version,
    }
