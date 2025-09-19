"""Tests for canonicalization and hashing utilities."""

from __future__ import annotations

import hashlib
import os

import pytest

from src.config.settings import HashingConfig
from src.utils.canonicalization import (
    CanonicalizationConfig,
    canonicalize_document,
    compute_hashes,
)


@pytest.fixture
def canonical_config() -> CanonicalizationConfig:
    """Create a canonicalization configuration suitable for tests."""
    hashing_defaults = HashingConfig()
    secret_source = os.environ.get(
        "DOCMIND_HASH_SECRET", hashing_defaults.hmac_secret
    ).strip()
    secret = secret_source.encode("utf-8")
    if len(secret) < 32:
        # Derive a deterministic high-entropy fallback for test environments.
        secret = hashlib.sha256(secret).digest()
    return CanonicalizationConfig(
        version=hashing_defaults.canonicalization_version,
        hmac_secret=secret,
        hmac_secret_version=hashing_defaults.hmac_secret_version,
        metadata_keys=("content_type", "language", "source"),
    )


def test_canonicalize_document_stable(canonical_config: CanonicalizationConfig) -> None:
    """Ensure canonicalization emits identical payloads for identical input."""
    content = "Résumé\r\n\r\nHello\u200b world".encode()
    metadata = {
        "content_type": "text/plain",
        "language": "fr",
        "ignored_field": "should_not_participate",
    }

    payload_first = canonicalize_document(content, metadata, canonical_config)
    payload_second = canonicalize_document(content, metadata, canonical_config)

    assert payload_first == payload_second
    assert "Résumé" in payload_first.decode("utf-8")
    assert "ignored_field" not in payload_first.decode("utf-8")


def test_compute_hashes_consistent(canonical_config: CanonicalizationConfig) -> None:
    """Verify compute_hashes returns stable results on repeated invocations."""
    content = b"abc123"
    metadata = {"content_type": "text/plain"}
    bundle = compute_hashes(content, metadata, canonical_config)

    assert bundle.raw_sha256 != bundle.canonical_hmac_sha256
    repeat = compute_hashes(content, metadata, canonical_config)
    assert repeat == bundle


def test_compute_hashes_changes_with_content(
    canonical_config: CanonicalizationConfig,
) -> None:
    """Confirm compute_hashes output changes when the content bytes differ."""
    metadata = {"content_type": "text/plain"}
    bundle_a = compute_hashes(b"abc", metadata, canonical_config)
    bundle_b = compute_hashes(b"abcd", metadata, canonical_config)
    assert bundle_a != bundle_b
