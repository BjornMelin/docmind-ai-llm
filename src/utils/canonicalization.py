"""Canonicalization and hashing utilities for deterministic document IDs.

This module defines helper functions that normalize document content and
selected metadata into a canonical byte representation. The canonical payload is
then hashed using both raw SHA-256 (content only) and HMAC-SHA-256 (canonical
payload + secret) so callers can shadow new identifiers while keeping legacy
hashes for migration workflows.

The design mirrors the guidance captured in the ingestion refactor plan:

* Normalize Unicode text to NFKC and collapse redundant whitespace.
* Strip zero-width characters and control codes that often vary between OCR
  runs.
* Include a curated set of metadata fields (content type, language, source,
  source_path) sorted deterministically.
* Version all canonicalization and HMAC secrets to support rotations without
  breaking determinism guarantees.

A small data class :class:`HashBundle` carries both digests along with version
metadata, simplifying transport across pipeline stages and Pydantic models.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import unicodedata
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CanonicalizationConfig",
    "HashBundle",
    "canonicalize_document",
    "compute_hashes",
]


# Regex patterns used during normalization. The patterns intentionally avoid
# expensive Unicode categories to keep runtime predictable.
_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
_CONTROL_CHAR_RE = re.compile(r"[\u0000-\u001F\u007F]")
_MULTI_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_METADATA_KEYS: tuple[str, ...] = (
    "content_type",
    "language",
    "source",
    "source_path",
)


@dataclass(frozen=True)
class CanonicalizationConfig:
    """Configuration for document canonicalization.

    Attributes:
        version: Canonicalization spec version string.
        hmac_secret: Raw bytes used for HMAC-SHA256 secret derivation.
        hmac_secret_version: Version identifier for the active secret.
        metadata_keys: Ordered metadata keys that participate in the canonical
            payload. Any missing keys are silently skipped to keep the envelope
            deterministic.
    """

    version: str
    hmac_secret: bytes
    hmac_secret_version: str
    metadata_keys: Sequence[str] = DEFAULT_METADATA_KEYS


@dataclass(frozen=True)
class HashBundle:
    """Container for raw and canonical digests."""

    raw_sha256: str
    canonical_hmac_sha256: str
    canonicalization_version: str
    hmac_secret_version: str

    def as_dict(self) -> dict[str, str]:
        """Return a serialisable representation for downstream models.

        Returns:
            dict[str, str]: Mapping of the bundle fields suitable for dumps.
        """
        return {
            "raw_sha256": self.raw_sha256,
            "canonical_hmac_sha256": self.canonical_hmac_sha256,
            "canonicalization_version": self.canonicalization_version,
            "hmac_secret_version": self.hmac_secret_version,
        }


def _normalise_text(text: str) -> str:
    """Normalise textual content for deterministic hashing.

    Args:
        text: Raw document text decoded from bytes.

    Returns:
        str: Normalised text with Unicode, whitespace, and control characters
        cleaned for stable hashing.
    """
    normalised = unicodedata.normalize("NFKC", text)
    normalised = normalised.replace("\r\n", "\n").replace("\r", "\n")
    normalised = _ZERO_WIDTH_RE.sub("", normalised)
    normalised = _CONTROL_CHAR_RE.sub("", normalised)
    # Collapse whitespace but keep single newlines by first splitting on lines.
    lines = [
        _MULTI_WHITESPACE_RE.sub(" ", line).strip() for line in normalised.split("\n")
    ]
    # Remove empty lines at head/tail but preserve intentional blank lines in
    # between paragraphs for stability.
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _filter_metadata(
    metadata: Mapping[str, Any],
    allowed_keys: Sequence[str],
) -> MutableMapping[str, Any]:
    """Select a stable subset of metadata fields.

    Args:
        metadata: Original metadata mapping.
        allowed_keys: Ordered keys retained in the canonical payload.

    Returns:
        dict[str, Any]: Filtered metadata with nested collections sorted for
        deterministic serialisation.
    """
    result: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if value is None:
            continue
        if isinstance(value, dict):
            result[key] = _sort_nested_dict(value)
        elif isinstance(value, (list, tuple, set)):
            result[key] = sorted(value)
        else:
            result[key] = value
    return result


def _sort_nested_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively sort dictionary keys for deterministic JSON dumps.

    Args:
        data: Arbitrary nested mapping to sort.

    Returns:
        dict[str, Any]: Mapping with keys sorted depth-first; nested lists/sets
        are converted to sorted lists for stability.
    """
    return {
        key: (
            _sort_nested_dict(value)
            if isinstance(value, Mapping)
            else sorted(value)
            if isinstance(value, (list, tuple, set))
            else value
        )
        for key, value in sorted(data.items(), key=lambda item: item[0])
    }


def canonicalize_document(
    content: bytes,
    metadata: Mapping[str, Any],
    config: CanonicalizationConfig,
    model_versions: Mapping[str, str] | None = None,
) -> bytes:
    """Return canonical payload bytes for ``content`` and ``metadata``.

    Args:
        content: Raw document bytes as read from disk.
        metadata: Arbitrary metadata dictionary; only keys present in
            ``config.metadata_keys`` are retained.
        config: Canonicalization configuration.
        model_versions: Optional mapping describing parser/OCR model versions.

    Returns:
        UTF-8 encoded canonical payload bytes.
    """
    text = content.decode("utf-8", errors="replace")

    normalised_text = _normalise_text(text)
    filtered_metadata = _filter_metadata(metadata, config.metadata_keys)
    filtered_models = _sort_nested_dict(model_versions) if model_versions else {}

    envelope = {
        "v": config.version,
        "text": normalised_text,
        "metadata": filtered_metadata,
        "models": filtered_models,
    }

    return json.dumps(
        envelope, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def compute_hashes(
    content: bytes,
    metadata: Mapping[str, Any],
    config: CanonicalizationConfig,
    model_versions: Mapping[str, str] | None = None,
) -> HashBundle:
    """Compute raw and canonical digests for document ``content``.

    Args:
        content: Raw document bytes.
        metadata: Metadata mapping used for canonical payload construction.
        config: Canonicalization configuration.
        model_versions: Optional mapping of component versions that influenced
            the canonical payload (e.g., parser or OCR model identifiers).

    Returns:
        :class:`HashBundle` with both raw and canonical digests.
    """
    canonical_payload = canonicalize_document(
        content=content,
        metadata=metadata,
        config=config,
        model_versions=model_versions,
    )
    raw_digest = hashlib.sha256(content).hexdigest()
    canonical_digest = hmac.new(
        config.hmac_secret,
        canonical_payload,
        hashlib.sha256,
    ).hexdigest()
    return HashBundle(
        raw_sha256=raw_digest,
        canonical_hmac_sha256=canonical_digest,
        canonicalization_version=config.version,
        hmac_secret_version=config.hmac_secret_version,
    )
