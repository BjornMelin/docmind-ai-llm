"""Hashing helpers.

Centralizes common digest functions used across persistence and utils modules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def document_id_from_sha256(digest: str) -> str:
    """Return the canonical document ID for a full SHA-256 hex digest."""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("digest must be a full lowercase SHA-256 hex value")
    return f"doc-{digest}"


def sha256_file(path: str | Path) -> str:
    """Return hex sha256 digest for a file."""
    digest = hashlib.sha256()
    resolved = Path(path)
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
