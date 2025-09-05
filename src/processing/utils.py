"""Processing utilities for Unstructured-first pipeline.

Provides helpers that keep core components simple and library-first.
"""

from __future__ import annotations

import hashlib
from typing import Any


def _normalize_text(value: str) -> str:
    """Normalize text for stable hashing.

    - Strips leading/trailing whitespace
    - Collapses internal whitespace to a single space

    Args:
        value: Input text

    Returns:
        Normalized text suitable for hashing
    """
    if not value:
        return ""
    # Collapse runs of whitespace and strip
    return " ".join(value.split()).strip()


def sha256_id(*parts: str | bytes) -> str:
    """Compute a deterministic SHA-256 hex digest from provided parts.

    Text inputs are normalized via :func:`_normalize_text` to ensure stability.

    Args:
        *parts: Strings or bytes to include in the digest

    Returns:
        Hex-encoded SHA-256 digest
    """
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        else:
            h.update(_normalize_text(str(p)).encode("utf-8"))
    return h.hexdigest()


def is_unstructured_like(element: Any) -> bool:
    """Return True if an element looks like an unstructured Element.

    Heuristic rules used to decide whether an object is safe to pass into
    Unstructured chunkers (e.g., chunk_by_title/basic) that expect rich
    metadata objects rather than unittest.mock instances.

    Args:
        element: Candidate element to inspect.

    Returns:
        True if the object appears compatible with Unstructured; otherwise False.
    """
    if not (
        hasattr(element, "text")
        and hasattr(element, "category")
        and hasattr(element, "metadata")
    ):
        return False

    meta = getattr(element, "metadata", None)
    # Avoid passing unittest.mock-based metadata into real chunkers
    mod = getattr(getattr(meta, "__class__", object), "__module__", "")
    return not mod.startswith("unittest")


__all__ = ["_normalize_text", "is_unstructured_like", "sha256_id"]
