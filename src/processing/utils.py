"""Processing utilities for Unstructured-first pipeline.

Provides helpers that keep core components simple and library-first.
"""

from __future__ import annotations

from typing import Any


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

