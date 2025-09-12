"""Canonical sorting and rounding helpers for stable outputs.

The canonical order is defined as primary key: ``-score`` (descending),
secondary tie-break: ``doc_id`` (ascending, as string). Rounding uses
fixed 6-decimal formatting.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _extract(item: Any) -> tuple[str, float]:
    """Extract ``(doc_id, score)`` from common container shapes.

    Supports:
        - ``(doc_id, score)`` tuples
        - dicts with keys ``doc_id`` and ``score``
        - objects with attributes ``doc_id`` and ``score``
    """
    # Tuple or list pair
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return str(item[0]), float(item[1])
    # Mapping
    if isinstance(item, dict) and "doc_id" in item and "score" in item:
        return str(item["doc_id"]), float(item["score"])
    # Object attributes
    did = getattr(item, "doc_id", None)
    sc = getattr(item, "score", None)
    if did is not None and sc is not None:
        return str(did), float(sc)
    raise TypeError(
        "Unsupported item shape for canonical_sort: "
        f"type={type(item).__name__}, repr={item!r}"
    )


def canonical_sort(items: Iterable[Any]) -> list[tuple[str, float]]:
    """Return items sorted by canonical order: ``-score``, then ``doc_id``.

    Args:
        items: Iterable of tuples/dicts/objects containing doc_id and score.

    Returns:
        A list of ``(doc_id, score)`` tuples in canonical order.
    """
    pairs = [_extract(x) for x in items]
    return sorted(pairs, key=lambda p: (-p[1], p[0]))


def round6(x: float) -> str:
    """Format a float with 6 decimal places (string).

    Args:
        x: Float value.

    Returns:
        String formatted to 6 decimals with standard rounding.
    """
    return f"{x:.6f}"


__all__ = ["canonical_sort", "round6"]
