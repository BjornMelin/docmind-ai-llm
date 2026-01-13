"""Shared exception groupings for reuse across modules."""

from __future__ import annotations

IMPORT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ImportError,
    AttributeError,
    TypeError,
    ValueError,
)
