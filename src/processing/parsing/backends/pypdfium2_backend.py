"""pypdfium2 backend diagnostics."""

from __future__ import annotations

from typing import Any


def pypdfium2_health() -> dict[str, Any]:
    """Return availability diagnostics for pypdfium2."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        return {"available": False, "error_type": type(exc).__name__}
    return {"available": True, "version": str(getattr(pdfium, "__version__", ""))}


__all__ = ["pypdfium2_health"]
