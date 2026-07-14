"""Thin, CPU-bounded adapter around RapidOCR's packaged defaults."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

_ENGINE_LOCK = Lock()


def run_rapidocr(image_path: Path) -> str:
    """Run RapidOCR on one image and return recognized text."""
    engine = _rapidocr_engine()
    with _ENGINE_LOCK:
        result = engine(str(image_path))
    return _rapidocr_text(result)


@lru_cache(maxsize=1)
def _rapidocr_engine() -> Any:
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("rapidocr is required for OCR") from exc

    return RapidOCR(
        params={
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )


def _rapidocr_text(result: Any) -> str:
    items = getattr(result, "txts", None)
    return "\n".join(str(item) for item in (items or ()) if str(item).strip())


__all__ = [
    "run_rapidocr",
]
