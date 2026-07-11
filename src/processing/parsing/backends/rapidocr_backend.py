"""RapidOCR adapter with import-safe health diagnostics."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

from src.processing.parsing.model_artifacts import (
    RAPIDOCR_ENGLISH_BUNDLE,
    ModelIntegrityError,
    install_downloaded_model_bundle,
    model_files,
    verify_model_bundle,
)

_ENGINE_LOCK = Lock()


def rapidocr_health() -> dict[str, Any]:
    """Return availability diagnostics for RapidOCR."""
    try:
        import rapidocr  # noqa: F401
    except ImportError as exc:
        return {"available": False, "error_type": type(exc).__name__}
    return {"available": True}


def rapidocr_model_files(model_cache_dir: Path) -> dict[str, Path]:
    """Return the required English ONNX model paths in the app cache."""
    return model_files(model_cache_dir, RAPIDOCR_ENGLISH_BUNDLE)


def missing_rapidocr_onnx_models(model_cache_dir: Path) -> list[str]:
    """Return required RapidOCR ONNX artifacts missing from the app cache."""
    return [
        name
        for name, path in rapidocr_model_files(model_cache_dir).items()
        if not path.exists()
    ]


def prefetch_rapidocr_models(model_cache_dir: Path, *, force: bool = False) -> Path:
    """Download RapidOCR ONNX artifacts into the app-owned cache."""
    try:
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("docling is required to prefetch RapidOCR models") from exc
    cache_dir = Path(model_cache_dir)
    if not force:
        try:
            return verify_model_bundle(cache_dir, RAPIDOCR_ENGLISH_BUNDLE)
        except ModelIntegrityError:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".rapidocr-", dir=cache_dir) as temp_dir:
        download_dir = Path(temp_dir) / "download"
        RapidOcrModel.download_models(
            "onnxruntime",
            local_dir=download_dir,
            force=force,
            progress=True,
            lang="english",
        )
        return install_downloaded_model_bundle(
            download_dir,
            cache_dir,
            RAPIDOCR_ENGLISH_BUNDLE,
        )


def verify_rapidocr_models(model_cache_dir: Path) -> Path:
    """Verify the exact RapidOCR English ONNX bundle."""
    return verify_model_bundle(model_cache_dir, RAPIDOCR_ENGLISH_BUNDLE)


def run_rapidocr(image_path: Path, *, model_cache_dir: Path) -> str:
    """Run RapidOCR on one image and return recognized text."""
    files = rapidocr_model_files(model_cache_dir)
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise RuntimeError(
            f"RapidOCR requires verified local model files: {', '.join(missing)}"
        )
    engine = _rapidocr_engine(str(Path(model_cache_dir).resolve()))
    with _ENGINE_LOCK:
        result = engine(str(image_path))
    return _rapidocr_text(result)


@lru_cache(maxsize=2)
def _rapidocr_engine(model_cache_dir: str) -> Any:
    verify_rapidocr_models(Path(model_cache_dir))
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("rapidocr is required for OCR") from exc

    files = rapidocr_model_files(Path(model_cache_dir))
    return RapidOCR(
        params={
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Det.model_path": str(
                files["onnx/PP-OCRv4/det/en_PP-OCRv3_det_mobile.onnx"]
            ),
            "Cls.model_path": str(
                files["onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx"]
            ),
            "Rec.model_path": str(
                files["onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx"]
            ),
            "Rec.rec_keys_path": str(
                files["paddle/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile/en_dict.txt"]
            ),
            "Rec.font_path": str(files["resources/fonts/FZYTK.TTF"]),
            "Global.font_path": str(files["resources/fonts/FZYTK.TTF"]),
        }
    )


def _rapidocr_text(result: Any) -> str:
    items = getattr(result, "txts", None)
    if items is not None:
        return "\n".join(str(item) for item in items if str(item).strip())
    if isinstance(result, tuple) and result:
        result = result[0]
    if isinstance(result, list):
        lines: list[str] = []
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text = item[1]
                if isinstance(text, (list, tuple)) and text:
                    text = text[0]
                if str(text).strip():
                    lines.append(str(text))
        return "\n".join(lines)
    return str(result or "")


__all__ = [
    "missing_rapidocr_onnx_models",
    "prefetch_rapidocr_models",
    "rapidocr_health",
    "rapidocr_model_files",
    "run_rapidocr",
    "verify_rapidocr_models",
]
