"""Parser dependency and offline-readiness diagnostics."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from shlex import join as shell_join
from typing import Any

from src.config.settings import DocMindSettings
from src.processing.parsing.backends.ocrmypdf_backend import ocrmypdf_health
from src.processing.parsing.model_artifacts import (
    DOCLING_LAYOUT_BUNDLE,
    RAPIDOCR_ENGLISH_BUNDLE,
    ModelBundle,
    model_directory_issues,
)


def parser_prefetch_command(model_cache_dir: Path) -> str:
    """Return a copy-safe parser prefetch command for the configured cache."""
    return shell_join(
        [
            "uv",
            "run",
            "python",
            "tools/models/pull.py",
            "--parser-defaults",
            "--rapidocr-cache-dir",
            str(model_cache_dir),
        ]
    )


def parser_health(settings: DocMindSettings) -> dict[str, Any]:
    """Return metadata-only parser health and offline readiness."""
    model_cache_dir = Path(settings.ocr.model_cache_dir)
    docling = _package_health(
        "docling",
        module="docling.datamodel.layout_model_specs",
    )
    pypdfium2 = _package_health("pypdfium2")
    rapidocr = _package_health("rapidocr")
    onnxruntime = _package_health("onnxruntime")
    docling_model_issues, docling_model_error = _model_issues(
        model_cache_dir,
        DOCLING_LAYOUT_BUNDLE,
        check=bool(docling["importable"]),
    )
    rapidocr_model_issues, rapidocr_model_error = _model_issues(
        model_cache_dir,
        RAPIDOCR_ENGLISH_BUNDLE,
        check=bool(rapidocr["importable"]),
    )
    ocr_pdf = ocrmypdf_health()
    core_packages = (docling, pypdfium2, rapidocr, onnxruntime)
    ready = (
        all(package["available"] and package["importable"] for package in core_packages)
        and docling_model_error is None
        and rapidocr_model_error is None
        and not docling_model_issues
        and not rapidocr_model_issues
    )
    return {
        "pdf_ready": ready,
        "prefetch_command": parser_prefetch_command(model_cache_dir),
        "docling": {
            **docling,
            "model_issues": docling_model_issues,
            "model_check_error_type": docling_model_error,
            "offline_ready": docling_model_error is None and not docling_model_issues,
        },
        "pypdfium2": pypdfium2,
        "rapidocr": {
            **rapidocr,
            "model_cache_dir": _redact_path(model_cache_dir),
            "model_issues": rapidocr_model_issues,
            "model_check_error_type": rapidocr_model_error,
            "offline_ready": rapidocr_model_error is None and not rapidocr_model_issues,
        },
        "onnxruntime": onnxruntime,
        "ocrmypdf": ocr_pdf,
    }


def _package_health(package: str, *, module: str | None = None) -> dict[str, Any]:
    found = _package_version(package)
    if found is None:
        return {"available": False, "version": None, "importable": False}
    try:
        import_module(module or package)
    except Exception as exc:
        return {
            "available": True,
            "version": found,
            "importable": False,
            "import_error_type": type(exc).__name__,
        }
    return {"available": True, "version": found, "importable": True}


def _model_issues(
    model_cache_dir: Path,
    bundle: ModelBundle,
    *,
    check: bool,
) -> tuple[dict[str, str], str | None]:
    if not check:
        return {}, "DependencyUnavailable"
    try:
        return model_directory_issues(model_cache_dir / bundle.root, bundle), None
    except Exception as exc:
        return {}, type(exc).__name__


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _redact_path(path: Path) -> str:
    return f".../{path.name}" if path.name else "..."


__all__ = ["parser_health", "parser_prefetch_command"]
