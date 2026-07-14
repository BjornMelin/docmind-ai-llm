"""Parser dependency and model-availability diagnostics."""

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
    ModelBundle,
    cached_model_directory_issues,
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
            "--parser-cache-dir",
            str(model_cache_dir),
        ]
    )


def parser_health(settings: DocMindSettings) -> dict[str, Any]:
    """Return metadata-only parser dependency and model availability."""
    model_cache_dir = Path(settings.parsing.model_cache_dir)
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
    ocr_pdf = ocrmypdf_health()
    core_packages = (docling, pypdfium2, rapidocr, onnxruntime)
    ready = (
        all(package["available"] and package["importable"] for package in core_packages)
        and docling_model_error is None
        and not docling_model_issues
    )
    return {
        "pdf_dependencies_ready": ready,
        "prefetch_command": parser_prefetch_command(model_cache_dir),
        "docling": {
            **docling,
            "model_issues": docling_model_issues,
            "model_check_error_type": docling_model_error,
            "models_ready": docling_model_error is None and not docling_model_issues,
        },
        "pypdfium2": pypdfium2,
        "rapidocr": {
            **rapidocr,
            "model_source": "package",
            "dependencies_ready": bool(
                rapidocr["importable"] and onnxruntime["importable"]
            ),
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
        return cached_model_directory_issues(
            model_cache_dir / bundle.root, bundle
        ), None
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
