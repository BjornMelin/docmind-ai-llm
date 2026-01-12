"""Document helper utilities (final-release).

This module provides lightweight, import-safe convenience wrappers for loading
documents using LlamaIndex (Unstructured when available) while enforcing
DocMind's path hygiene rules:

- Never persist absolute filesystem paths in document metadata.
- Normalize `metadata["source"]` to a safe basename when present.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.hashing import sha256_file


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _sanitize_doc_metadata(
    meta: dict[str, Any], *, source_filename: str
) -> dict[str, Any]:
    out = dict(meta or {})
    # Drop common path keys.
    for key in ("source_path", "file_path", "path"):
        out.pop(key, None)
    # Normalize `source` to basename (Unstructured/LlamaIndex frequently use
    # paths here).
    src = out.get("source")
    if isinstance(src, str) and ("/" in src or "\\" in src or src.startswith("file:")):
        out["source"] = Path(src).name
    out.setdefault("source_filename", source_filename)
    return out


async def load_documents_unstructured(
    file_paths: Sequence[str | Path], settings_obj: Any | None = None
) -> list[Any]:
    """Load documents using UnstructuredReader when installed (async-friendly).

    Returns a list of LlamaIndex `Document` objects. This function is safe to
    call even when Unstructured is unavailable: it falls back to a plain-text
    read for UTF-8-ish inputs.
    """
    from llama_index.core import Document

    paths = [Path(p) for p in file_paths]
    docs: list[Any] = []
    reader = None
    try:
        from llama_index.readers.file import UnstructuredReader

        reader = UnstructuredReader()
    except Exception:
        reader = None

    for path in paths:
        if not path.exists():
            continue
        doc_id = f"doc-{_sha256_file(path)[:16]}"
        if reader is not None:
            try:
                loaded = reader.load_data(  # type: ignore[call-arg]
                    file=path,
                    unstructured_kwargs={"filename": str(path)},
                )
                for item in loaded or []:
                    meta = getattr(item, "metadata", {}) or {}
                    item.doc_id = doc_id
                    item.metadata = _sanitize_doc_metadata(
                        meta,
                        source_filename=path.name,
                    )
                    docs.append(item)
                continue
            except Exception as exc:
                logger.debug("UnstructuredReader failed for %s: %s", path.name, exc)

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        docs.append(
            Document(
                text=text,
                doc_id=doc_id,
                metadata=_sanitize_doc_metadata({}, source_filename=path.name),
            )
        )
    return docs


async def load_documents_from_directory(
    directory_path: str | Path,
    settings_obj: Any | None = None,
    recursive: bool = True,
    supported_extensions: set[str] | None = None,
) -> list[Any]:
    """Load documents from a directory using UnstructuredReader when available."""
    root = Path(directory_path)
    if not root.exists():
        return []
    exts = {e.lower() for e in (supported_extensions or {".pdf", ".txt", ".md"})}
    glob = "**/*" if recursive else "*"
    paths = [p for p in root.glob(glob) if p.is_file() and p.suffix.lower() in exts]
    return await load_documents_unstructured(paths, settings_obj=settings_obj)


async def load_documents(file_paths: list[str | Path]) -> list[Any]:
    """Convenience alias retained for compatibility with async callers."""
    return await load_documents_unstructured(file_paths)


def ensure_spacy_model(model: str = "en_core_web_sm") -> Any:
    """Load a spaCy model when installed; raises a clear error when missing."""
    try:
        import spacy  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("spaCy is not installed (optional dependency).") from exc
    try:
        return spacy.load(model)
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(
            f"spaCy model '{model}' is not installed. "
            "Install it with: uv run python -m spacy download en_core_web_sm"
        ) from exc


def get_document_info(file_path: str | Path) -> dict[str, Any]:
    """Return basic file metadata (path-safe; no absolute paths persisted)."""
    path = Path(file_path)
    if not path.exists():
        return {"exists": False, "source_filename": path.name}
    stat = path.stat()
    return {
        "exists": True,
        "source_filename": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(path),
    }


def _resolve_ingestion_cache_dir(cache_dir: Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    try:
        from src.config.settings import settings as _settings

        return _settings.cache_dir / "ingestion"
    except Exception:  # pragma: no cover
        return Path("./cache/ingestion")


def clear_document_cache(*, cache_dir: Path | None = None) -> None:
    """Best-effort cache cleanup for local ingestion caches."""
    cache_path = _resolve_ingestion_cache_dir(cache_dir)
    with logger.catch(reraise=False):
        shutil.rmtree(cache_path, ignore_errors=True)


def get_cache_stats(*, cache_dir: Path | None = None) -> dict[str, Any]:
    """Return basic cache directory stats (best-effort)."""
    cache_path = _resolve_ingestion_cache_dir(cache_dir)
    cache_dir = cache_path
    if not cache_dir.exists():
        return {"exists": False, "files": 0, "bytes": 0}
    files = [p for p in cache_dir.glob("**/*") if p.is_file()]
    return {
        "exists": True,
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


__all__ = [
    "clear_document_cache",
    "ensure_spacy_model",
    "get_cache_stats",
    "get_document_info",
    "load_documents",
    "load_documents_from_directory",
    "load_documents_unstructured",
]
