"""Canonical ingestion API (loader + path hygiene).

This module centralizes local filesystem ingestion primitives:

- Path enumeration with deterministic ordering and strict symlink rejection.
- File loading into LlamaIndex ``Document`` objects via UnstructuredReader when
  available.
- Metadata sanitization to avoid persisting absolute paths.
- Targeted ingestion cache cleanup under ``settings.cache_dir / "ingestion"``.

Per ADR-045 + SPEC-026 this is the single source of truth for loader behavior.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.utils.hashing import sha256_file

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core import Document

    from src.models.processing import IngestionInput

_DEFAULT_EXTENSIONS: set[str] = {".pdf", ".txt", ".md"}


def _normalize_extensions(extensions: set[str] | None) -> set[str]:
    if not extensions:
        return set(_DEFAULT_EXTENSIONS)
    normalized: set[str] = set()
    for ext in extensions:
        value = f".{ext}" if ext and not ext.startswith(".") else ext
        if value:
            normalized.add(value.lower())
    return normalized or set(_DEFAULT_EXTENSIONS)


def _iter_relative_chain(root: Path, rel_parts: tuple[str, ...]) -> Iterable[Path]:
    current = root
    yield current
    for part in rel_parts:
        current = current / part
        yield current


def _assert_no_symlinks_under_root(root: Path, candidate: Path) -> None:
    """Reject any symlink in the root->candidate path chain."""
    rel = candidate.relative_to(root)
    for entry in _iter_relative_chain(root, rel.parts):
        if entry.is_symlink():
            raise ValueError(f"Refusing symlink in path chain: {entry}")


def _assert_resolves_within_root(root_resolved: Path, candidate: Path) -> Path:
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(
            "Refusing path that resolves outside the ingestion root: "
            f"{candidate} -> {resolved}"
        )
    return resolved


def _assert_no_symlink_components(path: Path) -> None:
    """Reject any symlink component for an arbitrary path."""
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"Refusing symlink component: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent


def generate_stable_id(file_path: str | Path) -> str:
    """Return a stable document ID derived from the file's sha256 digest.

    Args:
        file_path: Local filesystem path to hash.

    Returns:
        Stable document identifier in the form ``doc-<sha256[:16]>``.
    """
    digest = sha256_file(Path(file_path))
    return f"doc-{digest[:16]}"


def sanitize_document_metadata(
    meta: dict[str, Any] | None, *, source_filename: str
) -> dict[str, Any]:
    """Return metadata scrubbed of path-like fields and normalized for persistence.

    Args:
        meta: Raw metadata mapping (possibly containing path-like values).
        source_filename: Safe basename to persist as the canonical source identifier.

    Returns:
        A new mapping with path-like keys removed and ``source`` normalized to a
        basename when present.
    """
    out = dict(meta or {})
    for key in (
        "source_path",
        "file_path",
        "path",
        "filename",
        "file_directory",
        "directory",
        "uri",
        "url",
    ):
        out.pop(key, None)
    src = out.get("source")
    if isinstance(src, str) and ("/" in src or "\\" in src or src.startswith("file:")):
        try:
            out["source"] = Path(src).name or source_filename
        except (ValueError, TypeError):
            out["source"] = source_filename
    elif src is not None and not isinstance(src, str):
        out.pop("source", None)
    out["source_filename"] = source_filename
    return out


def collect_paths(
    root: Path | str,
    *,
    recursive: bool = True,
    extensions: set[str] | None = None,
) -> list[Path]:
    """Collect safe file paths under a root directory.

    Args:
        root: Directory to enumerate.
        recursive: When ``True``, traverse subdirectories recursively.
        extensions: Optional set of allowed file extensions (for example,
            ``{".pdf", ".txt"}``).

    Returns:
        Sorted list of resolved file paths that match the extension filter.

    Raises:
        NotADirectoryError: When ``root`` exists but is not a directory.
        ValueError: When ``root`` is a symlink.

    Security:
        - Rejects symlinks in any component between root and each file.
        - Rejects resolved paths that escape the resolved root.
    """
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return []
    if not root_path.is_dir():
        raise NotADirectoryError(str(root_path))
    if root_path.is_symlink():
        raise ValueError(f"Refusing symlink ingestion root: {root_path}")

    root_resolved = root_path.resolve(strict=True)
    exts = _normalize_extensions(extensions)
    pattern = "**/*" if recursive else "*"

    collected: list[Path] = []
    for entry in root_path.glob(pattern):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in exts:
            continue
        try:
            _assert_no_symlinks_under_root(root_path, entry)
            resolved = _assert_resolves_within_root(root_resolved, entry)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unsafe ingestion path {}: {}", entry, exc)
            continue
        collected.append(resolved)

    collected.sort(key=lambda p: str(p))
    return collected


def _default_unstructured_reader() -> Any | None:
    try:
        from llama_index.readers.file import UnstructuredReader

        return UnstructuredReader()
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


async def load_documents(
    paths: Sequence[Path | str],
    *,
    reader: Any | None = None,
) -> list[Document]:
    """Load local files into LlamaIndex Documents (async-friendly).

    Args:
        paths: File paths to load.
        reader: Optional Unstructured reader override.

    Returns:
        List of LlamaIndex ``Document`` objects.

    Notes:
        - Missing/non-file paths are skipped.
        - UnstructuredReader is used when available; failures fall back to UTF-8
          text read.
        - Symlink components are rejected to prevent traversal.
    """
    from llama_index.core import Document

    resolved_reader = reader if reader is not None else _default_unstructured_reader()

    docs: list[Document] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            continue
        if not path.is_file():
            continue
        _assert_no_symlink_components(path)

        doc_id_base = await asyncio.to_thread(generate_stable_id, path)
        if resolved_reader is not None:
            try:
                loaded = await asyncio.to_thread(
                    resolved_reader.load_data,  # type: ignore[call-arg]
                    file=path,
                    unstructured_kwargs={"filename": path.name},
                )
                loaded_items = list(loaded or [])
                for idx, item in enumerate(loaded_items):
                    meta = getattr(item, "metadata", {}) or {}
                    item.doc_id = (
                        f"{doc_id_base}-{idx}" if len(loaded_items) > 1 else doc_id_base
                    )
                    item.metadata = sanitize_document_metadata(
                        meta,
                        source_filename=path.name,
                    )
                    docs.append(item)
                if loaded_items:
                    continue
            except Exception as exc:
                logger.debug("UnstructuredReader failed for {}: {}", path.name, exc)

        try:
            text = await asyncio.to_thread(
                path.read_text, encoding="utf-8", errors="ignore"
            )
        except Exception as exc:
            logger.debug("Fallback text read failed for {}: {}", path.name, exc)
            text = ""
        docs.append(
            Document(
                text=text,
                doc_id=doc_id_base,
                metadata=sanitize_document_metadata({}, source_filename=path.name),
            )
        )
    return docs


async def load_documents_from_inputs(
    inputs: Sequence[IngestionInput],
    *,
    reader: Any | None = None,
) -> list[Document]:
    """Load Documents from normalized ingestion inputs.

    Args:
        inputs: Normalized ingestion inputs describing local files or in-memory
            payloads.
        reader: Optional Unstructured reader override.

    Returns:
        List of loaded ``Document`` objects.

    This is a narrow bridge for ``ingestion_pipeline``: it keeps all file IO and
    metadata sanitization within this module.
    """
    from llama_index.core import Document

    resolved_reader = reader if reader is not None else _default_unstructured_reader()

    documents: list[Document] = []
    for item in inputs:
        if item.source_path is not None:
            path = Path(item.source_path).expanduser()
            if not path.exists() or not path.is_file():
                logger.warning("Skipping missing ingestion source: {}", path)
                continue
            _assert_no_symlink_components(path)
            file_docs = await load_documents([path], reader=resolved_reader)
            for idx, doc in enumerate(file_docs):
                doc.doc_id = (
                    f"{item.document_id}-{idx}"
                    if len(file_docs) > 1
                    else item.document_id
                )
                base_meta = dict(getattr(doc, "metadata", {}) or {})
                base_meta.update(item.metadata)
                base_meta.setdefault("document_id", item.document_id)
                doc.metadata = sanitize_document_metadata(
                    base_meta,
                    source_filename=path.name,
                )
            documents.extend(file_docs)
            continue

        payload = item.payload_bytes or b""
        text = payload.decode("utf-8", errors="ignore")
        meta = sanitize_document_metadata(item.metadata, source_filename="<bytes>")
        meta.setdefault("document_id", item.document_id)
        documents.append(Document(text=text, doc_id=item.document_id, metadata=meta))

    return documents


def clear_ingestion_cache() -> None:
    """Best-effort cleanup of the ingestion cache directory.

    Security: only deletes ``settings.cache_dir / "ingestion"``.
    """
    from src.config.settings import settings

    base = Path(settings.cache_dir).expanduser()
    target = base / "ingestion"

    if not target.exists():
        return

    if target.is_symlink():
        logger.warning("Refusing to delete symlink cache dir: {}", target)
        return

    try:
        base_resolved = base.resolve(strict=True)
        target_resolved = target.resolve(strict=True)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("Skipping ingestion cache cleanup for {}: {}", target, exc)
        return

    if not target_resolved.is_relative_to(base_resolved):
        logger.warning(
            "Refusing ingestion cache cleanup outside base: {} -> {} (base={})",
            target,
            target_resolved,
            base_resolved,
        )
        return

    shutil.rmtree(target_resolved, ignore_errors=True)


__all__ = [
    "clear_ingestion_cache",
    "collect_paths",
    "generate_stable_id",
    "load_documents",
    "load_documents_from_inputs",
    "sanitize_document_metadata",
]
