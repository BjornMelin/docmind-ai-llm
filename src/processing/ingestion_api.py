"""Canonical ingestion API (loader + path hygiene).

This module centralizes local filesystem ingestion primitives:

- Path enumeration with deterministic ordering and strict symlink rejection.
- File loading into LlamaIndex ``Document`` objects via the CPU-safe DocMind
  parser service.
- Metadata sanitization to avoid persisting absolute paths.
- Targeted ingestion cache cleanup under ``settings.cache.dir / "ingestion"``.

Per ADR-045 + SPEC-026 this is the single source of truth for loader behavior.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.models.processing import CANONICAL_DOCUMENT_ID_KEY, IngestionInput
from src.processing.parsing.errors import DocumentParseError
from src.utils.hashing import document_id_from_sha256, sha256_file
from src.utils.log_safety import build_pii_log_entry

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core import Document

    from src.processing.parsing.canonical_types import DocumentParseResult

_DEFAULT_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".markdown", ".rst"}


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
        Stable document identifier in the form ``doc-<full-sha256>``.
    """
    path = Path(file_path)
    try:
        digest = sha256_file(path)
    except OSError as exc:
        raise DocumentParseError(
            path,
            stage="source_hash",
            reason="source_hash_failed",
            cause=exc,
        ) from exc
    return document_id_from_sha256(digest)


def require_unique_document_ids(inputs: Sequence[IngestionInput]) -> None:
    """Reject ambiguous document ownership before ingestion performs work."""
    seen: set[str] = set()
    for item in inputs:
        if item.document_id in seen:
            raise ValueError(
                f"Duplicate document_id in ingestion batch: {item.document_id}"
            )
        seen.add(item.document_id)


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
            redaction = build_pii_log_entry(str(exc), key_id="ingestion.unsafe_path")
            logger.warning(
                "Skipping unsafe ingestion path {} (error_type={}, error={})",
                entry.name,
                type(exc).__name__,
                redaction.redacted,
            )
            continue
        collected.append(resolved)

    collected.sort(key=str)
    return collected


async def load_documents(
    paths: Sequence[Path | str],
    *,
    doc_id: str | None = None,
    parsing_overrides: dict[str, Any] | None = None,
) -> list[Document]:
    """Load local files into LlamaIndex Documents (async-friendly).

    Args:
        paths: File paths to load.
        doc_id: Optional precomputed document ID (only used if paths has 1 item).
        parsing_overrides: Optional per-ingestion parsing overrides.

    Returns:
        List of LlamaIndex ``Document`` objects.

    Notes:
        - Missing/non-file paths are skipped.
        - The parser service uses Docling for binary formats and strict UTF-8
          reads for the explicit text allowlist.
        - Symlink components are rejected to prevent traversal.
    """
    from llama_index.core import Document

    docs: list[Document] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            continue
        if not path.is_file():
            continue
        try:
            _assert_no_symlink_components(path)
        except ValueError as exc:
            redaction = build_pii_log_entry(str(exc), key_id="ingestion.symlink_path")
            logger.warning(
                "Skipping symlinked ingestion path {} (error_type={}, error={})",
                path.name,
                type(exc).__name__,
                redaction.redacted,
            )
            continue

        if doc_id and len(paths) == 1:
            doc_id_base = doc_id
        else:
            doc_id_base = await asyncio.to_thread(generate_stable_id, path)
        try:
            parse_result = await _parse_path(
                path,
                document_id=doc_id_base,
                parsing_overrides=parsing_overrides,
            )
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError(
                path,
                stage="ingestion_facade",
                reason="parser_service_failed",
                cause=exc,
            ) from exc

        page_count = len(parse_result.pages)
        full_provenance = parse_result.provenance()
        for idx, page in enumerate(parse_result.pages):
            doc_identifier = f"{doc_id_base}-{idx}" if page_count > 1 else doc_id_base
            doc = Document(
                text=page.text_markdown,
                doc_id=doc_identifier,
                metadata=sanitize_document_metadata(
                    {
                        "document_id": doc_id_base,
                        CANONICAL_DOCUMENT_ID_KEY: doc_id_base,
                        "source_hash": parse_result.source_hash,
                        "page_number": page.page_index + 1,
                        "page_id": page.page_id,
                        "parsing": (
                            full_provenance
                            if idx == 0
                            else _page_parser_provenance(parse_result, page)
                        ),
                    },
                    source_filename=parse_result.source_filename,
                ),
            )
            _exclude_parser_metadata_from_embeddings(doc)
            docs.append(doc)
    return docs


async def load_documents_from_inputs(
    inputs: Sequence[IngestionInput],
) -> list[Document]:
    """Load Documents from normalized ingestion inputs.

    Args:
        inputs: Normalized ingestion inputs describing local files or in-memory
            payloads.

    Returns:
        List of loaded ``Document`` objects.

    This is a narrow bridge for ``ingestion_pipeline``: it keeps all file IO and
    metadata sanitization within this module.
    """
    from llama_index.core import Document

    require_unique_document_ids(inputs)
    documents: list[Document] = []
    for item in inputs:
        if item.source_path is not None:
            path = Path(item.source_path).expanduser()
            if not path.exists() or not path.is_file():
                logger.warning("Skipping missing ingestion source: {}", path.name)
                continue
            try:
                _assert_no_symlink_components(path)
            except ValueError as exc:
                redaction = build_pii_log_entry(
                    str(exc), key_id="ingestion.symlink_path"
                )
                logger.warning(
                    "Skipping symlinked ingestion path {} (error_type={}, error={})",
                    path.name,
                    type(exc).__name__,
                    redaction.redacted,
                )
                continue
            file_docs = await load_documents(
                [path],
                doc_id=item.document_id,
                parsing_overrides=item.parsing_overrides.model_dump(exclude_none=True),
            )
            for idx, doc in enumerate(file_docs):
                doc.doc_id = (
                    f"{item.document_id}-{idx}"
                    if len(file_docs) > 1
                    else item.document_id
                )
                parser_meta = dict(getattr(doc, "metadata", {}) or {})
                base_meta = dict(item.metadata)
                base_meta.update(parser_meta)
                base_meta["document_id"] = item.document_id
                base_meta[CANONICAL_DOCUMENT_ID_KEY] = item.document_id
                doc.metadata = sanitize_document_metadata(
                    base_meta,
                    source_filename=path.name,
                )
                _exclude_parser_metadata_from_embeddings(doc)
            documents.extend(file_docs)
            continue

        text = item.payload_text
        if text is None:  # pragma: no cover - enforced by IngestionInput
            raise ValueError("IngestionInput requires source_path or payload_text")
        meta = dict(item.metadata)
        meta["document_id"] = item.document_id
        meta[CANONICAL_DOCUMENT_ID_KEY] = item.document_id
        meta = sanitize_document_metadata(meta, source_filename="<text>")
        document = Document(text=text, doc_id=item.document_id, metadata=meta)
        _exclude_parser_metadata_from_embeddings(document)
        documents.append(document)

    return documents


def _exclude_parser_metadata_from_embeddings(doc: Document) -> None:
    excluded = {"parsing", CANONICAL_DOCUMENT_ID_KEY}
    doc.excluded_embed_metadata_keys = sorted(
        excluded.union(getattr(doc, "excluded_embed_metadata_keys", []) or [])
    )
    doc.excluded_llm_metadata_keys = sorted(
        excluded.union(getattr(doc, "excluded_llm_metadata_keys", []) or [])
    )


def _page_parser_provenance(parse_result: Any, page: Any) -> dict[str, Any]:
    """Return compact parser-owned provenance for non-leading pages."""
    return {
        "framework": parse_result.parser_framework.value,
        "profile": parse_result.profile.value,
        "ocr_engine": parse_result.ocr_engine.value,
        "config_hash": parse_result.config_hash,
        "page": {
            "index": page.page_index,
            "routing_reason": page.routing_reason,
            "ocr_applied": page.ocr_applied,
        },
    }


async def _parse_path(
    path: Path,
    *,
    document_id: str,
    parsing_overrides: dict[str, Any] | None = None,
) -> DocumentParseResult:
    from src.config.settings import settings
    from src.processing.parsing.service import parse_document

    effective_settings = _settings_with_parsing_overrides(
        settings,
        parsing_overrides or {},
    )
    return await parse_document(
        path,
        settings=effective_settings,
        document_id=document_id,
    )


def _settings_with_parsing_overrides(
    settings_obj: Any, overrides: dict[str, Any]
) -> Any:
    if not overrides:
        return settings_obj
    ocr_update: dict[str, Any] = {}
    if overrides.get("force_ocr") is not None:
        ocr_update["force_ocr"] = bool(overrides["force_ocr"])
    if overrides.get("export_searchable_pdf") is not None:
        ocr_update["searchable_pdf_enabled"] = bool(overrides["export_searchable_pdf"])
    return settings_obj.model_copy(
        update={
            "ocr": settings_obj.ocr.model_copy(update=ocr_update),
        }
    )


def clear_ingestion_cache() -> None:
    """Best-effort cleanup of the ingestion cache directory.

    Security: only deletes ``settings.cache.dir / "ingestion"``.
    """
    from src.config.settings import settings

    base = settings.cache.dir.expanduser()
    target = settings.cache.ingestion_db_path.parent.expanduser()

    if not target.exists():
        return

    if target.is_symlink():
        logger.warning("Refusing to delete symlink cache dir: {}", target.name)
        return

    try:
        base_resolved = base.resolve(strict=True)
        target_resolved = target.resolve(strict=True)
    except FileNotFoundError:
        return
    except OSError as exc:
        redaction = build_pii_log_entry(str(exc), key_id="ingestion.cache_cleanup")
        logger.warning(
            "Skipping ingestion cache cleanup for {} (error_type={}, error={})",
            target.name,
            type(exc).__name__,
            redaction.redacted,
        )
        return

    if not target_resolved.is_relative_to(base_resolved):
        logger.warning(
            "Refusing ingestion cache cleanup outside base: {} -> {} (base={})",
            target.name,
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
    "require_unique_document_ids",
    "sanitize_document_metadata",
]
