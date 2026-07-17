"""Snapshot utility helpers for staleness detection and config/corpus collection."""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.embedding_defaults import (
    DEFAULT_BM42_MODEL_ID,
    DEFAULT_BM42_SOURCE_REPO,
    DEFAULT_BM42_SOURCE_REVISION,
    DEFAULT_SIGLIP_MODEL_ID,
    DEFAULT_SIGLIP_MODEL_REVISION,
)
from src.config.settings import settings as default_settings
from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.persistence.upload_journal import reconstruct_precommit_corpus_hash
from src.retrieval import vector_contract

__all__ = [
    "activation_config_dict",
    "collect_corpus_paths",
    "compute_config_hash",
    "compute_staleness",
    "current_config_dict",
    "timestamped_export_path",
]


def collect_corpus_paths(base: Path) -> list[Path]:
    """Collect all files under ``base`` for inclusion in corpus hashing.

    Recursively traverses ``base`` and collects all file paths. Used to
    compute a corpus hash for staleness detection (whether documents changed).

    Args:
        base: Root directory to traverse (typically settings.data_dir /
            'uploads').

    Returns:
        List of all file paths under ``base``, or empty list if ``base``
        does not exist.
    """
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file() and not p.is_symlink()]


def current_config_dict(settings_obj: Any | None = None) -> dict[str, Any]:
    """Build configuration dictionary for staleness detection.

    Extracts key config parameters that affect index rebuilds.

    Args:
        settings_obj: DocMindSettings instance, or None to use
            default_settings.

    Returns:
        Dict mapping index-affecting config keys to current values.
    """
    s = settings_obj or default_settings
    siglip_revision = s.embedding.siglip_model_revision
    if (
        siglip_revision is None
        and s.embedding.siglip_model_id == DEFAULT_SIGLIP_MODEL_ID
    ):
        siglip_revision = DEFAULT_SIGLIP_MODEL_REVISION
    return {
        "activation": {
            "encrypt_images": bool(s.processing.encrypt_page_images),
            "graph_requested": bool(s.graphrag_cfg.enabled),
            "input_overrides": [],
        },
        "embedding": {
            "model_name": s.embedding.model_name,
            "model_revision": s.embedding.model_revision,
            "local_model_path": s.embedding.local_model_path,
            "dimension": s.embedding.dimension,
            "max_length": s.embedding.max_length,
            "normalize_text": s.embedding.normalize_text,
        },
        "image_embedding": {
            "model_id": s.embedding.siglip_model_id,
            "model_revision": siglip_revision,
            "normalize_image": s.embedding.normalize_image,
        },
        "processing": {
            "chunk_size": s.processing.chunk_size,
            "chunk_overlap": s.processing.chunk_overlap,
            "encrypt_page_images": s.processing.encrypt_page_images,
        },
        "parsing": {
            "framework": s.parsing.framework,
            "profile": s.parsing.profile,
            "max_pages": s.parsing.max_pages,
            "max_render_pixels": s.parsing.max_render_pixels,
            "render_dpi": s.pdf_backend.render_dpi,
            "ocr_engine": s.ocr.engine,
            "force_ocr": s.ocr.force_ocr,
        },
        "indexing": {
            "enable_server_hybrid": s.retrieval.enable_server_hybrid,
            "enable_keyword_tool": s.retrieval.enable_keyword_tool,
            "spacy_enabled": s.spacy.enabled,
            "spacy_model": s.spacy.model,
            "spacy_disable_pipes": sorted(s.spacy.disable_pipes),
        },
        "qdrant": {
            "text_collection": s.database.qdrant_collection,
            "image_collection": s.database.qdrant_image_collection,
            "dense_vector": vector_contract.DENSE_VECTOR_NAME,
            "sparse_vector": vector_contract.SPARSE_VECTOR_NAME,
            "sparse_enabled": vector_contract.sparse_retrieval_enabled(s),
            "sparse_model": DEFAULT_BM42_MODEL_ID,
            "sparse_source_repo": DEFAULT_BM42_SOURCE_REPO,
            "sparse_source_revision": DEFAULT_BM42_SOURCE_REVISION,
            "sparse_encoding_contract": vector_contract.SPARSE_ENCODING_CONTRACT,
        },
    }


def activation_config_dict(
    settings_obj: Any,
    *,
    inputs: Iterable[Any],
    encrypt_images: bool,
    graph_requested: bool,
) -> dict[str, Any]:
    """Return the exact index-affecting configuration for one ingestion run."""
    config = current_config_dict(settings_obj)
    input_overrides: list[dict[str, Any]] = []
    for item in inputs:
        overrides = item.parsing_overrides.model_dump(mode="json", exclude_none=True)
        if not overrides:
            continue
        input_overrides.append(
            {
                "document_id": str(item.document_id),
                "overrides": overrides,
            }
        )
    input_overrides.sort(key=lambda item: item["document_id"])
    config["activation"] = {
        "encrypt_images": bool(encrypt_images),
        "graph_requested": bool(graph_requested),
        "input_overrides": input_overrides,
    }
    return config


def compute_staleness(
    manifest: dict[str, Any],
    corpus_paths: Iterable[Path],
    cfg: dict[str, Any],
    *,
    settings_obj: Any | None = None,
) -> bool:
    """Return ``True`` when corpus/config hashes diverge from manifest values.

    Uses the single canonical uploads-relative corpus identity. Returns ``True``
    immediately when configuration differs. A durable source journal temporarily
    bridges the narrow source-mutation-to-CURRENT activation window.

    Args:
        manifest: Saved manifest dict with 'config_hash' and 'corpus_hash' keys.
        corpus_paths: Iterable of file paths to hash (e.g., from collect_corpus_paths).
        cfg: Current config dict (e.g., from current_config_dict).
        settings_obj: DocMindSettings instance, or None to use default_settings.

    Returns:
        True if manifest is stale (config or corpus changed), False if up-to-date.
    """
    s = settings_obj or default_settings
    uploads_dir = s.data_dir / "uploads"
    corpus_list = list(corpus_paths)
    cfg_hash = compute_config_hash(cfg)
    if manifest.get("config_hash") != cfg_hash:
        return True
    chash_norm = compute_corpus_hash(corpus_list, base_dir=uploads_dir)
    if manifest.get("corpus_hash") == chash_norm:
        return False
    reconstructed_hash = reconstruct_precommit_corpus_hash(s.data_dir, manifest)
    return reconstructed_hash != manifest.get("corpus_hash")


def timestamped_export_path(
    out_dir: Path, extension: str, *, prefix: str = "graph_export"
) -> Path:
    """Return timestamped export path, with collision avoidance.

    Generates a filename using current UTC timestamp (YYYYMMDDTHHMMSSZ
    format). If the file already exists, appends a counter to avoid
    collisions. Enforces a safety limit of 1000 attempts to prevent
    infinite loops.

    Args:
        out_dir: Directory where the export file will be created.
        extension: File extension (e.g., 'json', 'yaml') without the dot.
        prefix: Filename prefix (default: 'graph_export').

    Returns:
        Path with timestamped filename (e.g.,
        'graph_export-20240115T120030Z.json' or
        'graph_export-20240115T120030Z-42.json' if collision).
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sep_candidates = {"/", "\\"}
    if os.path.sep:
        sep_candidates.add(os.path.sep)
    if os.path.altsep:
        sep_candidates.add(os.path.altsep)

    raw_ext = extension.strip()
    if not raw_ext:
        raise ValueError("extension must be a non-empty string")
    if any(sep in raw_ext for sep in sep_candidates) or ".." in raw_ext:
        raise ValueError("extension must not contain path separators or '..'")
    if "." in raw_ext.strip("."):
        raise ValueError("extension must be a single token without dots")
    ext = raw_ext.lstrip(".")

    raw_prefix = prefix.strip()
    if not raw_prefix:
        raise ValueError("prefix must be a non-empty string")
    if any(sep in raw_prefix for sep in sep_candidates) or ".." in raw_prefix:
        raise ValueError("prefix must not contain path separators or '..'")
    normalized_prefix = raw_prefix.rstrip("-")
    if not normalized_prefix:
        raise ValueError("prefix must be a non-empty string")
    candidate = out_dir / f"{normalized_prefix}-{ts}.{ext}"
    counter = 1
    max_attempts = 1000  # Safety limit to prevent infinite loops
    while candidate.exists() and counter < max_attempts:
        candidate = out_dir / f"{normalized_prefix}-{ts}-{counter}.{ext}"
        counter += 1
    if candidate.exists():
        raise RuntimeError(
            f"Failed to generate unique export path after {max_attempts} attempts"
        )
    return candidate
