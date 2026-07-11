"""Library-first ingestion pipeline built on LlamaIndex primitives.

This module assembles an :class:`~llama_index.core.ingestion.IngestionPipeline`
using canonical DocMind configuration objects and returns normalized
:class:`~src.models.processing.IngestionResult` payloads. Parser routing lives in
``src.processing.parsing``; this module only assembles chunking, metadata,
artifact, cache, and snapshot orchestration.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from llama_index.core import Document
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TransformComponent
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
from loguru import logger
from opentelemetry import trace

from src.config import settings as app_settings
from src.config import setup_llamaindex
from src.config.integrations import get_settings_embed_model
from src.config.settings import ProcessingConfig
from src.config.settings_utils import (
    endpoint_url_allowed,
    parse_endpoint_allowlist_hosts,
)
from src.models.processing import (
    ExportArtifact,
    IngestionConfig,
    IngestionInput,
    IngestionResult,
    ManifestSummary,
)
from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.processing.ingestion_api import (
    load_documents_from_inputs,
    require_unique_document_ids,
)
from src.processing.pdf_pages import save_pdf_page_images
from src.utils.log_safety import safe_url_for_log

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core.base.embeddings.base import BaseEmbedding

    from src.nlp.spacy_service import SpacyNlpService
    from src.retrieval.image_index import PageImageRecord
else:
    BaseEmbedding = object

_TRACER = trace.get_tracer("docmind.ingestion")


def _manifest_parsing_config(inputs: Sequence[IngestionInput]) -> dict[str, Any]:
    """Return the canonical parser configuration represented by a manifest."""
    per_input = [
        {
            "document_id": item.document_id,
            "overrides": item.parsing_overrides.model_dump(
                mode="json", exclude_none=True
            ),
        }
        for item in inputs
    ]
    per_input.sort(key=lambda item: str(item["document_id"]))
    return {
        **app_settings.parsing.model_dump(mode="json"),
        "ocr": app_settings.ocr.model_dump(mode="json"),
        "pdf_backend": app_settings.pdf_backend.model_dump(mode="json"),
        "input_overrides": per_input,
    }


def _ingestion_corpus_hash(inputs: Sequence[IngestionInput]) -> str:
    """Hash file corpus state plus ordered in-memory payload identities."""
    corpus_paths = [Path(item.source_path) for item in inputs if item.source_path]
    base_dir: Path | None = None
    if corpus_paths:
        try:
            base_dir = Path(os.path.commonpath([str(p.parent) for p in corpus_paths]))
        except ValueError:
            base_dir = None
    file_corpus_hash = compute_corpus_hash(corpus_paths, base_dir=base_dir)
    payload_hashes = sorted(
        (
            item.document_id,
            hashlib.sha256(item.payload_text.encode("utf-8")).hexdigest(),
        )
        for item in inputs
        if item.payload_text is not None
    )
    if not payload_hashes:
        return file_corpus_hash
    return compute_config_hash(
        {
            "file_corpus_hash": file_corpus_hash,
            "payload_text": payload_hashes,
        }
    )


class SettingsWithProcessing(Protocol):
    """Protocol for settings with processing section."""

    @property
    def processing(self) -> ProcessingConfig: ...


_ROMAN_MAP: dict[str, int] = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


def _roman_to_int(value: str) -> int | None:
    """Convert a roman numeral to int; return None for invalid/empty values."""
    if not value:
        return None
    total = 0
    prev = 0
    for ch in reversed(value):
        cur = _ROMAN_MAP.get(ch)
        if cur is None:
            return None
        if cur < prev:
            total -= cur
        else:
            total += cur
            prev = cur
    return total if total > 0 else None


def _parse_page_number(raw: object) -> int | None:
    """Best-effort page number parser (ints, digits, roman numerals)."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() and raw > 0 else None
    text = str(raw).strip()
    if not text:
        return None
    digits = re.search(r"\d+", text)
    if digits:
        value = int(digits.group())
        return value if value > 0 else None
    roman = re.sub(r"[^IVXLCDM]", "", text.upper())
    return _roman_to_int(roman)


def _ensure_cache_path(cfg: IngestionConfig) -> Path:
    """Resolve and materialize the cache path used by the ingestion pipeline.

    Args:
        cfg: Runtime ingestion configuration.

    Returns:
        Path: Fully qualified DuckDB cache path guaranteed to exist.
    """
    cache_path = (
        Path(cfg.cache_dir) / app_settings.cache.filename
        if cfg.cache_dir is not None
        else app_settings.cache.ingestion_db_path
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return cache_path


def build_ingestion_pipeline(
    cfg: IngestionConfig,
    *,
    embedding: BaseEmbedding | None,
    nlp_service: SpacyNlpService | None = None,
) -> tuple[IngestionPipeline, Path]:
    """Construct an :class:`IngestionPipeline` configured with local persistence.

    Args:
        cfg: Ingestion options controlling chunking, caching, and persistence.
        embedding: Optional embedding component injected into the pipeline.
        nlp_service: Optional spaCy service used for NLP enrichment.

    Returns:
        tuple[IngestionPipeline, Path]: Pipeline instance and cache database path.
    """
    cache_path = _ensure_cache_path(cfg)
    kv_store = DuckDBKVStore(
        database_name=cache_path.name,
        persist_dir=str(cache_path.parent),
    )
    ingest_cache = IngestionCache(cache=kv_store, collection=cfg.cache_collection)
    transformations: list[TransformComponent] = [
        TokenTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separator="\n",
        ),
    ]

    # Optional NLP enrichment (spaCy) runs after chunking and before embeddings.
    try:
        from src.nlp.spacy_service import SpacyNlpService
        from src.processing.nlp_enrichment import SpacyNlpEnrichmentTransform

        spacy_cfg = getattr(app_settings, "spacy", None)
        if spacy_cfg is not None and bool(getattr(spacy_cfg, "enabled", False)):
            service = nlp_service or SpacyNlpService(spacy_cfg)
            transformations.append(
                SpacyNlpEnrichmentTransform(cfg=spacy_cfg, service=service)
            )
    except Exception as exc:  # pragma: no cover - fail open
        logger.debug("spaCy enrichment unavailable: {}", type(exc).__name__)

    if embedding is not None:
        transformations.append(embedding)

    pipeline = IngestionPipeline(
        transformations=transformations,
        cache=ingest_cache,
    )
    return pipeline, cache_path


def _resolve_embedding(embedding: BaseEmbedding | None) -> BaseEmbedding | None:
    """Return a usable embedding instance for ingestion.

    Args:
        embedding: Optional embedding provided by the caller.

    Returns:
        BaseEmbedding | None: Resolved embedding model, or ``None`` when no
        embedding is available and the pipeline should run without the
        transformation.
    """
    if embedding is not None:
        return embedding if embedding_allowed_for_ingestion(embedding) else None

    resolved = get_settings_embed_model()
    if resolved is not None and embedding_allowed_for_ingestion(resolved):
        return resolved

    try:
        setup_llamaindex(force_embed=True)
    except (
        RuntimeError,
        ValueError,
        ImportError,
    ) as exc:  # pragma: no cover - defensive
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(
            str(exc), key_id="ingestion.embedding_autosetup"
        )
        logger.warning(
            "Embedding auto-setup failed; continuing without embedding "
            "(error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        resolved = get_settings_embed_model()
        return resolved if embedding_allowed_for_ingestion(resolved) else None

    resolved = get_settings_embed_model()
    if resolved is not None and not embedding_allowed_for_ingestion(resolved):
        resolved = None
    if resolved is None:
        logger.warning(
            "Embedding unavailable after auto-setup; "
            "pipeline will run without embeddings"
        )
    return resolved


def embedding_allowed_for_ingestion(embedding: BaseEmbedding | None) -> bool:
    """Return True when an embedding model is allowed by endpoint policy.

    Local embedding implementations usually do not expose an endpoint URL and
    are accepted. OpenAI-compatible embedding objects expose ``api_base`` or
    ``base_url`` and must pass the same local-first endpoint policy used by the
    rest of the application.

    Args:
        embedding: Candidate embedding instance.

    Returns:
        True when ingestion may use the embedding model.
    """
    if embedding is None:
        return False

    endpoint = getattr(embedding, "api_base", None)
    if endpoint is None:
        endpoint = getattr(embedding, "base_url", None)
    if endpoint is None:
        return True

    if bool(app_settings.security.allow_remote_endpoints):
        return True

    allowed_hosts = parse_endpoint_allowlist_hosts(
        app_settings.security.endpoint_allowlist
    )
    allowed = endpoint_url_allowed(endpoint, allowed_hosts=allowed_hosts)
    if not allowed:
        logger.warning(
            "Embedding endpoint blocked by local-first endpoint policy: {}",
            safe_url_for_log(str(endpoint)),
        )
    return bool(allowed)


def _page_image_exports(
    path: Path,
    cfg: IngestionConfig,
    encrypt_override: bool,
    *,
    document_id: str | None = None,
) -> list[ExportArtifact]:
    """Generate optional page-image export artifacts for PDF inputs.

    Args:
        path: Source document path.
        cfg: Ingestion configuration controlling cache directories.
        encrypt_override: Per-document override to force image encryption.
        document_id: Stable document identifier for metadata attachment.

    Returns:
        list[ExportArtifact]: Export metadata describing rendered page images.
    """
    if path.suffix.lower() != ".pdf":
        return []

    base_dir = cfg.cache_dir or app_settings.cache.dir
    output_dir = base_dir / "page_images" / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    desired_flag = (
        encrypt_override
        or cfg.enable_image_encryption
        or getattr(app_settings.processing, "encrypt_page_images", False)
    )
    entries = save_pdf_page_images(path, output_dir, encrypt=desired_flag)

    exports: list[ExportArtifact] = []
    for entry in entries:
        image_path = Path(entry["image_path"])
        name = image_path.name.lower()
        content_type = "application/octet-stream"
        if name.endswith(".webp") or name.endswith(".webp.enc"):
            content_type = "image/webp"
        elif (
            name.endswith(".jpg")
            or name.endswith(".jpeg")
            or name.endswith(".jpg.enc")
            or name.endswith(".jpeg.enc")
        ):
            content_type = "image/jpeg"

        # Avoid returning path-like fields in persistable metadata.
        metadata = {
            k: v for k, v in entry.items() if k not in {"image_path", "thumbnail_path"}
        }
        # Attach stable identifiers for downstream indexing.
        if document_id:
            metadata.setdefault("doc_id", document_id)
            metadata.setdefault("document_id", document_id)
        metadata.setdefault("source_filename", path.name)

        page_no = (
            entry.get("page_no")
            or entry.get("page")
            or entry.get("page_number")
            or entry.get("page_num")
        )
        try:
            page_no_int = int(page_no) if page_no is not None else 0
        except (TypeError, ValueError):
            page_no_int = 0
        exports.append(
            ExportArtifact(
                name=f"pdf-page-{page_no_int}" if page_no_int else "pdf-page",
                path=image_path,
                content_type=content_type,
                metadata=metadata,
            )
        )
    return exports


async def _load_documents(
    cfg: IngestionConfig, inputs: Sequence[IngestionInput]
) -> tuple[list[Document], list[ExportArtifact]]:
    """Load source corpus into ``Document`` instances and capture exports.

    Args:
        cfg: Ingestion configuration used to resolve cache locations.
        inputs: Sequence of normalized ingestion inputs.

    Returns:
        tuple[list[Document], list[ExportArtifact]]: Parsed documents paired
        with any generated artifact exports.
    """
    documents = await load_documents_from_inputs(inputs)
    exports: list[ExportArtifact] = []

    for doc in documents:
        doc_id = str(doc.metadata.get("document_id") or "")
        page_raw = (
            doc.metadata.get("page_number")
            or doc.metadata.get("page")
            or doc.metadata.get("page_no")
            or doc.metadata.get("page_num")
        )
        page_no = _parse_page_number(page_raw)
        if doc_id and page_no and not doc.metadata.get("page_id"):
            doc.metadata["page_id"] = f"{doc_id}::page::{page_no}"

    for item in inputs:
        if item.source_path is None:
            continue
        exports.extend(
            _page_image_exports(
                Path(item.source_path),
                cfg,
                item.encrypt_images,
                document_id=item.document_id,
            )
        )
    return documents, exports


def _store_image_artifact(
    store: ArtifactStore,
    export: ExportArtifact,
    settings: SettingsWithProcessing,
) -> tuple[ArtifactRef, Path, ArtifactRef | None, Path | None]:
    img_ref = store.put_file(Path(export.path))
    img_path = store.resolve_path(img_ref)
    thumb_ref = None
    thumb_path = None
    try:
        from src.utils.images import ensure_thumbnail

        thumb_dir = Path(export.path).parent / "thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_local = ensure_thumbnail(
            Path(export.path),
            max_side=int(getattr(settings.processing, "thumbnail_max_side", 384)),
            thumb_dir=thumb_dir,
            encrypt=Path(export.path).suffix == ".enc",
        )
        thumb_ref = store.put_file(Path(thumb_local))
        thumb_path = store.resolve_path(thumb_ref)
    except Exception as exc:
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="ingestion.thumbnail")
        logger.debug(
            "Thumbnail generation failed (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        thumb_ref = None
        thumb_path = None
    return img_ref, img_path, thumb_ref, thumb_path


def _build_page_image_records(
    exports: list[ExportArtifact],
    store: ArtifactStore,
    settings: SettingsWithProcessing,
) -> tuple[list[PageImageRecord], int]:
    from src.retrieval.image_index import PageImageRecord

    records: list[PageImageRecord] = []
    skipped = 0
    for export in exports:
        meta = dict(getattr(export, "metadata", {}) or {})
        if not isinstance(getattr(export, "metadata", None), dict):
            export.metadata = dict(meta)
        doc_id = str(meta.get("doc_id") or meta.get("document_id") or "")
        page_no_raw = meta.get("page_no") or meta.get("page") or meta.get("page_number")
        try:
            page_no = int(page_no_raw) if page_no_raw is not None else 0
        except (TypeError, ValueError):
            page_no = 0
        if not doc_id or page_no <= 0:
            skipped += 1
            continue

        try:
            img_ref, img_path, thumb_ref, thumb_path = _store_image_artifact(
                store, export, settings
            )
        except (OSError, ValueError) as exc:
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(exc), key_id="ingestion.artifact_store")
            logger.debug(
                "ArtifactStore put failed (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            skipped += 1
            continue

        # NOTE: This mutates export.metadata in-place so downstream consumers of
        # `exports` see stable artifact references.
        export.metadata.update(
            {
                "image_artifact_id": img_ref.sha256,
                "image_artifact_suffix": img_ref.suffix,
            }
        )
        if thumb_ref is not None:
            export.metadata.update(
                {
                    "thumbnail_artifact_id": thumb_ref.sha256,
                    "thumbnail_artifact_suffix": thumb_ref.suffix,
                }
            )

        try:
            records.append(
                PageImageRecord(
                    doc_id=doc_id,
                    page_no=page_no,
                    image=img_ref,
                    image_path=img_path,
                    thumbnail=thumb_ref,
                    thumbnail_path=thumb_path,
                    phash=meta.get("phash"),
                    page_text=meta.get("page_text"),
                    bbox=meta.get("bbox"),
                )
            )
        except Exception as exc:  # pragma: no cover
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(
                str(exc), key_id="ingestion.page_image_record"
            )
            logger.debug(
                "PageImageRecord build failed (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            skipped += 1
    return records, skipped


def _index_page_images_orchestrator(
    records: list[PageImageRecord],
    cfg: IngestionConfig,
    *,
    purge_doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    from qdrant_client import QdrantClient

    from src.retrieval.image_index import (
        delete_page_images_for_doc_id,
        index_page_images_siglip,
    )
    from src.utils.siglip_adapter import SiglipEmbedding
    from src.utils.storage import get_client_config

    t0 = time.time()
    client = QdrantClient(**get_client_config())
    purged_points = 0
    indexed = 0
    try:
        embedder = SiglipEmbedding()
        if purge_doc_ids:
            for doc_id in sorted({str(d) for d in purge_doc_ids if d}):
                with contextlib.suppress(Exception):
                    purged_points += int(
                        delete_page_images_for_doc_id(
                            client,
                            app_settings.database.qdrant_image_collection,
                            doc_id=doc_id,
                        )
                    )
        indexed = index_page_images_siglip(
            client,
            collection_name=app_settings.database.qdrant_image_collection,
            records=records,
            embedder=embedder,
            batch_size=int(getattr(cfg, "image_index_batch_size", 8)),
        )
    finally:
        with contextlib.suppress(Exception):
            client.close()

    return {
        "image_index.collection": app_settings.database.qdrant_image_collection,
        "image_index.indexed": int(indexed),
        "image_index.purged_points": int(purged_points),
        "image_index.latency_ms": int((time.time() - t0) * 1000),
    }


def _index_page_images(
    exports: list[ExportArtifact],
    cfg: IngestionConfig,
    *,
    purge_doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Index rendered PDF page images into Qdrant (best-effort).

    Wiring:
    - Convert page-image exports to content-addressed artifact refs.
    - Index page images into Qdrant image collection using SigLIP embeddings.
    - Store **only** artifact references in Qdrant payload (no base64, no raw paths).

    Returns:
        dict[str, Any]: PII-safe counters/flags to include in the ingestion result.
    """
    if not getattr(cfg, "enable_image_indexing", True):
        return {"image_index.enabled": False, "image_index.indexed": 0}

    image_exports = [e for e in exports if str(e.content_type).startswith("image/")]
    if not image_exports:
        return {"image_index.enabled": True, "image_index.indexed": 0}

    store = ArtifactStore.from_settings(app_settings)

    records, skipped = _build_page_image_records(image_exports, store, app_settings)

    if not records:
        return {
            "image_index.enabled": True,
            "image_index.indexed": 0,
            "image_index.skipped": skipped,
        }

    try:
        orchestration = _index_page_images_orchestrator(
            records, cfg, purge_doc_ids=purge_doc_ids
        )
        return {
            "image_index.enabled": True,
            "image_index.skipped": skipped,
            **orchestration,
        }
    except Exception as exc:  # pragma: no cover - fail open
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="ingestion.image_indexing")
        logger.info("Image indexing skipped: {}", type(exc).__name__)
        logger.debug(
            "Image indexing error (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        return {
            "image_index.enabled": True,
            "image_index.indexed": 0,
            "image_index.skipped": skipped,
            "image_index.error_type": type(exc).__name__,
        }


def _collect_parsing_provenance(documents: Sequence[Document]) -> dict[str, Any]:
    """Collect sanitized parser provenance from loaded documents."""
    by_doc: dict[str, dict[str, Any]] = {}
    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        parsing = meta.get("parsing")
        document_id = str(meta.get("document_id") or getattr(doc, "doc_id", ""))
        if not document_id or not isinstance(parsing, dict):
            continue
        by_doc.setdefault(document_id, dict(parsing))
    if not by_doc:
        return {}

    profiles = {
        str(item.get("profile"))
        for item in by_doc.values()
        if item.get("profile") is not None
    }
    if len(profiles) > 1:
        raise ValueError("ingestion produced multiple parser profiles")
    profile = next(iter(profiles), None)
    frameworks = sorted(
        {
            str(item.get("framework"))
            for item in by_doc.values()
            if item.get("framework") is not None
        }
    )
    return {
        "parsing.document_count": len(by_doc),
        "parsing.frameworks": frameworks,
        "parsing.profile": profile,
        "parsing.provenance": by_doc,
    }


def _prune_artifacts_best_effort() -> int:
    """Prune artifact store to enforce size budget (best-effort)."""
    try:
        artifacts_cfg = getattr(app_settings, "artifacts", None)
        max_mb = int(getattr(artifacts_cfg, "max_total_mb", 0) or 0)
        if max_mb <= 0:
            return 0
        store = ArtifactStore.from_settings(app_settings)
        return int(
            store.prune(
                max_total_bytes=max_mb * 1024 * 1024,
                min_age_seconds=int(
                    getattr(artifacts_cfg, "gc_min_age_seconds", 0) or 0
                ),
            )
        )
    except Exception as exc:
        logger.debug(
            "Artifact pruning failed (ArtifactStore.from_settings/prune): {}", exc
        )
        return 0


async def ingest_documents(
    cfg: IngestionConfig,
    inputs: Sequence[IngestionInput],
    *,
    embedding: BaseEmbedding | None = None,
    nlp_service: SpacyNlpService | None = None,
) -> IngestionResult:
    """Run the configured ingestion pipeline and normalize the result payload.

    Args:
        cfg: Ingestion configuration specifying chunking and persistence.
        inputs: Normalized ingestion inputs to process.
        embedding: Optional embedding instance overriding global Settings.embed_model.
        nlp_service: Optional spaCy service used for NLP enrichment.

    Returns:
        IngestionResult: Structured ingestion output including nodes, manifest
        summary, exports (enriched in-place with artifact reference fields),
        metadata, and execution timing.
    """
    require_unique_document_ids(inputs)
    resolved_embedding = _resolve_embedding(embedding)
    if resolved_embedding is None:
        logger.warning(
            "No embedding model configured; proceeding without embedding transform"
        )

    pipeline, cache_path = build_ingestion_pipeline(
        cfg, embedding=resolved_embedding, nlp_service=nlp_service
    )
    documents, exports = await _load_documents(cfg, inputs)

    start = time.perf_counter()
    with _TRACER.start_as_current_span("ingest_documents") as span:
        span.set_attribute("docmind.document_count", len(documents))
        nodes = list(await pipeline.arun(documents=documents))
    duration_ms = (time.perf_counter() - start) * 1000.0

    if inputs and not nodes:
        raise RuntimeError("Ingestion produced no nodes for a non-empty input batch")

    img_index_meta = _index_page_images(exports, cfg)
    artifact_gc_deleted = _prune_artifacts_best_effort()

    manifest = ManifestSummary(
        corpus_hash=_ingestion_corpus_hash(inputs),
        config_hash=compute_config_hash(
            {
                "ingestion": cfg.model_dump(mode="json"),
                "parsing": _manifest_parsing_config(inputs),
            }
        ),
        payload_count=len(nodes),
        complete=False,
    )

    metadata = {
        "document_count": len(inputs),
        # Avoid emitting absolute local filesystem paths in
        # structured results that could be logged or persisted.
        "cache_db": cache_path.name,
        **img_index_meta,
        **_collect_parsing_provenance(documents),
        "artifact_gc_deleted": int(artifact_gc_deleted),
    }

    nlp_enabled = bool(getattr(getattr(app_settings, "spacy", None), "enabled", False))
    if nlp_enabled:
        enriched_nodes = 0
        entity_count = 0
        for node in nodes:
            meta = getattr(node, "metadata", None) or {}
            payload = meta.get("docmind_nlp")
            if isinstance(payload, dict):
                enriched_nodes += 1
                ents = payload.get("entities")
                if isinstance(ents, list):
                    entity_count += len(ents)
        metadata.update(
            {
                "nlp.enabled": True,
                "nlp.enriched_nodes": int(enriched_nodes),
                "nlp.entity_count": int(entity_count),
            }
        )
    else:
        metadata["nlp.enabled"] = False

    return IngestionResult(
        nodes=nodes,
        documents=documents,
        manifest=manifest,
        exports=exports,
        metadata=metadata,
        duration_ms=duration_ms,
    )


def ingest_documents_sync(
    cfg: IngestionConfig,
    inputs: Sequence[IngestionInput],
    *,
    embedding: BaseEmbedding | None = None,
    nlp_service: SpacyNlpService | None = None,
) -> IngestionResult:
    """Run :func:`ingest_documents` synchronously via ``asyncio.run``.

    Args:
        cfg: Ingestion configuration specifying chunking and persistence.
        inputs: Normalized ingestion inputs to process.
        embedding: Optional embedding instance overriding global Settings.embed_model.
        nlp_service: Optional spaCy service used for NLP enrichment.

    Returns:
        IngestionResult: Structured ingestion output from the async pipeline.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # No running loop, safe to create one
        return asyncio.run(
            ingest_documents(cfg, inputs, embedding=embedding, nlp_service=nlp_service)
        )
    raise RuntimeError(
        "ingest_documents_sync cannot be called while an event loop is running; "
        "await ingest_documents(...) instead"
    )


def reindex_page_images_sync(
    cfg: IngestionConfig,
    inputs: Sequence[IngestionInput],
) -> dict[str, Any]:
    """Rebuild page-image exports and reindex them into Qdrant (best-effort).

    This is an operational helper for artifact repair and lifecycle maintenance.
    It does **not** ingest text or rebuild vector/graph indices.

    Returns:
        dict[str, Any]: Mapping containing updated exports and PII-safe metadata.
    """
    exports: list[ExportArtifact] = []
    purge_doc_ids: set[str] = set()
    for item in inputs:
        if item.source_path is None:
            continue
        purge_doc_ids.add(str(item.document_id))
        exports.extend(
            _page_image_exports(
                Path(item.source_path),
                cfg,
                item.encrypt_images,
                document_id=item.document_id,
            )
        )
    meta = _index_page_images(exports, cfg, purge_doc_ids=purge_doc_ids)
    meta["artifact_gc_deleted"] = int(_prune_artifacts_best_effort())
    return {
        "document_count": len([i for i in inputs if i.source_path is not None]),
        "export_count": len(exports),
        "metadata": meta,
        "exports": exports,
    }


__all__ = [
    "build_ingestion_pipeline",
    "ingest_documents",
    "ingest_documents_sync",
    "reindex_page_images_sync",
]
