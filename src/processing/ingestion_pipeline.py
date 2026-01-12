"""Library-first ingestion pipeline built on LlamaIndex primitives.

This module assembles an :class:`~llama_index.core.ingestion.IngestionPipeline`
using canonical DocMind configuration objects and returns normalized
:class:`~src.models.processing.IngestionResult` payloads. It replaces the legacy
custom DocumentProcessor while leaning entirely on maintained LlamaIndex and
Unstructured integrations (KISS/library-first).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llama_index.core import Document
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
from loguru import logger
from opentelemetry import trace

try:
    from llama_index.readers.file import UnstructuredReader
except ImportError:  # pragma: no cover - optional dependency
    UnstructuredReader = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.readers.file import UnstructuredReader as ReaderType
else:
    ReaderType = Any

from src.config import settings as app_settings
from src.config import setup_llamaindex
from src.config.integrations import get_settings_embed_model
from src.models.processing import (
    ExportArtifact,
    IngestionConfig,
    IngestionInput,
    IngestionResult,
    ManifestSummary,
)
from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.processing.pdf_pages import save_pdf_page_images

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core.base.embeddings.base import BaseEmbedding
else:
    BaseEmbedding = Any

_TRACER = trace.get_tracer("docmind.ingestion")


def _ensure_cache_path(cfg: IngestionConfig) -> Path:
    """Resolve and materialize the cache path used by the ingestion pipeline.

    Args:
        cfg: Runtime ingestion configuration.

    Returns:
        Path: Fully qualified DuckDB cache path guaranteed to exist.
    """
    base_dir = cfg.cache_dir or app_settings.cache_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_filename = getattr(cfg, "cache_filename", None)
    if cache_filename is None:
        cache_settings = getattr(app_settings, "cache", None)
        cache_filename = getattr(cache_settings, "filename", "docmind.duckdb")
    return base_dir / Path(cache_filename)


def _ensure_docstore(cfg: IngestionConfig) -> tuple[SimpleDocumentStore, Path | None]:
    """Return a document store instance, restoring persisted state when present.

    Args:
        cfg: Runtime ingestion configuration.

    Returns:
        tuple[SimpleDocumentStore, Path | None]: Instantiated docstore and the
        persistence path when configured. The path is ``None`` when persistence
        is disabled.
    """
    path = cfg.docstore_path
    if path is None:
        return SimpleDocumentStore(), None
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return SimpleDocumentStore.from_persist_path(str(path)), path
    return SimpleDocumentStore(), path


def build_ingestion_pipeline(
    cfg: IngestionConfig,
    *,
    embedding: BaseEmbedding | None,
) -> tuple[IngestionPipeline, Path, Path | None]:
    """Construct an :class:`IngestionPipeline` configured with local persistence.

    Args:
        cfg: Ingestion options controlling chunking, caching, and persistence.
        embedding: Optional embedding component injected into the pipeline.

    Returns:
        tuple[IngestionPipeline, Path, Path | None]: Pipeline instance, cache
        database path, and optional docstore persist path.
    """
    cache_path = _ensure_cache_path(cfg)
    kv_store = DuckDBKVStore(database_name=str(cache_path))
    ingest_cache = IngestionCache(cache=kv_store, collection=cfg.cache_collection)
    docstore, docstore_path = _ensure_docstore(cfg)

    transformations: list[Any] = [
        TokenTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separator="\n",
        ),
    ]

    # Title extraction remains optional; failures simply skip the transform.
    try:
        extractor = TitleExtractor(show_progress=False)
        transformations.append(extractor)
    except (ImportError, ValueError, RuntimeError) as exc:  # pragma: no cover
        logger.debug("TitleExtractor unavailable: %s", exc)

    if embedding is not None:
        transformations.append(embedding)

    pipeline = IngestionPipeline(
        transformations=transformations,
        cache=ingest_cache,
        docstore=docstore,
    )
    return pipeline, cache_path, docstore_path


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
        return embedding

    resolved = get_settings_embed_model()
    if resolved is not None:
        return resolved

    try:
        setup_llamaindex(force_embed=True)
    except (
        RuntimeError,
        ValueError,
        ImportError,
    ) as exc:  # pragma: no cover - defensive
        logger.warning(
            "Embedding auto-setup failed; continuing without embedding: %s", exc
        )
        return get_settings_embed_model()

    resolved = get_settings_embed_model()
    if resolved is None:
        logger.warning(
            "Embedding unavailable after auto-setup; "
            "pipeline will run without embeddings"
        )
    return resolved


def _document_from_input(
    reader: ReaderType | None, item: IngestionInput
) -> list[Document]:
    """Convert an ingestion input into LlamaIndex ``Document`` objects.

    Args:
        reader: Optional Unstructured reader used for rich parsing.
        item: Normalized ingestion payload describing the source corpus item.

    Returns:
        list[Document]: Documents populated with normalized metadata.
    """
    if item.source_path is not None and reader is not None:
        try:
            source_path = Path(item.source_path)
            docs = reader.load_data(  # type: ignore[call-arg]
                file=source_path,
                unstructured_kwargs={"filename": str(source_path)},
            )
        except (
            TypeError,
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:  # pragma: no cover
            logger.debug("UnstructuredReader failed: %s", exc)
            safe_name = Path(item.source_path).name if item.source_path else "<bytes>"
            logger.info(
                "Falling back to plain-text read for %s (doc_id=%s) due to "
                "UnstructuredReader failure",
                safe_name,
                item.document_id,
            )
            text = Path(item.source_path).read_text(encoding="utf-8", errors="ignore")
            docs = [Document(text=text, doc_id=item.document_id)]
    elif item.source_path is not None:
        text = Path(item.source_path).read_text(encoding="utf-8", errors="ignore")
        docs = [Document(text=text, doc_id=item.document_id)]
    else:
        payload = item.payload_bytes or b""
        text = payload.decode("utf-8", errors="ignore")
        docs = [Document(text=text, doc_id=item.document_id)]

    for doc in docs:
        doc.doc_id = item.document_id
        doc.metadata.update(item.metadata)
        doc.metadata.setdefault("document_id", item.document_id)
        # Final-release: do not persist raw filesystem paths in node metadata.
        # Keep stable identifiers (document_id, sha256, source_filename) instead.
        src = doc.metadata.get("source")
        if isinstance(src, str):
            # Unstructured/LlamaIndex often set `source` to a local path/URI.
            # Normalize to a safe basename so durable stores never see absolute paths.
            with contextlib.suppress(Exception):
                doc.metadata["source"] = Path(src).name
        elif src is not None:
            with contextlib.suppress(Exception):
                doc.metadata.pop("source", None)
        for k in ("source_path", "file_path", "path"):
            with contextlib.suppress(Exception):
                doc.metadata.pop(k, None)
    return docs


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

    base_dir = cfg.cache_dir or app_settings.cache_dir
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

        # Final-release: avoid returning path-like fields in persistable metadata.
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


def _load_documents(
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
    reader = UnstructuredReader() if UnstructuredReader is not None else None
    documents: list[Document] = []
    exports: list[ExportArtifact] = []
    for item in inputs:
        documents.extend(_document_from_input(reader, item))
        if item.source_path is not None:
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
    settings: Any,
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
            encrypt=bool(str(export.path).endswith(".enc")),
        )
        thumb_ref = store.put_file(Path(thumb_local))
        thumb_path = store.resolve_path(thumb_ref)
    except Exception as exc:
        logger.debug("Thumbnail generation failed: %s", exc)
        thumb_ref = None
        thumb_path = None
    return img_ref, img_path, thumb_ref, thumb_path


def _build_page_image_records(
    exports: list[ExportArtifact],
    store: ArtifactStore,
    settings: Any,
) -> tuple[list[Any], int]:
    from src.retrieval.image_index import PageImageRecord

    records: list[Any] = []
    skipped = 0
    for export in exports:
        meta = dict(getattr(export, "metadata", {}) or {})
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
            logger.debug("ArtifactStore put failed: %s", exc)
            skipped += 1
            continue

        # Update export metadata with stable references (safe for persistence).
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
            logger.debug("PageImageRecord build failed: %s", exc)
            skipped += 1
    return records, skipped


def _index_page_images_orchestrator(
    records: list[Any],
    store: ArtifactStore,
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
    embedder = SiglipEmbedding()
    try:
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

    pruned = 0
    try:
        artifacts_cfg = getattr(app_settings, "artifacts", None)
        max_mb = int(getattr(artifacts_cfg, "max_total_mb", 0) or 0)
        if max_mb > 0:
            pruned = store.prune(
                max_total_bytes=max_mb * 1024 * 1024,
                min_age_seconds=int(
                    getattr(artifacts_cfg, "gc_min_age_seconds", 0) or 0
                ),
            )
    except Exception:
        pruned = 0

    return {
        "image_index.collection": app_settings.database.qdrant_image_collection,
        "image_index.indexed": int(indexed),
        "image_index.purged_points": int(purged_points),
        "image_index.latency_ms": int((time.time() - t0) * 1000),
        "image_index.artifact_gc_deleted": int(pruned),
    }


def _index_page_images(
    exports: list[ExportArtifact],
    cfg: IngestionConfig,
    *,
    purge_doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Index rendered PDF page images into Qdrant (best-effort).

    Final-release wiring:
    - Convert page-image exports to content-addressed artifact refs.
    - Index page images into Qdrant image collection using SigLIP embeddings.
    - Store **only** artifact references in Qdrant payload (no base64, no raw paths).

    Returns:
        dict[str, Any]: PII-safe counters/flags to include in the ingestion result.
    """
    if not bool(getattr(cfg, "enable_image_indexing", True)):
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
            records, store, cfg, purge_doc_ids=purge_doc_ids
        )
        return {
            "image_index.enabled": True,
            "image_index.skipped": skipped,
            **orchestration,
        }
    except Exception as exc:  # pragma: no cover - fail open
        logger.info("Image indexing skipped: %s", type(exc).__name__)
        logger.debug("Image indexing error: %s", exc)
        return {
            "image_index.enabled": True,
            "image_index.indexed": 0,
            "image_index.skipped": skipped,
            "image_index.error_type": type(exc).__name__,
        }


async def ingest_documents(
    cfg: IngestionConfig,
    inputs: Sequence[IngestionInput],
    *,
    embedding: BaseEmbedding | None = None,
) -> IngestionResult:
    """Run the configured ingestion pipeline and normalize the result payload.

    Args:
        cfg: Ingestion configuration specifying chunking and persistence.
        inputs: Normalized ingestion inputs to process.
        embedding: Optional embedding instance overriding global Settings.embed_model.

    Returns:
        IngestionResult: Structured ingestion output including nodes, manifest
        summary, exports, metadata, and execution timing.
    """
    resolved_embedding = _resolve_embedding(embedding)
    if resolved_embedding is None:
        logger.warning(
            "No embedding model configured; proceeding without embedding transform"
        )

    pipeline, cache_path, docstore_path = build_ingestion_pipeline(
        cfg, embedding=resolved_embedding
    )
    documents, exports = _load_documents(cfg, inputs)

    start = time.perf_counter()
    with _TRACER.start_as_current_span("ingest_documents") as span:
        span.set_attribute("docmind.document_count", len(documents))
        nodes = await pipeline.arun(documents=documents)
    duration_ms = (time.perf_counter() - start) * 1000.0

    if docstore_path is not None and pipeline.docstore is not None:
        pipeline.docstore.persist(str(docstore_path))

    img_index_meta = _index_page_images(exports, cfg)

    corpus_paths = [Path(item.source_path) for item in inputs if item.source_path]
    base_dir: Path | None = None
    if corpus_paths:
        try:
            base_dir = Path(os.path.commonpath([str(p.parent) for p in corpus_paths]))
        except ValueError:
            base_dir = None
    manifest = ManifestSummary(
        corpus_hash=compute_corpus_hash(corpus_paths, base_dir=base_dir),
        config_hash=compute_config_hash(cfg.model_dump()),
        payload_count=len(nodes),
        complete=False,
    )

    metadata = {
        "document_count": len(inputs),
        # Final-release: avoid emitting absolute local filesystem paths in
        # structured results that could be logged or persisted.
        "cache_db": cache_path.name,
        "docstore_enabled": bool(docstore_path),
        "docstore_filename": docstore_path.name if docstore_path else None,
        **img_index_meta,
    }

    return IngestionResult(
        nodes=list(nodes),
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
) -> IngestionResult:
    """Run :func:`ingest_documents` synchronously via ``asyncio.run``.

    Args:
        cfg: Ingestion configuration specifying chunking and persistence.
        inputs: Normalized ingestion inputs to process.
        embedding: Optional embedding instance overriding global Settings.embed_model.

    Returns:
        IngestionResult: Structured ingestion output from the async pipeline.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # No running loop, safe to create one
        return asyncio.run(ingest_documents(cfg, inputs, embedding=embedding))
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
