"""Library-first ingestion pipeline built on LlamaIndex primitives.

This module assembles an :class:`~llama_index.core.ingestion.IngestionPipeline`
using canonical DocMind configuration objects and returns normalized
:class:`~src.models.processing.IngestionResult` payloads. It replaces the legacy
custom DocumentProcessor while leaning entirely on maintained LlamaIndex and
Unstructured integrations (KISS/library-first).
"""

from __future__ import annotations

import asyncio
import logging
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
from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.processing.pdf_pages import save_pdf_page_images

if TYPE_CHECKING:  # pragma: no cover
    from llama_index.core.base.embeddings.base import BaseEmbedding
else:
    BaseEmbedding = Any

_TRACER = trace.get_tracer("docmind.ingestion")
logger = logging.getLogger(__name__)


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
            logger.info(
                "Falling back to plain-text read for %s due to UnstructuredReader "
                "failure",
                item.source_path,
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
        if item.source_path is not None:
            doc.metadata.setdefault("source_path", str(item.source_path))
    return docs


def _page_image_exports(
    path: Path, cfg: IngestionConfig, encrypt_override: bool
) -> list[ExportArtifact]:
    """Generate optional page-image export artifacts for PDF inputs.

    Args:
        path: Source document path.
        cfg: Ingestion configuration controlling cache directories.
        encrypt_override: Per-document override to force image encryption.

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

        metadata = {k: v for k, v in entry.items() if k != "image_path"}
        exports.append(
            ExportArtifact(
                name=f"pdf-page-{entry['page']}",
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
                _page_image_exports(Path(item.source_path), cfg, item.encrypt_images)
            )
    return documents, exports


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
        "cache_path": str(cache_path),
        "docstore_path": str(docstore_path) if docstore_path else None,
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


__all__ = [
    "build_ingestion_pipeline",
    "ingest_documents",
    "ingest_documents_sync",
]
