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
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore

try:
    from llama_index.readers.file import UnstructuredReader
except ImportError:  # pragma: no cover - optional dependency
    UnstructuredReader = None  # type: ignore
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
from opentelemetry import trace

from src.config import settings as app_settings
from src.models.processing import (
    ExportArtifact,
    IngestionConfig,
    IngestionInput,
    IngestionResult,
    ManifestSummary,
)
from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.processing.pdf_pages import save_pdf_page_images

try:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:  # pragma: no cover - optional dependency
    HuggingFaceEmbedding = None  # type: ignore
    BaseEmbedding = Any  # type: ignore


_TRACER = trace.get_tracer("docmind.ingestion")
logger = logging.getLogger(__name__)


def _default_embedding() -> BaseEmbedding | None:
    """Return the default embedding model when the optional dependency loads.

    Returns:
        BaseEmbedding | None: HuggingFace embedding instance, or ``None`` when
        the optional package is unavailable or model initialization fails.
    """
    if HuggingFaceEmbedding is None:  # pragma: no cover - optional path
        return None
    try:
        return HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    except (OSError, ValueError, RuntimeError) as exc:  # pragma: no cover
        logger.debug("Default embedding unavailable: %s", exc)
        return None


def _ensure_cache_path(cfg: IngestionConfig) -> Path:
    """Resolve and materialize the cache path used by the ingestion pipeline.

    Args:
        cfg: Runtime ingestion configuration.

    Returns:
        Path: Fully qualified DuckDB cache path guaranteed to exist.
    """
    base_dir = cfg.cache_dir or app_settings.cache_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "docmind.duckdb"


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
    embedding: BaseEmbedding | None = None,
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

    # Title extraction is optional; when the embedding is missing we still allow
    # the pipeline to run quickly without network access.
    try:
        extractor = TitleExtractor(show_progress=False)
        transformations.append(extractor)
    except (ImportError, ValueError, RuntimeError) as exc:  # pragma: no cover
        logger.debug("TitleExtractor unavailable: %s", exc)

    embed_model = embedding or _default_embedding()
    if embed_model is not None:
        transformations.append(embed_model)

    pipeline = IngestionPipeline(
        transformations=transformations,
        cache=ingest_cache,
        docstore=docstore,
    )
    return pipeline, cache_path, docstore_path


def _document_from_input(
    reader: UnstructuredReader | None, item: IngestionInput
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
            docs = reader.load_data(
                unstructured_kwargs={"filename": str(item.source_path)}
            )
        except (OSError, ValueError, RuntimeError) as exc:  # pragma: no cover
            logger.debug("UnstructuredReader failed: %s", exc)
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

    original_flag = app_settings.processing.encrypt_page_images
    desired_flag = encrypt_override or cfg.enable_image_encryption or original_flag
    if desired_flag != original_flag:
        app_settings.processing.encrypt_page_images = desired_flag
    try:
        entries = save_pdf_page_images(path, output_dir)
    finally:
        app_settings.processing.encrypt_page_images = original_flag

    exports: list[ExportArtifact] = []
    for entry in entries:
        image_path = Path(entry["image_path"])
        suffix = image_path.suffix.lower()
        content_type = "application/octet-stream"
        if suffix.endswith(".webp") or suffix.endswith(".webp.enc"):
            content_type = "image/webp"
        elif suffix.endswith(".jpg") or suffix.endswith(".jpeg"):
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
        embedding: Optional embedding instance overriding defaults.

    Returns:
        IngestionResult: Structured ingestion output including nodes, manifest
        summary, exports, metadata, and execution timing.
    """
    pipeline, cache_path, docstore_path = build_ingestion_pipeline(
        cfg, embedding=embedding
    )
    documents, exports = _load_documents(cfg, inputs)

    start = time.perf_counter()
    with _TRACER.start_as_current_span("ingest_documents") as span:
        span.set_attribute("docmind.document_count", len(documents))
        nodes = await pipeline.arun(documents=documents)
    duration_ms = (time.perf_counter() - start) * 1000.0

    if docstore_path is not None:
        pipeline.docstore.persist(str(docstore_path))

    corpus_paths = [Path(item.source_path) for item in inputs if item.source_path]
    manifest = ManifestSummary(
        corpus_hash=compute_corpus_hash(corpus_paths),
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
) -> IngestionResult:
    """Run :func:`ingest_documents` synchronously via ``asyncio.run``.

    Args:
        cfg: Ingestion configuration specifying chunking and persistence.
        inputs: Normalized ingestion inputs to process.
        embedding: Optional embedding instance overriding defaults.

    Returns:
        IngestionResult: Structured ingestion output from the async pipeline.
    """
    return asyncio.run(ingest_documents(cfg, inputs, embedding=embedding))


__all__ = [
    "build_ingestion_pipeline",
    "ingest_documents",
    "ingest_documents_sync",
]
