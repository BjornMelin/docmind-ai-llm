"""Streamlit ingestion adapter backed by the LlamaIndex pipeline."""

from __future__ import annotations

import contextlib
import hashlib
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llama_index.core import StorageContext, VectorStoreIndex

from src.config import setup_llamaindex
from src.config.integrations import get_settings_embed_model
from src.config.settings import settings
from src.models.processing import IngestionConfig, IngestionInput
from src.processing.ingestion_pipeline import ingest_documents_sync
from src.telemetry.opentelemetry import configure_observability
from src.utils.storage import create_vector_store

try:
    from llama_index.core import PropertyGraphIndex
except ImportError:  # pragma: no cover - optional dependency
    PropertyGraphIndex = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from src.nlp.spacy_service import SpacyNlpService
else:
    SpacyNlpService = object  # type: ignore

_LOG = logging.getLogger(__name__)

_NLP_PREVIEW_MAX_ENTITIES = 50
_NLP_PREVIEW_MAX_SENTENCES = 25


def ingest_files(
    files: Sequence[Any],
    *,
    enable_graphrag: bool = False,
    encrypt_images: bool | None = None,
    nlp_service: SpacyNlpService | None = None,
) -> dict[str, Any]:
    """Ingest uploaded files using the canonical pipeline.

    Args:
        files: Uploaded file-like objects (e.g., ``streamlit.UploadedFile``).
        enable_graphrag: When ``True``, attempts to build a PropertyGraphIndex.
        encrypt_images: Optional override for page-image encryption.
        nlp_service: Optional spaCy service used for NLP enrichment.

    Returns:
        Mapping containing ingestion metadata and constructed indices.
    """
    setup_llamaindex(force_embed=False)
    embed_model_before = get_settings_embed_model()

    if not files:
        return {
            "count": 0,
            "vector_index": None,
            "pg_index": None,
            "manifest": None,
            "exports": [],
            "duration_ms": 0.0,
        }

    if embed_model_before is None:
        _LOG.debug(
            "Embedding missing before ingestion; deferring vector index decision"
        )

    configure_observability(settings)

    saved_inputs: list[IngestionInput] = []
    for file_obj in files:
        stored_path, digest = save_uploaded_file(file_obj)
        metadata = {
            "source_filename": getattr(file_obj, "name", stored_path.name),
            "uploaded_at": datetime.now(UTC).isoformat(),
            "sha256": digest,
        }
        saved_inputs.append(
            IngestionInput(
                document_id=f"doc-{digest[:16]}",
                source_path=stored_path,
                metadata=metadata,
                encrypt_images=bool(encrypt_images),
            )
        )

    cfg = _build_ingestion_config(encrypt_images)
    result = ingest_documents_sync(cfg, saved_inputs, nlp_service=nlp_service)

    embed_model_after = get_settings_embed_model()
    if embed_model_after is None:
        if embed_model_before is None:
            _LOG.warning(
                "No embedding configured; vector index creation will be skipped"
            )
    else:
        if embed_model_before is None:
            _LOG.info(
                "Embedding configured during ingestion; vector index will be built"
            )

    vector_index = (
        _build_vector_index(result.nodes) if embed_model_after is not None else None
    )
    pg_index = _build_property_graph(result.documents) if enable_graphrag else None

    exports: list[dict[str, Any]] = []
    for artifact in result.exports:
        dumped = artifact.model_dump()
        # Do not emit raw filesystem paths into Streamlit state.
        dumped.pop("path", None)
        exports.append(dumped)

    nlp_preview = _build_nlp_preview(result.nodes)

    return {
        "count": len(saved_inputs),
        "vector_index": vector_index,
        "pg_index": pg_index,
        "manifest": result.manifest.model_dump(),
        "exports": exports,
        "duration_ms": result.duration_ms,
        "metadata": dict(result.metadata or {}),
        "nlp_preview": nlp_preview,
        "documents": result.documents,
    }


def save_uploaded_file(file_obj: Any) -> tuple[Path, str]:
    """Persist an uploaded file into ``data/uploads`` and return its path and hash."""
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    original_name = getattr(file_obj, "name", "document")
    safe_name = Path(original_name).name or "document"
    data = _read_bytes(file_obj)
    digest = hashlib.sha256(data).hexdigest()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = Path(safe_name).stem or "document"
    suffix = Path(safe_name).suffix
    dest = uploads_dir / f"{stem}-{timestamp}-{digest[:8]}{suffix}"
    counter = 1
    while dest.exists():
        dest = uploads_dir / f"{stem}-{timestamp}-{digest[:8]}-{counter}{suffix}"
        counter += 1
    dest.write_bytes(data)
    return dest, digest


def _build_ingestion_config(encrypt_images: bool | None) -> IngestionConfig:
    """Create an :class:`IngestionConfig` aligned with runtime settings."""
    enable_encryption = (
        encrypt_images
        if encrypt_images is not None
        else settings.processing.encrypt_page_images
    )
    cache_dir = settings.cache_dir / "ingestion"
    cache_dir.mkdir(parents=True, exist_ok=True)
    docstore_path = cache_dir / "docstore.json"

    observability = settings.observability
    enable_observability = bool(observability.enabled and observability.endpoint)

    return IngestionConfig(
        chunk_size=settings.processing.chunk_size,
        chunk_overlap=settings.processing.chunk_overlap,
        enable_image_encryption=enable_encryption,
        cache_dir=cache_dir,
        cache_collection="docmind_ingestion",
        docstore_path=docstore_path,
        enable_observability=enable_observability,
        observability_sample_rate=observability.sampling_ratio,
        span_exporter_endpoint=observability.endpoint if enable_observability else None,
    )


def _build_vector_index(nodes: list[Any]) -> Any | None:
    """Instantiate a vector index over the ingested nodes."""
    if not nodes:
        return None
    try:
        store = create_vector_store(
            settings.database.qdrant_collection,
            enable_hybrid=getattr(settings.retrieval, "enable_server_hybrid", True),
        )
        storage_context = StorageContext.from_defaults(vector_store=store)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=False,
        )
        return index
    except (
        RuntimeError,
        ValueError,
        ImportError,
        ConnectionError,
        TimeoutError,
        OSError,
        FileNotFoundError,
        TypeError,
    ) as exc:  # pragma: no cover - defensive
        _LOG.warning("Vector index creation failed: %s", exc)
        return None


def _build_property_graph(documents: list[Any]) -> Any | None:
    """Construct a PropertyGraphIndex when dependencies permit."""
    if PropertyGraphIndex is None or not documents:
        return None
    try:
        return PropertyGraphIndex.from_documents(documents, show_progress=False)
    except (
        RuntimeError,
        ValueError,
        ImportError,
        AttributeError,
    ) as exc:  # pragma: no cover - defensive
        _LOG.warning("PropertyGraphIndex build failed: %s", exc)
        return None


def _read_bytes(file_obj: Any) -> bytes:
    """Return file contents as bytes without consuming the original stream."""
    if hasattr(file_obj, "getbuffer"):
        buffer = file_obj.getbuffer()
        data = bytes(buffer)
    else:
        data = file_obj.read()
    if hasattr(file_obj, "seek"):
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            file_obj.seek(0)
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    return bytes(data)


def _build_nlp_preview(nodes: list[Any]) -> dict[str, Any]:
    """Build a small, UI-safe NLP preview from enriched node metadata."""
    entities: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []

    for node in nodes:
        meta = getattr(node, "metadata", None) or {}
        payload = meta.get("docmind_nlp")
        if not isinstance(payload, dict):
            continue

        payload_entities = payload.get("entities")
        if (
            isinstance(payload_entities, list)
            and len(entities) < _NLP_PREVIEW_MAX_ENTITIES
        ):
            for ent in payload_entities:
                if not isinstance(ent, dict):
                    continue
                label = ent.get("label")
                text = ent.get("text")
                if not isinstance(label, str) or not isinstance(text, str):
                    continue
                entities.append(
                    {
                        "label": label,
                        "text": text,
                        "start_char": ent.get("start_char"),
                        "end_char": ent.get("end_char"),
                    }
                )
                if len(entities) >= _NLP_PREVIEW_MAX_ENTITIES:
                    break

        payload_sentences = payload.get("sentences")
        if (
            isinstance(payload_sentences, list)
            and len(sentences) < _NLP_PREVIEW_MAX_SENTENCES
        ):
            for sent in payload_sentences:
                if not isinstance(sent, dict):
                    continue
                text = sent.get("text")
                if not isinstance(text, str):
                    continue
                sentences.append(
                    {
                        "text": text,
                        "start_char": sent.get("start_char"),
                        "end_char": sent.get("end_char"),
                    }
                )
                if len(sentences) >= _NLP_PREVIEW_MAX_SENTENCES:
                    break

        if (
            len(entities) >= _NLP_PREVIEW_MAX_ENTITIES
            and len(sentences) >= _NLP_PREVIEW_MAX_SENTENCES
        ):
            break

    return {
        "enabled": bool(entities or sentences),
        "entities": entities,
        "sentences": sentences,
    }


__all__ = ["ingest_files", "save_uploaded_file"]
