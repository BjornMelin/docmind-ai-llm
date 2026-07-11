"""Streamlit ingestion adapter backed by the LlamaIndex pipeline."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

try:
    from grpc import RpcError
except ImportError:  # pragma: no cover - qdrant-client normally installs grpc
    RpcError = RuntimeError  # type: ignore[assignment]
from llama_index.core import StorageContext, VectorStoreIndex
from pydantic import JsonValue
from qdrant_client import models as qmodels

from src.config import setup_llamaindex
from src.config.integrations import get_settings_embed_model
from src.config.settings import settings
from src.models.processing import (
    CANONICAL_DOCUMENT_ID_KEY,
    IngestionConfig,
    IngestionInput,
)
from src.processing.ingestion_api import require_unique_document_ids
from src.processing.ingestion_pipeline import (
    embedding_allowed_for_ingestion,
    ingest_documents_sync,
)
from src.telemetry.opentelemetry import configure_observability
from src.utils.hashing import document_id_from_sha256
from src.utils.storage import (
    QdrantCollectionIncompatibleError,
    create_vector_store,
)

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
    if not files:
        return {
            "count": 0,
            "vector_index": None,
            "pg_index": None,
            "manifest": None,
            "exports": [],
            "duration_ms": 0.0,
            "metadata": {},
            "nlp_preview": None,
            "documents": [],
        }

    saved_inputs: list[IngestionInput] = []
    default_encrypt = (
        encrypt_images
        if encrypt_images is not None
        else settings.processing.encrypt_page_images
    )
    for file_obj in files:
        stored_path, digest = save_uploaded_file(file_obj)
        metadata: dict[str, JsonValue] = {
            "uploaded_at": datetime.now(UTC).isoformat(),
            "sha256": digest,
        }
        saved_inputs.append(
            IngestionInput(
                document_id=document_id_from_sha256(digest),
                source_path=stored_path,
                metadata=metadata,
                encrypt_images=default_encrypt,
            )
        )

    require_unique_document_ids(saved_inputs)
    return ingest_inputs(
        saved_inputs,
        enable_graphrag=enable_graphrag,
        nlp_service=nlp_service,
    )


def ingest_inputs(
    inputs: Sequence[IngestionInput],
    *,
    enable_graphrag: bool = False,
    encrypt_images: bool | None = None,
    nlp_service: SpacyNlpService | None = None,
) -> dict[str, Any]:
    """Ingest pre-saved inputs using the canonical pipeline.

    Args:
        inputs: Normalized ingestion inputs (paths or payload bytes).
        enable_graphrag: When ``True``, attempts to build a PropertyGraphIndex.
        encrypt_images: Optional override for page-image encryption configuration.
        nlp_service: Optional spaCy service used for NLP enrichment.

    Returns:
        Mapping containing ingestion metadata and constructed indices.
    """
    if not inputs:
        return {
            "count": 0,
            "vector_index": None,
            "pg_index": None,
            "manifest": None,
            "exports": [],
            "duration_ms": 0.0,
            "metadata": {},
            "nlp_preview": None,
            "documents": [],
        }

    require_unique_document_ids(inputs)
    setup_llamaindex(force_embed=False)
    embed_model_before = get_settings_embed_model()

    if embed_model_before is None:
        _LOG.debug(
            "Embedding missing before ingestion; deferring vector index decision"
        )

    configure_observability(settings)

    cfg = _build_ingestion_config(encrypt_images)
    result = ingest_documents_sync(cfg, inputs, nlp_service=nlp_service)

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

    embedding_usable = embedding_allowed_for_ingestion(embed_model_after)
    if embed_model_after is not None and not embedding_usable:
        _LOG.warning("Configured embedding is blocked by endpoint policy")
    loaded_document_ids = _loaded_document_ids(result.documents)
    vector_index = (
        _build_vector_index(
            result.nodes,
            document_ids=loaded_document_ids,
        )
        if embedding_usable
        else None
    )
    pg_index = _build_property_graph(result.documents) if enable_graphrag else None

    exports: list[dict[str, Any]] = []
    for artifact in result.exports:
        dumped = artifact.model_dump()
        # Do not emit raw filesystem paths into Streamlit state.
        dumped.pop("path", None)
        exports.append(dumped)

    nlp_enabled = result.metadata.get("nlp.enabled", False)
    nlp_preview = _build_nlp_preview(result.nodes) if nlp_enabled else None

    return {
        "count": len(inputs),
        "vector_index": vector_index,
        "pg_index": pg_index,
        "manifest": result.manifest.model_dump(),
        "exports": exports,
        "duration_ms": result.duration_ms,
        "metadata": dict(result.metadata or {}),
        "nlp_preview": nlp_preview,
        "documents": result.documents,
    }


def _loaded_document_ids(documents: Sequence[Any]) -> set[str]:
    """Return canonical IDs for documents that completed the loading boundary."""
    document_ids: set[str] = set()
    for document in documents:
        metadata = getattr(document, "metadata", None) or {}
        document_id = metadata.get(CANONICAL_DOCUMENT_ID_KEY)
        if isinstance(document_id, str) and document_id:
            document_ids.add(document_id)
    return document_ids


def save_uploaded_file(file_obj: Any) -> tuple[Path, str]:
    """Persist an uploaded file into ``data/uploads`` and return its path and hash."""
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    original_name = getattr(file_obj, "name", "document")
    safe_name = Path(original_name).name or "document"
    max_bytes = int(settings.processing.max_document_size_mb) * 1024 * 1024
    digest = hashlib.sha256()
    size = 0
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".upload-",
            dir=uploads_dir,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for chunk in _iter_upload_chunks(file_obj):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError("Upload exceeds configured max_document_size_mb")
                digest.update(chunk)
                handle.write(chunk)
        digest_hex = digest.hexdigest()

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        stem = Path(safe_name).stem or "document"
        suffix = Path(safe_name).suffix
        dest = uploads_dir / f"{stem}-{timestamp}-{digest_hex[:8]}{suffix}"
        counter = 1
        while dest.exists():
            dest = uploads_dir / (
                f"{stem}-{timestamp}-{digest_hex[:8]}-{counter}{suffix}"
            )
            counter += 1
        os.replace(temp_path, dest)
        temp_path = None
        return dest, digest_hex
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if hasattr(file_obj, "seek"):
            with contextlib.suppress(Exception):
                file_obj.seek(0)


def _build_ingestion_config(encrypt_images: bool | None) -> IngestionConfig:
    """Create an :class:`IngestionConfig` aligned with runtime settings."""
    enable_encryption = (
        encrypt_images
        if encrypt_images is not None
        else settings.processing.encrypt_page_images
    )
    cache_dir = settings.cache.ingestion_db_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    observability = settings.observability
    enable_observability = bool(observability.enabled and observability.endpoint)

    return IngestionConfig(
        chunk_size=settings.processing.chunk_size,
        chunk_overlap=settings.processing.chunk_overlap,
        enable_image_encryption=enable_encryption,
        cache_dir=cache_dir,
        cache_collection="docmind_ingestion",
        enable_observability=enable_observability,
        observability_sample_rate=observability.sampling_ratio,
        span_exporter_endpoint=observability.endpoint if enable_observability else None,
    )


def _build_vector_index(
    nodes: list[Any],
    *,
    document_ids: set[str],
) -> Any | None:
    """Upsert ingested nodes and remove stale points for the same documents."""
    if not document_ids:
        return None
    if not nodes:
        raise ValueError("Cannot replace document vectors with an empty node set")
    store: Any | None = None
    try:
        _assign_stable_node_ids(nodes)
        vector_store = create_vector_store(
            settings.database.qdrant_collection,
            enable_hybrid=getattr(settings.retrieval, "enable_server_hybrid", True),
        )
        store = vector_store
        previous_ids = _existing_text_point_ids(vector_store, document_ids)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=False,
        )
    except QdrantCollectionIncompatibleError:
        raise
    except (
        RuntimeError,
        ValueError,
        ImportError,
        ConnectionError,
        TimeoutError,
        OSError,
        FileNotFoundError,
        TypeError,
        RpcError,
    ) as exc:  # pragma: no cover - defensive
        _LOG.warning("Vector index creation failed: %s", exc)
        if store is not None:
            with contextlib.suppress(Exception):
                store.client.close()
        return None

    current_ids = {node.node_id for node in nodes}
    stale_ids = previous_ids.difference(current_ids)
    if stale_ids:
        try:
            vector_store.delete_nodes(node_ids=sorted(stale_ids, key=str))
        except Exception as exc:
            with contextlib.suppress(Exception):
                vector_store.client.close()
            raise RuntimeError(
                "Text vector upsert succeeded but stale-point deletion failed"
            ) from exc
    return index


def _assign_stable_node_ids(nodes: list[Any]) -> None:
    """Assign deterministic Qdrant-compatible IDs within each source page."""
    positions: defaultdict[tuple[str, str], int] = defaultdict(int)
    for node in nodes:
        metadata = getattr(node, "metadata", None) or {}
        document_id = metadata.get(CANONICAL_DOCUMENT_ID_KEY)
        if not isinstance(document_id, str) or not document_id:
            continue
        page_id = str(metadata.get("page_id") or "")
        position_key = (document_id, page_id)
        position = positions[position_key]
        positions[position_key] += 1
        node.id_ = str(
            uuid5(
                NAMESPACE_URL,
                f"docmind:{document_id}:{page_id}:{position}",
            )
        )


def _existing_text_point_ids(store: Any, document_ids: set[str]) -> set[str]:
    """Return existing Qdrant point IDs owned by the supplied documents."""
    point_ids: set[str] = set()
    for document_id in sorted(document_ids):
        offset: Any | None = None
        while True:
            points, next_offset = store.client.scroll(
                collection_name=store.collection_name,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key=CANONICAL_DOCUMENT_ID_KEY,
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.update(str(point.id) for point in points)
            if next_offset is None:
                break
            offset = next_offset
    return point_ids


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


def _iter_upload_chunks(file_obj: Any) -> Any:
    """Yield upload bytes without materializing a second full copy."""
    chunk_size = 1024 * 1024
    if hasattr(file_obj, "getbuffer"):
        buffer = file_obj.getbuffer()
        for offset in range(0, len(buffer), chunk_size):
            yield buffer[offset : offset + chunk_size]
        return
    while True:
        chunk = file_obj.read(chunk_size)
        if not chunk:
            return
        if not isinstance(chunk, bytes | bytearray | memoryview):
            raise TypeError("Uploaded file stream must return bytes")
        yield chunk


def _build_nlp_preview(nodes: list[Any]) -> dict[str, Any] | None:
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

    if not entities and not sentences:
        return None

    return {
        "enabled": True,
        "entities": entities,
        "sentences": sentences,
    }


__all__ = ["ingest_files", "save_uploaded_file"]
