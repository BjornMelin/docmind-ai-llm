"""Streamlit ingestion adapter backed by the LlamaIndex pipeline."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from llama_index.core import StorageContext, VectorStoreIndex

from src.config import setup_llamaindex
from src.config.integrations import get_settings_embed_model
from src.config.settings import settings
from src.models.processing import (
    CANONICAL_DOCUMENT_ID_KEY,
    IngestionConfig,
    IngestionInput,
)
from src.persistence.hashing import compute_config_hash, compute_corpus_hash_entries
from src.persistence.snapshot_utils import (
    activation_config_dict,
    collect_corpus_paths,
    current_config_dict,
)
from src.processing.ingestion_api import require_unique_document_ids
from src.processing.ingestion_pipeline import (
    embedding_allowed_for_ingestion,
    ingest_documents_sync,
)
from src.retrieval import vector_contract
from src.telemetry.opentelemetry import configure_observability
from src.ui.vector_session import VectorIndexResource
from src.utils.hashing import document_id_from_sha256, sha256_file
from src.utils.storage import (
    close_vector_store_clients,
    create_vector_store,
    get_client_config,
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


def _fsync_dir(path: Path) -> None:
    """Persist a directory entry update where the platform supports it."""
    with contextlib.suppress(OSError):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _activation_corpus_hash(
    *,
    excluded_source_paths: tuple[Path, ...],
    path_aliases: Mapping[Path, Path],
) -> str:
    """Hash durable uploads plus staged bytes under their final logical paths."""
    uploads_root = (settings.data_dir / "uploads").resolve()
    excluded = {Path(path).resolve() for path in excluded_source_paths}
    entries = [
        (path.resolve().relative_to(uploads_root).as_posix(), path)
        for path in collect_corpus_paths(uploads_root)
        if path.resolve() not in excluded
    ]
    for source_path, logical_path in path_aliases.items():
        source = Path(source_path).resolve(strict=True)
        logical = Path(logical_path).resolve(strict=False)
        if source_path.is_symlink() or not source.is_file():
            raise ValueError("Prospective corpus source is not a regular file")
        try:
            relative = logical.relative_to(uploads_root).as_posix()
        except ValueError as exc:
            raise ValueError("Prospective corpus path escapes uploads") from exc
        if logical in excluded:
            continue
        entries.append((relative, source))
    return compute_corpus_hash_entries(entries)


def ingest_inputs(
    inputs: Sequence[IngestionInput],
    *,
    text_collection_name: str,
    image_collection_name: str,
    excluded_source_paths: tuple[Path, ...] = (),
    activation_path_aliases: Mapping[Path, Path] | None = None,
    enable_graphrag: bool = False,
    encrypt_images: bool | None = None,
    nlp_service: SpacyNlpService | None = None,
) -> dict[str, Any]:
    """Ingest pre-saved inputs using the canonical pipeline.

    Args:
        inputs: Normalized ingestion inputs (paths or payload bytes).
        text_collection_name: New physical text collection for this build.
        image_collection_name: New physical image collection for this build.
        excluded_source_paths: Exact upload paths intentionally omitted from rebuild.
        activation_path_aliases: Staged sources mapped to their final upload paths.
        enable_graphrag: When ``True``, attempts to build a PropertyGraphIndex.
        encrypt_images: Optional override for page-image encryption configuration.
        nlp_service: Optional spaCy service used for NLP enrichment.

    Returns:
        Mapping containing ingestion metadata and constructed indices.
    """
    canonical_inputs = _canonicalize_source_inputs(inputs)
    require_unique_document_ids(canonical_inputs)
    setup_llamaindex(force_embed=False)
    embed_model = get_settings_embed_model()
    if embed_model is None:
        raise RuntimeError("Canonical embedding is unavailable for ingestion")
    if not embedding_allowed_for_ingestion(embed_model):
        raise RuntimeError("Canonical embedding is blocked by endpoint policy")

    configure_observability(settings)

    cfg = _build_ingestion_config(
        encrypt_images,
        image_collection_name=image_collection_name,
    )
    _ensure_staged_image_collection(image_collection_name)
    extra_inputs, corpus_document_ids = _additional_corpus_inputs(
        canonical_inputs,
        cfg,
        excluded_source_paths=excluded_source_paths,
    )
    authoritative_inputs = list(canonical_inputs)
    authoritative_inputs.extend(extra_inputs)
    require_unique_document_ids(authoritative_inputs)
    path_aliases = {
        Path(source).resolve(): Path(destination).resolve(strict=False)
        for source, destination in (activation_path_aliases or {}).items()
    }
    activation_corpus_hash = _activation_corpus_hash(
        excluded_source_paths=excluded_source_paths,
        path_aliases=path_aliases,
    )
    activation_config = activation_config_dict(
        settings,
        inputs=authoritative_inputs,
        encrypt_images=cfg.enable_image_encryption,
        graph_requested=enable_graphrag,
    )
    activation_config_hash = compute_config_hash(activation_config)
    snapshot_config_hash = compute_config_hash(current_config_dict(settings))
    expected_document_ids = {str(item.document_id) for item in authoritative_inputs}
    expected_document_ids.update(corpus_document_ids)

    result = (
        ingest_documents_sync(cfg, authoritative_inputs, nlp_service=nlp_service)
        if authoritative_inputs
        else None
    )
    expected_image_points = (
        int(result.metadata.get("image_index.indexed", 0) or 0)
        if result is not None
        else 0
    )
    indexed_documents = list(result.documents) if result is not None else []
    indexed_nodes = list(result.nodes) if result is not None else []

    _verify_image_collection_count(
        image_collection_name,
        expected_count=expected_image_points,
    )

    loaded_document_ids = _loaded_document_ids(indexed_documents)
    if loaded_document_ids != expected_document_ids:
        raise RuntimeError("Corpus rebuild did not load every authoritative upload")
    if (
        _activation_corpus_hash(
            excluded_source_paths=excluded_source_paths,
            path_aliases=path_aliases,
        )
        != activation_corpus_hash
    ):
        raise RuntimeError("Authoritative upload corpus changed during ingestion")
    vector_resource = _build_vector_index(
        indexed_nodes,
        document_ids=loaded_document_ids,
        collection_name=text_collection_name,
    )
    vector_index = vector_resource.index
    try:
        pg_index = (
            _build_property_graph(indexed_documents)
            if enable_graphrag and indexed_documents
            else None
        )

        exports: list[dict[str, Any]] = []
        for artifact in result.exports if result is not None else []:
            dumped = artifact.model_dump()
            # Do not emit raw filesystem paths into Streamlit state.
            dumped.pop("path", None)
            exports.append(dumped)

        metadata = dict(result.metadata or {}) if result is not None else {}
        nlp_enabled = metadata.get("nlp.enabled", False)
        nlp_preview = _build_nlp_preview(indexed_nodes) if nlp_enabled else None

        return {
            "count": len(expected_document_ids),
            "vector_index": vector_index,
            "vector_resource": vector_resource,
            "pg_index": pg_index,
            "manifest": result.manifest.model_dump() if result is not None else None,
            "activation_corpus_hash": activation_corpus_hash,
            "activation_config": activation_config,
            "activation_config_hash": activation_config_hash,
            "snapshot_config_hash": snapshot_config_hash,
            "exports": exports,
            "duration_ms": result.duration_ms if result is not None else 0.0,
            "metadata": metadata,
            "nlp_preview": nlp_preview,
            "documents": indexed_documents,
            "collections": {
                "text": text_collection_name,
                "image": image_collection_name,
            },
        }
    except Exception:
        if vector_resource is not None:
            vector_resource.close()
        raise


def _loaded_document_ids(documents: Sequence[Any]) -> set[str]:
    """Return canonical IDs for documents that completed the loading boundary."""
    document_ids: set[str] = set()
    for document in documents:
        metadata = getattr(document, "metadata", None) or {}
        document_id = metadata.get(CANONICAL_DOCUMENT_ID_KEY)
        if isinstance(document_id, str) and document_id:
            document_ids.add(document_id)
    return document_ids


def _additional_corpus_inputs(
    current_inputs: Sequence[IngestionInput],
    cfg: IngestionConfig,
    *,
    excluded_source_paths: tuple[Path, ...],
) -> tuple[list[IngestionInput], set[str]]:
    """Return authoritative uploads not already present in the current batch."""
    current_ids = {str(item.document_id) for item in current_inputs}
    current_paths = {
        Path(item.source_path).resolve()
        for item in current_inputs
        if item.source_path is not None
    }
    excluded = {Path(path).resolve() for path in excluded_source_paths}
    corpus_ids: set[str] = set()
    additional_ids: set[str] = set()
    additional: list[IngestionInput] = []
    uploads_root = settings.data_dir / "uploads"
    if not uploads_root.exists():
        return additional, corpus_ids

    for path in sorted(uploads_root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in excluded:
            continue
        digest = sha256_file(path)
        document_id = document_id_from_sha256(digest)
        corpus_ids.add(document_id)
        if (
            resolved in current_paths
            or document_id in current_ids
            or document_id in additional_ids
        ):
            continue
        additional.append(
            IngestionInput(
                document_id=document_id,
                source_path=path,
                metadata={"sha256": digest},
                encrypt_images=cfg.enable_image_encryption,
            )
        )
        additional_ids.add(document_id)
    return additional, corpus_ids


def _canonicalize_source_inputs(
    inputs: Sequence[IngestionInput],
) -> list[IngestionInput]:
    """Rehash file-backed inputs at the worker boundary and dedupe by path."""
    canonical: list[IngestionInput] = []
    seen_paths: set[Path] = set()
    for item in inputs:
        if item.source_path is None:
            canonical.append(item)
            continue
        source_path = Path(item.source_path)
        if source_path.is_symlink():
            raise ValueError("Ingestion source must be a regular non-symlink file")
        try:
            source = source_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("Ingestion source is unavailable") from exc
        if not source.is_file():
            raise ValueError("Ingestion source must be a regular non-symlink file")
        if source in seen_paths:
            raise ValueError("Ingestion source path is duplicated")
        seen_paths.add(source)
        digest = sha256_file(source)
        metadata = dict(item.metadata)
        metadata["sha256"] = digest
        canonical.append(
            item.model_copy(
                update={
                    "document_id": document_id_from_sha256(digest),
                    "source_path": source,
                    "metadata": metadata,
                }
            )
        )
    return canonical


def save_uploaded_file(
    file_obj: Any,
    *,
    destination_dir: Path | None = None,
) -> tuple[Path, str]:
    """Persist an upload into the requested local staging directory."""
    uploads_dir = destination_dir or (settings.data_dir / "uploads")
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
            handle.flush()
            os.fsync(handle.fileno())
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
        _fsync_dir(uploads_dir)
        temp_path = None
        return dest, digest_hex
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if hasattr(file_obj, "seek"):
            with contextlib.suppress(Exception):
                file_obj.seek(0)


def _build_ingestion_config(
    encrypt_images: bool | None,
    *,
    image_collection_name: str,
) -> IngestionConfig:
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
        image_collection_name=image_collection_name,
        strict_image_indexing=True,
        cache_dir=cache_dir,
        cache_collection="docmind_ingestion",
        enable_observability=enable_observability,
        observability_sample_rate=observability.sampling_ratio,
        span_exporter_endpoint=observability.endpoint if enable_observability else None,
    )


def _ensure_staged_image_collection(collection_name: str) -> None:
    """Create and validate the physical image collection for this build."""
    from qdrant_client import QdrantClient

    from src.retrieval.image_index import ensure_siglip_image_collection

    client = QdrantClient(**get_client_config())
    try:
        ensure_siglip_image_collection(client, collection_name)
    finally:
        with contextlib.suppress(Exception):
            client.close()


def _verify_image_collection_count(
    collection_name: str,
    *,
    expected_count: int,
) -> None:
    """Require an exact image point count before snapshot promotion."""
    from qdrant_client import QdrantClient

    client = QdrantClient(**get_client_config())
    try:
        result = client.count(collection_name=collection_name, exact=True)
        if int(getattr(result, "count", -1)) != int(expected_count):
            raise RuntimeError(
                "Image collection exact point count does not match build"
            )
    finally:
        with contextlib.suppress(Exception):
            client.close()


def _build_vector_index(
    nodes: list[Any],
    *,
    document_ids: set[str],
    collection_name: str,
) -> VectorIndexResource:
    """Build and verify one immutable physical text collection."""
    if bool(document_ids) != bool(nodes):
        raise ValueError("Document and node emptiness must agree at activation")
    store: Any | None = None
    try:
        _assign_stable_node_ids(nodes)
        vector_store = create_vector_store(
            collection_name,
        )
        store = vector_store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = (
            VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                show_progress=False,
            )
            if nodes
            else VectorStoreIndex.from_vector_store(vector_store)
        )
        _verify_text_collection(
            vector_store,
            nodes=nodes,
            document_ids=document_ids,
            sparse_enabled=vector_contract.sparse_retrieval_enabled(),
        )
        return VectorIndexResource.from_vector_store(index, vector_store)
    except Exception:
        close_vector_store_clients(store)
        raise


def _verify_text_collection(
    store: Any,
    *,
    nodes: list[Any],
    document_ids: set[str],
    sparse_enabled: bool,
) -> None:
    """Require exact point, vector-head, and corpus coverage before promotion."""
    expected_ids = {str(node.node_id) for node in nodes}
    if len(expected_ids) != len(nodes):
        raise RuntimeError("Text collection build produced duplicate node IDs")
    count_result = store.client.count(
        collection_name=store.collection_name,
        exact=True,
    )
    if int(getattr(count_result, "count", -1)) != len(expected_ids):
        raise RuntimeError("Text collection exact point count does not match build")

    retrieved_ids: set[str] = set()
    retrieved_document_ids: set[str] = set()
    ordered_ids = sorted(expected_ids)
    for start in range(0, len(ordered_ids), 256):
        points = store.client.retrieve(
            collection_name=store.collection_name,
            ids=ordered_ids[start : start + 256],
            with_payload=[CANONICAL_DOCUMENT_ID_KEY],
            with_vectors=True,
        )
        for point in points or []:
            retrieved_ids.add(str(getattr(point, "id", "")))
            payload = getattr(point, "payload", None) or {}
            document_id = payload.get(CANONICAL_DOCUMENT_ID_KEY)
            if isinstance(document_id, str):
                retrieved_document_ids.add(document_id)
            vectors = getattr(point, "vector", None)
            if (
                not isinstance(vectors, dict)
                or vector_contract.DENSE_VECTOR_NAME not in vectors
            ):
                raise RuntimeError("Text collection point is missing its dense vector")
            if sparse_enabled and vector_contract.SPARSE_VECTOR_NAME not in vectors:
                raise RuntimeError("Text collection point is missing its sparse vector")
    if retrieved_ids != expected_ids:
        raise RuntimeError("Text collection verification found missing point IDs")
    if retrieved_document_ids != document_ids:
        raise RuntimeError(
            "Text collection verification found incomplete corpus coverage"
        )


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


def _build_property_graph(documents: list[Any]) -> Any | None:
    """Construct the required full-corpus PropertyGraphIndex or fail closed."""
    if PropertyGraphIndex is None:
        raise RuntimeError("PropertyGraphIndex dependency is unavailable")
    if not documents:
        raise ValueError("Cannot build GraphRAG without corpus documents")
    return PropertyGraphIndex.from_documents(documents, show_progress=False)


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


__all__ = ["ingest_inputs", "save_uploaded_file"]
