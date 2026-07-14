"""Snapshot payload builder for a caller-owned transaction.

Extracted from the Documents page to keep Streamlit pages lightweight and
importable in tests. The caller owns lock, workspace, and failure cleanup; this
module writes the payload and commits it through ``SnapshotManager``.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.persistence.snapshot import SnapshotManager, SnapshotPersistenceError
from src.persistence.snapshot_utils import (
    collect_corpus_paths,
    timestamped_export_path,
)
from src.utils.hashing import sha256_file
from src.utils.log_safety import build_pii_log_entry
from src.version import get_version


class VectorIndexProtocol(Protocol):
    """Protocol for vector index with embed model."""

    @property
    def embed_model(self) -> Any:
        """Get the embed model from the vector index."""


class PgIndexProtocol(Protocol):
    """Protocol for property graph index with required attributes."""

    @property
    def property_graph_store(self) -> Any:
        """Get the property graph store."""

    @property
    def storage_context(self) -> Any:
        """Get the storage context."""


@dataclass(frozen=True, slots=True)
class SnapshotActivation:
    """Caller-owned transaction inputs required for one atomic activation."""

    manager: SnapshotManager
    workspace: Path
    text_collection: str
    image_collection: str
    expected_corpus_hash: str
    expected_config_hash: str
    activation_config: dict[str, Any]
    activation_config_hash: str
    collection_metadata: dict[str, Any]
    graph_requested: bool


def _noop_log_export_event(_payload: dict[str, Any]) -> None:
    """No-op callback for export events."""


def _noop_record_graph_export_metric(*_args: Any, **_kwargs: Any) -> None:
    """No-op callback for graph export metrics."""


def _init_callbacks(
    log_export_event: Callable[[dict[str, Any]], None] | None,
    record_graph_export_metric: Callable[..., None] | None,
) -> tuple[Callable[[dict[str, Any]], None], Callable[..., None]]:
    """Initialize optional callbacks with no-op defaults."""
    if log_export_event is None:
        log_export_event = _noop_log_export_event
    if record_graph_export_metric is None:
        record_graph_export_metric = _noop_record_graph_export_metric
    return log_export_event, record_graph_export_metric


def _persist_indices(
    mgr: SnapshotManager,
    workspace: Path,
    vector_index: VectorIndexProtocol | None,
    pg_index: PgIndexProtocol | None,
) -> tuple[Any | None, Any | None, PgIndexProtocol | None]:
    """Persist graph artifacts and return graph handles.

    Args:
        mgr: SnapshotManager instance for persistence.
        workspace: Workspace path for snapshot.
        vector_index: Live Qdrant-backed index used for graph export seeds.
        pg_index: PropertyGraphIndex-like instance (optional).

    Returns:
        Tuple of (property_graph_store, storage_context, pg_index).
    """
    if vector_index is None:
        raise TypeError("vector_index is required")
    storage_context = getattr(pg_index, "storage_context", None) if pg_index else None
    graph_store = getattr(pg_index, "property_graph_store", None) if pg_index else None
    if graph_store is not None:
        if storage_context is None:
            raise SnapshotPersistenceError(
                "Property graph index has no persistable storage context"
            )
        mgr.persist_graph_storage_context(storage_context, workspace)
    return graph_store, storage_context, pg_index


def _export_graphs(
    *,
    workspace: Path,
    pg_index: PgIndexProtocol | None,
    vector_index: VectorIndexProtocol,
    graph_store: Any,
    storage_context: Any,
    settings_obj: Any,
    log_export_event: Callable[[dict[str, Any]], None],
    record_graph_export_metric: Callable[..., None],
) -> list[dict[str, Any]]:
    """Export graph artifacts and return manifest metadata entries."""
    can_export_graph = (
        pg_index is not None and graph_store is not None and storage_context is not None
    )
    graphrag_cfg = getattr(settings_obj, "graphrag_cfg", None)
    export_cap = int(
        getattr(graphrag_cfg, "export_seed_cap", 32) if graphrag_cfg else 32
    )
    seeds: list[str] = []
    if can_export_graph:
        from src.retrieval.graph_config import get_export_seed_ids

        seeds = get_export_seed_ids(pg_index, vector_index, cap=export_cap)

    graph_dir = workspace / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    exports_meta: list[dict[str, Any]] = []

    def record_export(path: Path, fmt: str, duration_ms: float) -> None:
        if not path.exists():
            return
        sha = sha256_file(path)
        metadata = {
            "filename": path.name,
            "format": fmt,
            "seed_count": len(seeds),
            "size_bytes": path.stat().st_size,
            "created_at": datetime.now(UTC).isoformat(),
            "duration_ms": round(duration_ms, 3),
            "sha256": sha,
        }
        exports_meta.append(metadata)
        log_export_event(
            {
                "export_performed": True,
                "export_type": f"graph_{fmt}",
                "seed_count": metadata["seed_count"],
                "dest_basename": path.name,
                "context": "snapshot",
                "duration_ms": metadata["duration_ms"],
                "size_bytes": metadata["size_bytes"],
                "sha256": sha,
            }
        )
        record_graph_export_metric(
            f"graph_{fmt}",
            duration_ms=metadata["duration_ms"],
            seed_count=metadata["seed_count"],
            size_bytes=metadata["size_bytes"],
            context="snapshot",
        )

    if can_export_graph:
        from src.retrieval.graph_config import export_graph_jsonl, export_graph_parquet

        jsonl_path = timestamped_export_path(
            graph_dir, "jsonl", prefix="graph_export-snapshot"
        )
        try:
            start_json = time.perf_counter()
            export_graph_jsonl(
                property_graph_index=pg_index,
                output_path=jsonl_path,
                seed_node_ids=seeds,
            )
            record_export(
                jsonl_path,
                "jsonl",
                duration_ms=(time.perf_counter() - start_json) * 1000.0,
            )
        except Exception as exc:
            redaction = build_pii_log_entry(str(exc), key_id="snapshot.graph_jsonl")
            logger.warning(
                "Graph JSONL export failed (snapshot) (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            log_export_event(
                {
                    "export_performed": False,
                    "export_type": "graph_jsonl",
                    "context": "snapshot",
                    "error_type": type(exc).__name__,
                    "error": redaction.redacted,
                }
            )

        parquet_path = timestamped_export_path(
            graph_dir, "parquet", prefix="graph_export-snapshot"
        )
        try:
            start_parquet = time.perf_counter()
            export_graph_parquet(
                property_graph_index=pg_index,
                output_path=parquet_path,
                seed_node_ids=seeds,
            )
            record_export(
                parquet_path,
                "parquet",
                duration_ms=(time.perf_counter() - start_parquet) * 1000.0,
            )
        except Exception as exc:
            redaction = build_pii_log_entry(str(exc), key_id="snapshot.graph_parquet")
            logger.warning(
                "Graph Parquet export failed (snapshot) (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            log_export_event(
                {
                    "export_performed": False,
                    "export_type": "graph_parquet",
                    "context": "snapshot",
                    "error_type": type(exc).__name__,
                    "error": redaction.redacted,
                }
            )

    return exports_meta


def _collect_corpus_paths(settings_obj: Any) -> tuple[list[Path], Path]:
    """Collect the complete authoritative upload corpus without a second cache."""
    uploads_dir = settings_obj.data_dir / "uploads"
    return collect_corpus_paths(uploads_dir), uploads_dir


def _build_versions(
    settings_obj: Any, vector_index: VectorIndexProtocol, embed_model: Any | None
) -> dict[str, str]:
    """Build version metadata for the manifest.

    Args:
        settings_obj: Settings object with app_version and database.
        vector_index: Vector index instance (used for fallback embed model).
        embed_model: Optional explicit embed model. If None, falls back in order:
            1. llama_index.core.Settings.embed_model
            2. vector_index.embed_model (public interface)
            3. vector_index._embed_model (private, last resort only)

    Returns:
        Dictionary of component versions for manifest metadata.
    """
    versions: dict[str, str] = {"app": get_version()}
    with contextlib.suppress(metadata.PackageNotFoundError):  # pragma: no cover
        versions["llama_index"] = metadata.version("llama-index-core")

    # Resolve embed_model following documented fallback order.
    if embed_model is None:
        with contextlib.suppress(Exception):
            from llama_index.core import Settings  # type: ignore

            embed_model = getattr(Settings, "embed_model", None)

    if embed_model is None:
        embed_model = getattr(vector_index, "embed_model", None)

    if embed_model is None and hasattr(vector_index, "_embed_model"):
        embed_model = getattr(vector_index, "_embed_model", None)
        if embed_model:
            logger.debug(
                "Using private _embed_model fallback; prefer passing "
                "embed_model explicitly"
            )

    if embed_model is None:
        logger.debug(
            "Embed model not found: pass explicitly or ensure public "
            "interface available"
        )

    # Extract model name safely
    embed_model_name = (
        getattr(embed_model, "model_name", "unknown")
        if embed_model is not None
        else "unknown"
    )
    versions.setdefault("embed_model", embed_model_name)

    # Add client versions if available
    with contextlib.suppress(Exception):  # pragma: no cover
        from qdrant_client import __version__ as qdrant_version  # type: ignore[import]

        versions.setdefault("qdrant_client", qdrant_version)
    if settings_obj.database is not None and hasattr(
        settings_obj.database, "client_version"
    ):
        versions.setdefault("vector_client", str(settings_obj.database.client_version))
    return versions


def rebuild_snapshot(
    vector_index: VectorIndexProtocol,
    pg_index: PgIndexProtocol | None,
    settings_obj: Any,
    activation: SnapshotActivation,
    *,
    commit_source_changes: Callable[[], None] | None = None,
    embed_model: Any | None = None,
    log_export_event: Callable[[dict[str, Any]], None] | None = None,
    record_graph_export_metric: Callable[..., None] | None = None,
) -> Path:
    """Rebuild snapshot for current indices and return final path.

    Args:
        vector_index: Vector index instance (required).
        pg_index: Optional PropertyGraphIndex-like instance (may be None).
        settings_obj: Settings object with `data_dir`, `database`, and `app_version`.
        activation: Caller-owned manager, workspace, physical collection identities,
            corpus identity, collection metadata, and GraphRAG requirement.
        commit_source_changes: Late source promotion/quarantine callback invoked after
            expensive payload preparation and immediately before verification.
        embed_model: Optional embed model instance for version reporting.
            Pass explicitly for reliable version tracking; otherwise we fall back
            to Settings.embed_model and (last-resort) vector_index._embed_model.
        log_export_event: Optional callback for local JSONL telemetry emission.
        record_graph_export_metric: Optional callback for OpenTelemetry metrics.

    Returns:
        Path: Finalized snapshot directory path.
    """
    log_export_event, record_graph_export_metric = _init_callbacks(
        log_export_event, record_graph_export_metric
    )

    if settings_obj.database is None:
        raise ValueError(
            "settings_obj.database must be configured for snapshot rebuild"
        )
    if activation.graph_requested and pg_index is None:
        raise SnapshotPersistenceError(
            "GraphRAG was requested but no property graph was produced"
        )
    graph_store, storage_context, pg_index = _persist_indices(
        activation.manager, activation.workspace, vector_index, pg_index
    )
    if activation.graph_requested:
        if graph_store is None:
            raise SnapshotPersistenceError(
                "GraphRAG was requested but no persistable graph store was produced"
            )
        graph_dir = activation.workspace / "graph"
        if not graph_dir.is_dir() or not any(
            path.is_file() and not path.is_symlink() for path in graph_dir.rglob("*")
        ):
            raise SnapshotPersistenceError(
                "GraphRAG was requested but graph persistence produced no payload"
            )
    exports_meta = _export_graphs(
        workspace=activation.workspace,
        pg_index=pg_index,
        vector_index=vector_index,
        graph_store=graph_store,
        storage_context=storage_context,
        settings_obj=settings_obj,
        log_export_event=log_export_event,
        record_graph_export_metric=record_graph_export_metric,
    )

    if (
        compute_config_hash(activation.activation_config)
        != activation.activation_config_hash
    ):
        raise SnapshotPersistenceError("Activation configuration identity is invalid")
    versions = _build_versions(settings_obj, vector_index, embed_model)

    exports_meta.sort(
        key=lambda item: str(item.get("filename") or item.get("sha256") or "")
    )
    activation.manager.write_manifest(
        activation.workspace,
        index_id="docmind",
        graph_store_type="property_graph" if pg_index is not None else "none",
        vector_store_type=settings_obj.database.vector_store_type,
        text_collection=activation.text_collection,
        image_collection=activation.image_collection,
        corpus_hash=activation.expected_corpus_hash,
        config_hash=activation.expected_config_hash,
        versions=versions,
        graph_exports=exports_meta,
        collection_metadata=activation.collection_metadata,
        activation_config=activation.activation_config,
        activation_config_hash=activation.activation_config_hash,
    )
    if commit_source_changes is not None:
        commit_source_changes()
    corpus_paths, uploads_dir = _collect_corpus_paths(settings_obj)
    chash = compute_corpus_hash(corpus_paths, base_dir=uploads_dir)
    if chash != activation.expected_corpus_hash:
        raise SnapshotPersistenceError(
            "Authoritative corpus changed at the activation boundary"
        )
    return activation.manager.finalize_snapshot(activation.workspace)


__all__ = ["SnapshotActivation", "rebuild_snapshot"]
