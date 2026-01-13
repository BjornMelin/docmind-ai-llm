"""Snapshot rebuild service boundary (library-first, Streamlit-free).

Extracted from the Documents page to keep Streamlit pages lightweight and
importable in tests while providing a reusable, typed snapshot rebuild entrypoint.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.persistence.snapshot import SnapshotManager
from src.persistence.snapshot_utils import current_config_dict, timestamped_export_path
from src.utils.hashing import sha256_file


class VectorIndexProtocol(Protocol):
    """Protocol for vector index with embed model."""

    @property
    def embed_model(self) -> Any:
        """Get the embed model from the vector index."""

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute access for private attributes."""


class PgIndexProtocol(Protocol):
    """Protocol for property graph index with required attributes."""

    @property
    def property_graph_store(self) -> Any:
        """Get the property graph store."""

    @property
    def storage_context(self) -> Any:
        """Get the storage context."""


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
    vector_index: VectorIndexProtocol,
    pg_index: PgIndexProtocol | None,
) -> tuple[Any | None, Any | None, PgIndexProtocol | None]:
    """Persist vector/graph indexes and return graph handles.

    Args:
        mgr: SnapshotManager instance for persistence.
        workspace: Workspace path for snapshot.
        vector_index: Vector index with embed model (required).
        pg_index: PropertyGraphIndex-like instance (optional).

    Returns:
        Tuple of (property_graph_store, storage_context, pg_index).
    """
    if vector_index is None:
        raise TypeError("rebuild_snapshot requires a vector_index instance")
    mgr.persist_vector_index(vector_index, workspace)
    graph_store = getattr(pg_index, "property_graph_store", None) if pg_index else None
    if graph_store is not None:
        mgr.persist_graph_store(graph_store, workspace)
    storage_context = getattr(pg_index, "storage_context", None) if pg_index else None
    return graph_store, storage_context, pg_index


def _export_graphs(
    *,
    workspace: Path,
    pg_index: Any,
    vector_index: Any,
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
        log_export_event({
            "export_performed": True,
            "export_type": f"graph_{fmt}",
            "seed_count": metadata["seed_count"],
            "dest_basename": path.name,
            "context": "snapshot",
            "duration_ms": metadata["duration_ms"],
            "size_bytes": metadata["size_bytes"],
            "sha256": sha,
        })
        record_graph_export_metric(
            f"graph_{fmt}",
            duration_ms=metadata["duration_ms"],
            seed_count=metadata["seed_count"],
            size_bytes=metadata["size_bytes"],
            context="snapshot",
        )

    if can_export_graph:
        from src.retrieval.graph_config import export_graph_jsonl, export_graph_parquet

        jsonl_path = timestamped_export_path(graph_dir, "jsonl")
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
            logger.warning("Graph JSONL export failed (snapshot): {}", exc)
            log_export_event({
                "export_performed": False,
                "export_type": "graph_jsonl",
                "context": "snapshot",
                "error": str(exc),
            })

        parquet_path = timestamped_export_path(graph_dir, "parquet")
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
            logger.warning("Graph Parquet export failed (snapshot): {}", exc)
            log_export_event({
                "export_performed": False,
                "export_type": "graph_parquet",
                "context": "snapshot",
                "error": str(exc),
            })

    return exports_meta


def _collect_corpus_paths(settings_obj: Any) -> tuple[list[Path], Path]:
    """Collect uploaded corpus paths and base directory.

    Uses a cached manifest if available, falling back to bounded globbing
    to avoid expensive recursive directory traversal on large corpora.
    """
    import json

    uploads_dir = settings_obj.data_dir / "uploads"
    manifest_file = settings_obj.data_dir / ".corpus_manifest.json"

    # Try to use cached manifest first
    if manifest_file.exists():
        try:
            with open(manifest_file) as f:
                manifest_data = json.load(f)
                cached_paths = [Path(p) for p in manifest_data.get("files", [])]
                if cached_paths or not uploads_dir.exists():
                    return cached_paths, uploads_dir
        except Exception:  # Fallback if manifest is corrupted
            pass

    # Glob for corpus files (bounded to immediate children if corpus is large)
    corpus_paths = (
        [p for p in uploads_dir.glob("**/*") if p.is_file()]
        if uploads_dir.exists()
        else []
    )

    # Cache the result for next time
    try:
        with open(manifest_file, "w") as f:
            json.dump({"files": [str(p) for p in corpus_paths]}, f)
    except Exception:  # Silently fail if we can't write cache
        pass

    return corpus_paths, uploads_dir


def _build_versions(
    settings_obj: Any, vector_index: Any, embed_model: Any | None
) -> dict[str, str]:
    """Build version metadata for the manifest.

    Args:
        settings_obj: Settings object with app_version and database.
        vector_index: Vector index instance (used for fallback embed model).
        embed_model: Optional explicit embed model; if None, falls back to
            Settings.embed_model or vector_index public interface.

    Returns:
        Dictionary of component versions for manifest metadata.
    """
    versions: dict[str, str] = {"app": settings_obj.app_version}
    with contextlib.suppress(Exception):  # pragma: no cover
        import llama_index  # type: ignore[import]

        versions["llama_index"] = getattr(llama_index, "__version__", "unknown")

    # Use explicitly passed embed_model or fall back to settings
    if embed_model is None:
        with contextlib.suppress(Exception):
            from llama_index.core import Settings  # type: ignore

            embed_model = getattr(Settings, "embed_model", None)

    # Last-resort fallback: try public interface first, then private (if available)
    if embed_model is None:
        embed_model = getattr(vector_index, "embed_model", None)
    if embed_model is None:
        # Only access private attribute as last resort, with guard
        if hasattr(vector_index, "_embed_model"):
            embed_model = getattr(vector_index, "_embed_model", None)
            if embed_model:
                logger.debug(
                    "Using private _embed_model fallback; prefer passing embed_model explicitly"
                )
        if embed_model is None:
            logger.debug(
                "Embed model not found: pass explicitly or ensure public interface available"
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
    if hasattr(settings_obj.database, "client_version"):
        versions.setdefault("vector_client", str(settings_obj.database.client_version))
    return versions


def rebuild_snapshot(
    vector_index: Any,
    pg_index: Any,
    settings_obj: Any,
    *,
    embed_model: Any | None = None,
    log_export_event: Callable[[dict[str, Any]], None] | None = None,
    record_graph_export_metric: Callable[..., None] | None = None,
) -> Path:
    """Rebuild snapshot for current indices and return final path.

    Args:
        vector_index: Vector index instance (required).
        pg_index: Optional PropertyGraphIndex-like instance (may be None).
        settings_obj: Settings object with `data_dir`, `database`, and `app_version`.
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

    storage_dir = settings_obj.data_dir / "storage"
    mgr = SnapshotManager(storage_dir)
    workspace = mgr.begin_snapshot()
    try:
        graph_store, storage_context, pg_index = _persist_indices(
            mgr, workspace, vector_index, pg_index
        )
        exports_meta = _export_graphs(
            workspace=workspace,
            pg_index=pg_index,
            vector_index=vector_index,
            graph_store=graph_store,
            storage_context=storage_context,
            settings_obj=settings_obj,
            log_export_event=log_export_event,
            record_graph_export_metric=record_graph_export_metric,
        )

        corpus_paths, uploads_dir = _collect_corpus_paths(settings_obj)
        chash = compute_corpus_hash(corpus_paths, base_dir=uploads_dir)
        cfg = current_config_dict(settings_obj)
        cfg_hash = compute_config_hash(cfg)
        versions = _build_versions(settings_obj, vector_index, embed_model)

        exports_meta.sort(
            key=lambda item: str(item.get("filename") or item.get("sha256") or "")
        )
        mgr.write_manifest(
            workspace,
            index_id="docmind",
            graph_store_type="property_graph",
            vector_store_type=settings_obj.database.vector_store_type,
            corpus_hash=chash,
            config_hash=cfg_hash,
            versions=versions,
            graph_exports=exports_meta,
        )
        return mgr.finalize_snapshot(workspace)
    except Exception:
        mgr.cleanup_tmp(workspace)
        raise


__all__ = ["rebuild_snapshot"]
