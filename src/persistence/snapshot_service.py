"""Snapshot rebuild service boundary (library-first, Streamlit-free).

Extracted from the Documents page to keep Streamlit pages lightweight and
importable in tests while providing a reusable, typed snapshot rebuild entrypoint.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.persistence.snapshot import SnapshotManager
from src.persistence.snapshot_utils import current_config_dict


def rebuild_snapshot(
    vector_index: Any,
    pg_index: Any,
    settings_obj: Any,
    *,
    log_export_event: Callable[[dict[str, Any]], None] | None = None,
    record_graph_export_metric: Callable[..., None] | None = None,
) -> Path:
    """Rebuild snapshot for current indices and return final path.

    Args:
        vector_index: Vector index instance (required).
        pg_index: Optional PropertyGraphIndex-like instance (may be None).
        settings_obj: Settings object with `data_dir`, `database`, and `app_version`.
        log_export_event: Optional callback for local JSONL telemetry emission.
        record_graph_export_metric: Optional callback for OpenTelemetry metrics.

    Returns:
        Path: Finalized snapshot directory path.
    """
    if log_export_event is None:
        log_export_event = lambda _payload: None  # noqa: E731
    if record_graph_export_metric is None:
        record_graph_export_metric = lambda *_args, **_kwargs: None  # noqa: E731

    storage_dir = settings_obj.data_dir / "storage"
    mgr = SnapshotManager(storage_dir)
    workspace = mgr.begin_snapshot()
    try:
        if vector_index is None:
            raise TypeError("rebuild_snapshot requires a vector_index instance")
        mgr.persist_vector_index(vector_index, workspace)
        graph_store = getattr(pg_index, "property_graph_store", None)
        if graph_store is not None:
            mgr.persist_graph_store(graph_store, workspace)

        storage_context = getattr(pg_index, "storage_context", None)
        can_export_graph = (
            pg_index is not None
            and graph_store is not None
            and storage_context is not None
        )
        export_cap = int(
            getattr(
                getattr(settings_obj, "graphrag_cfg", object()), "export_seed_cap", 32
            )
        )
        seeds: list[str] = []
        if can_export_graph:
            from src.retrieval.graph_config import get_export_seed_ids

            seeds = get_export_seed_ids(pg_index, vector_index, cap=export_cap)

        graph_dir = workspace / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        exports_meta: list[dict[str, Any]] = []

        def _file_sha256(path: Path) -> str:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

        def _timestamped_export_path(out_dir: Path, extension: str) -> Path:
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            candidate = out_dir / f"graph_export-{ts}.{extension}"
            counter = 1
            while candidate.exists():
                candidate = out_dir / f"graph_export-{ts}-{counter}.{extension}"
                counter += 1
            return candidate

        def _record_export(path: Path, fmt: str, duration_ms: float) -> None:
            if not path.exists():
                return
            sha = _file_sha256(path)
            metadata = {
                # Final-release: avoid persisting filesystem paths in durable
                # manifest metadata.
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
            from src.retrieval.graph_config import (
                export_graph_jsonl,
                export_graph_parquet,
            )

            jsonl_path = _timestamped_export_path(graph_dir, "jsonl")
            try:
                start_json = time.perf_counter()
                export_graph_jsonl(
                    property_graph_index=pg_index,
                    output_path=jsonl_path,
                    seed_node_ids=seeds,
                )
                _record_export(
                    jsonl_path,
                    "jsonl",
                    duration_ms=(time.perf_counter() - start_json) * 1000.0,
                )
            except Exception as exc:
                logger.warning("Graph JSONL export failed (snapshot): {}", exc)
                log_export_event(
                    {
                        "export_performed": False,
                        "export_type": "graph_jsonl",
                        "context": "snapshot",
                        "error": str(exc),
                    }
                )

            parquet_path = _timestamped_export_path(graph_dir, "parquet")
            try:
                start_parquet = time.perf_counter()
                export_graph_parquet(
                    property_graph_index=pg_index,
                    output_path=parquet_path,
                    seed_node_ids=seeds,
                )
                _record_export(
                    parquet_path,
                    "parquet",
                    duration_ms=(time.perf_counter() - start_parquet) * 1000.0,
                )
            except Exception as exc:
                logger.warning("Graph Parquet export failed (snapshot): {}", exc)
                log_export_event(
                    {
                        "export_performed": False,
                        "export_type": "graph_parquet",
                        "context": "snapshot",
                        "error": str(exc),
                    }
                )

        uploads_dir = settings_obj.data_dir / "uploads"
        corpus_paths = (
            [p for p in uploads_dir.glob("**/*") if p.is_file()]
            if uploads_dir.exists()
            else []
        )
        chash = compute_corpus_hash(corpus_paths, base_dir=uploads_dir)
        cfg = current_config_dict(settings_obj)
        cfg_hash = compute_config_hash(cfg)
        versions: dict[str, str] = {"app": settings_obj.app_version}
        with contextlib.suppress(Exception):  # pragma: no cover
            import llama_index  # type: ignore[import]

            versions["llama_index"] = getattr(llama_index, "__version__", "unknown")

        embed_model_name = "unknown"
        with contextlib.suppress(Exception):
            service_context = getattr(vector_index, "service_context", None)
            embed_model = getattr(service_context, "embed_model", None)
            if embed_model is not None:
                embed_model_name = getattr(embed_model, "model_name", "unknown")
        versions.setdefault("embed_model", embed_model_name)
        with contextlib.suppress(Exception):  # pragma: no cover
            from qdrant_client import (
                __version__ as qdrant_version,  # type: ignore[import]
            )

            versions.setdefault("qdrant_client", qdrant_version)
        if hasattr(settings_obj.database, "client_version"):
            versions.setdefault(
                "vector_client", str(settings_obj.database.client_version)
            )

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
        final = mgr.finalize_snapshot(workspace)
        return final
    except Exception:
        mgr.cleanup_tmp(workspace)
        raise


__all__ = ["rebuild_snapshot"]
