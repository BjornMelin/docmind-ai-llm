"""SnapshotManager for index and property graph persistence (SPEC-014).

Implements atomic, versioned snapshots with a manifest and single-writer lock.
The manifest carries corpus/config hashes for staleness detection.

Provides atomic snapshots of vector and graph indices with a manifest for
staleness detection. Uses a directory lock to ensure single-writer semantics
and an atomic rename to finalize snapshots.

Design constraints (library-first):
- Persist vector index via ``index.storage_context.persist(path)``.
- Persist property graph store via its documented ``persist(path)`` method
  when available; otherwise, skip gracefully.
- Do not mutate indices.
"""

from __future__ import annotations

import json
import os
import posixpath
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from loguru import logger

from src.config.settings import settings


@dataclass(frozen=True)
class SnapshotPaths:
    base_dir: Path
    lock_file: Path


def _snapshot_paths() -> SnapshotPaths:
    base = settings.data_dir / "storage"
    return SnapshotPaths(base_dir=base, lock_file=base / ".lock")


def _acquire_lock(timeout: float = 120.0) -> bool:
    paths = _snapshot_paths()
    start = time.perf_counter()
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            # Simple exclusive create; fails if exists
            fd = os.open(paths.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if (time.perf_counter() - start) > timeout:
                return False
            time.sleep(0.2)


def _release_lock() -> None:
    paths = _snapshot_paths()
    try:
        paths.lock_file.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("Snapshot lock cleanup failed: %s", exc)


def begin_snapshot() -> Path:
    """Create a temp snapshot dir under storage with a lock.

    Returns the temporary directory path (e.g., storage/_tmp-<ts>). Caller MUST
    call `finalize_snapshot` or `cleanup_tmp` and `_release_lock` eventually.
    """
    if not _acquire_lock():  # pragma: no cover - time sensitive
        raise TimeoutError("Snapshot lock acquisition timed out")
    paths = _snapshot_paths()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    tmp_dir = paths.base_dir / f"_tmp-{ts}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "vector").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "graph").mkdir(parents=True, exist_ok=True)
    return tmp_dir


def persist_vector_index(index: Any, out_dir: Path) -> None:
    """Persist vector index storage context under `out_dir`.

    Uses LlamaIndex StorageContext to persist artifacts.
    """
    try:
        index.storage_context.persist(persist_dir=str(out_dir))  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - library path
        raise RuntimeError(f"Persist vector index failed: {exc}") from exc


def persist_graph_store(store: Any, out_dir: Path) -> None:
    """Persist property graph store to a directory at `out_dir`.

    Uses documented persist_dir semantics for property graph stores.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            store.persist(persist_dir=str(out_dir))  # type: ignore[attr-defined]
        except TypeError:
            # Fallback for stores expecting positional arg
            store.persist(str(out_dir))
    except Exception as exc:  # pragma: no cover - library path
        raise RuntimeError(f"Persist graph store failed: {exc}") from exc


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # pragma: no cover - best-effort
        pass


def finalize_snapshot(tmp_dir: Path) -> Path:
    """Atomically finalize snapshot and always release the lock.

    - Ensures a unique final directory name by appending a numeric suffix if a
      timestamp collision occurs (e.g., 20250101T000000-1).
    - Calls ``_release_lock()`` even if rename fails to avoid lock leaks.
    """
    paths = _snapshot_paths()
    if not tmp_dir.exists():  # pragma: no cover
        _release_lock()
        raise FileNotFoundError(f"tmp snapshot dir missing: {tmp_dir}")
    # ensure metadata is flushed
    _fsync_dir(tmp_dir)
    ts_final = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    # Find a unique final directory name
    candidate = paths.base_dir / ts_final
    if candidate.exists():
        suffix = 1
        while True:
            alt = paths.base_dir / f"{ts_final}-{suffix}"
            if not alt.exists():
                candidate = alt
                break
            suffix += 1
    try:
        tmp_dir.rename(candidate)
        _fsync_dir(paths.base_dir)
        logger.info("Snapshot finalized at %s", candidate)
        return candidate
    finally:
        _release_lock()


def cleanup_tmp(tmp_dir: Path) -> None:
    """Remove a temporary snapshot dir and release lock (best-effort)."""
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    finally:
        _release_lock()


def _manifest_path(snapshot_dir: Path) -> Path:
    return snapshot_dir / "manifest.json"


def write_manifest(tmp_dir: Path, manifest: dict[str, Any]) -> None:
    """Write snapshot manifest to a temporary directory.

    Args:
        tmp_dir: Temporary directory path where the manifest should be written.
        manifest: Dictionary containing manifest data with corpus and config hashes.

    The manifest file is written as formatted JSON with proper encoding.
    """
    out = _manifest_path(tmp_dir)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_snapshot_dir() -> Path | None:
    """Get the most recent snapshot directory.

    Returns:
        Path to the latest snapshot directory, or None if no snapshots exist.

    Scans the snapshots base directory for valid snapshot directories
    (excluding temporary ones starting with "_tmp-") and returns the
    most recently created one.
    """
    paths = _snapshot_paths()
    if not paths.base_dir.exists():
        return None
    dirs = [
        p
        for p in paths.base_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_tmp-")
    ]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def load_vector_index(snapshot_dir: Path | None = None) -> Any | None:
    """Load a vector index from a snapshot using LlamaIndex storage loaders.

    Returns a VectorStoreIndex-compatible instance or None if loading fails.
    """
    try:
        from llama_index.core import StorageContext, load_index_from_storage
    except Exception:  # pragma: no cover - import guard
        return None

    snap = snapshot_dir or latest_snapshot_dir()
    if not snap:
        return None
    vec_dir = snap / "vector"
    if not vec_dir.exists():
        return None
    try:
        storage = StorageContext.from_defaults(persist_dir=str(vec_dir))
        return load_index_from_storage(storage)
    except Exception:  # pragma: no cover - defensive
        return None


def load_property_graph_index(snapshot_dir: Path | None = None) -> Any | None:
    """Load a PropertyGraphIndex from a snapshot dir.

    Loads the graph store via SimplePropertyGraphStore.from_persist_dir and wraps it
    with PropertyGraphIndex.from_existing.
    """
    try:
        from llama_index.core import PropertyGraphIndex
        from llama_index.core.graph_stores import SimplePropertyGraphStore
    except Exception:  # pragma: no cover - import guard
        return None

    snap = snapshot_dir or latest_snapshot_dir()
    if not snap:
        return None
    graph_dir = snap / "graph"
    if not graph_dir.exists():
        return None
    try:
        store = SimplePropertyGraphStore.from_persist_dir(str(graph_dir))
        return PropertyGraphIndex.from_existing(property_graph_store=store)
    except Exception:  # pragma: no cover - defensive
        return None


def load_manifest(snapshot_dir: Path | None = None) -> dict[str, Any] | None:
    """Load manifest from a snapshot directory.

    Args:
        snapshot_dir: Specific snapshot directory to load from. If None,
            uses the latest snapshot directory.

    Returns:
        Dictionary containing manifest data, or None if no manifest exists
        or cannot be parsed.

    The manifest contains corpus and configuration hashes used for
    staleness detection.
    """
    snap = snapshot_dir or latest_snapshot_dir()
    if not snap:
        return None
    m = _manifest_path(snap)
    if not m.exists():
        return None
    try:
        return json.loads(m.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover
        return None


def compute_corpus_hash(
    upload: Path | list[Path] | None = None, *, base_dir: Path | None = None
) -> str:
    """Compute a stable corpus hash for change detection.

    Args:
        upload: Either a directory path to scan for files, a list of file paths,
            or None to use the default uploads directory.
        base_dir: Optional base directory to make file paths relative to when
            computing the hash. When provided, relative POSIX-style paths are
            used for stability across machines.

    Returns:
        SHA256 hash string prefixed with "sha256:" representing the corpus state.

    The hash is computed from file paths, sizes, and modification times to detect
    changes in the document corpus. Files are sorted to ensure deterministic results.
    """
    files: list[Path] = []
    if upload is None:
        base = settings.data_dir / "uploads"
        if base.exists():
            files = [p for p in base.rglob("*") if p.is_file()]
    elif isinstance(upload, Path):
        base = upload
        if base.exists():
            files = [p for p in base.rglob("*") if p.is_file()]
    else:
        files = [p for p in upload if isinstance(p, Path) and p.is_file()]
    items = []
    for p in files:
        try:
            stat = p.stat()
            if base_dir is not None:
                try:
                    rel = p.relative_to(base_dir)
                except ValueError:
                    rel = p
                name = posixpath.join(*rel.parts)
            else:
                name = str(p)
            items.append((name, stat.st_size, stat.st_mtime_ns))
        except OSError:  # pragma: no cover
            continue
    h = sha256()
    for name, size, mtime in sorted(items):
        h.update(f"{name}|{size}|{mtime}".encode())
    return f"sha256:{h.hexdigest()}"


def compute_config_hash(cfg: dict[str, Any] | None = None) -> str:
    """Compute a stable hash of configuration settings for change detection.

    Args:
        cfg: Configuration dictionary to hash. If None, uses current settings
            subset including embedding model, retrieval settings, and GraphRAG config.

    Returns:
        SHA256 hash string prefixed with "sha256:" representing the config state.

    Used to detect when configuration changes require snapshot invalidation.
    """
    if cfg is None:
        cfg = {
            "embedding_model": getattr(settings.embedding, "model_name", None),
            "embedding_dim": getattr(settings.embedding, "dimension", None),
            "retrieval": {
                "fusion_mode": getattr(settings.retrieval, "fusion_mode", None),
                "reranking_top_k": getattr(settings.retrieval, "reranking_top_k", None),
            },
            "graphrag": {
                "enabled": getattr(settings, "enable_graphrag", False),
                "default_path_depth": 1,
            },
        }
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return f"sha256:{sha256(blob.encode('utf-8')).hexdigest()}"


def is_stale(manifest: dict[str, Any] | None) -> bool:
    """Check if a snapshot is stale by comparing hashes with current state.

    Args:
        manifest: Snapshot manifest dictionary containing corpus and config hashes.
            If None, returns True (considered stale).

    Returns:
        True if the snapshot is stale (corpus or config has changed), False otherwise.

    Compares current corpus and configuration hashes against those stored in
    the manifest to determine if the snapshot needs to be refreshed.
    """
    if not manifest:
        return True
    base = settings.data_dir / "uploads"
    cur = {
        "corpus_hash": compute_corpus_hash(base, base_dir=base),
        "config_hash": compute_config_hash(),
    }
    return (
        manifest.get("corpus_hash") != cur["corpus_hash"]
        or manifest.get("config_hash") != cur["config_hash"]
    )


__all__ = [
    "begin_snapshot",
    "cleanup_tmp",
    "compute_config_hash",
    "compute_corpus_hash",
    "finalize_snapshot",
    "is_stale",
    "latest_snapshot_dir",
    "load_manifest",
    "load_property_graph_index",
    "load_vector_index",
    "persist_graph_store",
    "persist_vector_index",
    "write_manifest",
]


class SnapshotManager:
    """Class-based snapshot manager (UI compatibility layer)."""

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def begin_snapshot(self) -> Path:
        return begin_snapshot()

    def persist_vector_index(self, index: Any, tmp_dir: Path) -> None:
        persist_vector_index(index, tmp_dir / "vector")

    def persist_graph_store(self, store: Any, tmp_dir: Path) -> None:
        persist_graph_store(store, tmp_dir / "graph")

    def write_manifest(
        self,
        tmp_dir: Path,
        *,
        index_id: str,
        graph_store_type: str,
        vector_store_type: str,
        corpus_hash: str,
        config_hash: str,
        versions: dict[str, Any] | None = None,
    ) -> None:
        # Enrich versions and manifest metadata
        pkg_versions: dict[str, Any] = {}
        try:
            import llama_index  # type: ignore

            pkg_versions["llama_index"] = getattr(llama_index, "__version__", None)
        except Exception:  # pragma: no cover - best-effort
            import logging

            logging.getLogger(__name__).debug(
                "Unable to capture llama_index version", exc_info=True
            )
        try:
            import qdrant_client  # type: ignore

            pkg_versions["qdrant_client"] = getattr(qdrant_client, "__version__", None)
        except Exception:  # pragma: no cover - best-effort
            import logging

            logging.getLogger(__name__).debug(
                "Unable to capture qdrant_client version", exc_info=True
            )
        if getattr(settings, "embedding", None) is not None:
            pkg_versions["embed_model"] = getattr(
                settings.embedding, "model_name", None
            )
        if versions:
            pkg_versions.update(versions)

        data = {
            "index_id": index_id,
            "graph_store_type": graph_store_type,
            "vector_store_type": vector_store_type,
            "corpus_hash": corpus_hash,
            "config_hash": config_hash,
            "created_at": datetime.now(UTC).isoformat(),
            "versions": pkg_versions,
            "schema_version": 1,
            "persist_format_version": 1,
            "complete": True,
        }
        write_manifest(tmp_dir, data)

    def finalize_snapshot(self, tmp_dir: Path) -> Path:
        return finalize_snapshot(tmp_dir)

    def cleanup_tmp(self, tmp_dir: Path) -> None:
        cleanup_tmp(tmp_dir)
