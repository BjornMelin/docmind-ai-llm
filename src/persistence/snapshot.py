"""SnapshotManager for index persistence (SPEC-014).

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

import contextlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger


def _atomic_mkdir(path: Path) -> None:
    """Create a directory atomically.

    Raises FileExistsError if the directory exists.
    """
    path.mkdir(parents=False, exist_ok=False)


@dataclass
class _DirLock:
    """Simple directory-based lock with timeout.

    Cross-platform by relying on the atomicity of directory creation.
    """

    path: Path
    timeout_s: float = 120.0
    poll_interval_s: float = 0.25

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                _atomic_mkdir(self.path)
                return
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Timeout acquiring lock: {self.path}") from None
                time.sleep(self.poll_interval_s)

    def release(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self.path)


def _fsync_dir(path: Path) -> None:
    """Fsync a directory entry to strengthen rename durability."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # pragma: no cover - best effort on some FS
        pass


def _stable_sha256(parts: Iterable[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def compute_corpus_hash(paths: Iterable[Path]) -> str:
    """Compute a stable SHA-256 over (relpath,size,mtime_ns) sorted by relpath.

    Args:
        paths: Iterable of file paths to include.

    Returns:
        str: Hex digest representing the corpus hash.
    """
    records: list[tuple[str, int, int]] = []
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        rel = str(p)
        records.append(
            (
                rel,
                int(st.st_size),
                int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
            )
        )
    records.sort(key=lambda x: x[0])
    parts = [f"{r[0]}|{r[1]}|{r[2]}" for r in records]
    return _stable_sha256(parts)


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a stable SHA-256 over canonical JSON of a config mapping."""
    # Canonicalize with sorted keys, no whitespace variance
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return _stable_sha256([blob])


@dataclass
class SnapshotPaths:
    """Paths that make up a single snapshot session."""

    root: Path
    tmp_dir: Path
    vector_dir: Path
    graph_dir: Path
    manifest_path: Path


class SnapshotManager:
    """Manage atomic snapshots of retrieval indices.

    Usage:
        mgr = SnapshotManager(base_dir)
        paths = mgr.begin_snapshot()
        mgr.persist_vector_index(index, paths)
        mgr.persist_graph_store(store, paths)
        mgr.write_manifest(paths, meta)
        final_path = mgr.finalize_snapshot(paths)
    """

    def __init__(self, storage_dir: Path) -> None:
        """Initialize snapshot manager bound to a storage directory."""
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = _DirLock(self.storage_dir / ".lock")

    def begin_snapshot(self) -> SnapshotPaths:
        """Acquire lock and create a temp snapshot directory structure."""
        self._lock.acquire()
        tmp = self.storage_dir / f"_tmp-{uuid.uuid4().hex}"
        vector = tmp / "vector"
        graph = tmp / "graph"
        manifest = tmp / "manifest.json"
        vector.mkdir(parents=True, exist_ok=True)
        graph.mkdir(parents=True, exist_ok=True)
        return SnapshotPaths(self.storage_dir, tmp, vector, graph, manifest)

    def persist_vector_index(self, index: Any, paths: SnapshotPaths) -> None:
        """Persist a LlamaIndex vector index to the temp directory."""
        try:
            storage = getattr(index, "storage_context", None)
            if storage is None:
                logger.warning("Index has no storage_context; skipping vector persist")
                return
            storage.persist(persist_dir=str(paths.vector_dir))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Vector persist failed: %s", exc)

    def persist_graph_store(self, graph_store: Any, paths: SnapshotPaths) -> None:
        """Persist a property graph store JSON to the temp directory."""
        try:
            persist = getattr(graph_store, "persist", None)
            if callable(persist):
                out = paths.graph_dir / "graph_store.json"
                persist(str(out))
            else:
                logger.warning("Graph store has no persist(); skipping")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Graph store persist failed: %s", exc)

    def write_manifest(
        self,
        paths: SnapshotPaths,
        *,
        index_id: str | None = None,
        graph_store_type: str | None = None,
        vector_store_type: str | None = None,
        corpus_hash: str | None = None,
        config_hash: str | None = None,
        versions: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write manifest.json with required fields."""
        manifest = {
            "index_id": index_id or uuid.uuid4().hex,
            "graph_store_type": graph_store_type or "unknown",
            "vector_store_type": vector_store_type or "unknown",
            "corpus_hash": corpus_hash or "",
            "config_hash": config_hash or "",
            "created_at": datetime.now(UTC).isoformat(),
            "versions": versions or {},
        }
        if extra:
            manifest.update(extra)
        paths.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _fsync_dir(paths.tmp_dir)

    def finalize_snapshot(self, paths: SnapshotPaths) -> Path:
        """Atomically rename the temp directory into a timestamped snapshot dir.

        Releases the lock after rename.
        """
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        final = paths.root / ts
        # Ensure unique by suffixing if needed
        counter = 0
        candidate = final
        while candidate.exists():  # pragma: no cover - rare
            counter += 1
            candidate = Path(f"{final}_{counter}")
        os.replace(str(paths.tmp_dir), str(candidate))
        _fsync_dir(paths.root)
        # Release lock
        self._lock.release()
        return candidate

    def cleanup_tmp(self, paths: SnapshotPaths) -> None:
        """Cleanup temporary snapshot directory and release lock if held."""
        with logger.catch(message="cleanup_tmp failed"):
            if paths.tmp_dir.exists():
                shutil.rmtree(paths.tmp_dir, ignore_errors=True)
            # Always attempt to release lock
            self._lock.release()


__all__ = [
    "SnapshotManager",
    "SnapshotPaths",
    "compute_config_hash",
    "compute_corpus_hash",
]
