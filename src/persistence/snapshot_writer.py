"""Helpers for managing snapshot workspaces and manifest emission."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_FORMAT_VERSION = "1.0"
_MANIFEST_FILENAMES = {
    "manifest.jsonl",
    "manifest.meta.json",
    "manifest.checksum",
}


@dataclass(slots=True)
class SnapshotWorkspace:
    """Filesystem layout for an in-progress snapshot."""

    root: Path

    @property
    def vector_dir(self) -> Path:
        """Directory for persisted vector indices."""
        return self.root / "vector"

    @property
    def graph_dir(self) -> Path:
        """Directory for persisted property graph data."""
        return self.root / "graph"


__all__ = [
    "MANIFEST_FORMAT_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "SnapshotWorkspace",
    "iter_payload_files",
    "load_manifest_entries",
    "mark_manifest_complete",
    "start_workspace",
    "write_manifest",
]


def start_workspace(base_dir: Path) -> SnapshotWorkspace:
    """Create a temporary workspace rooted at ``base_dir``.

    Args:
        base_dir: Parent directory that houses all snapshot versions.

    Returns:
        SnapshotWorkspace: Descriptor for the created workspace.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    workspace = base_dir / f"_tmp-{uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "vector").mkdir(parents=True, exist_ok=True)
    (workspace / "graph").mkdir(parents=True, exist_ok=True)
    logger.debug("Created snapshot workspace {}", workspace.name)
    return SnapshotWorkspace(workspace)


def iter_payload_files(workspace_path: Path) -> Iterator[Path]:
    """Yield manifest payload files contained within ``workspace_path``."""
    for path in workspace_path.rglob("*"):
        if not path.is_file():
            continue
        if path.name in _MANIFEST_FILENAMES:
            continue
        yield path


def write_manifest(workspace_path: Path, manifest_meta: dict[str, Any]) -> None:
    """Write manifest artifacts (JSONL + meta + checksum) for a workspace.

    Args:
        workspace_path: Workspace directory containing payload files.
        manifest_meta: Metadata describing the snapshot (versions, hashes, etc.).
    """
    entries: list[dict[str, Any]] = []
    for payload in sorted(iter_payload_files(workspace_path)):
        relative = payload.relative_to(workspace_path).as_posix()
        entries.append(
            {
                "path": relative,
                "sha256": _sha256(payload),
                "size_bytes": payload.stat().st_size,
                "content_type": _guess_content_type(payload),
            }
        )

    manifest_path = workspace_path / "manifest.jsonl"
    serialized_lines = (
        json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        for entry in entries
    )
    serialized = "\n".join(serialized_lines) + "\n"
    _write_text_atomic(manifest_path, serialized)

    meta_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "persist_format_version": MANIFEST_FORMAT_VERSION,
        "complete": False,
    }
    meta_payload.update(manifest_meta)
    meta_payload.setdefault("created_at", datetime.now(UTC).isoformat())
    meta_path = workspace_path / "manifest.meta.json"
    _write_json_atomic(meta_path, meta_payload)

    checksum_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_sha256": _hash_manifest(entries, meta_payload),
        "created_at": datetime.now(UTC).isoformat(),
    }
    checksum_path = workspace_path / "manifest.checksum"
    _write_json_atomic(checksum_path, checksum_payload)
    _fsync_dir(workspace_path)


def mark_manifest_complete(snapshot_dir: Path) -> None:
    """Mark manifest metadata as complete and refresh checksum."""
    meta_path = snapshot_dir / "manifest.meta.json"
    if not meta_path.exists():
        return
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if payload.get("complete") is True:
        return
    payload["complete"] = True
    _write_json_atomic(meta_path, payload)
    entries = load_manifest_entries(snapshot_dir)
    checksum_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_sha256": _hash_manifest(entries, payload),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json_atomic(snapshot_dir / "manifest.checksum", checksum_payload)
    _fsync_dir(snapshot_dir)


def load_manifest_entries(snapshot_dir: Path) -> list[dict[str, Any]]:
    """Return manifest entry rows for ``snapshot_dir``."""
    manifest_path = snapshot_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed manifest entry in {}", manifest_path.name
                )
    return entries


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_manifest(entries: list[dict[str, Any]], manifest_meta: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for entry in entries:
        hasher.update(entry["sha256"].encode("utf-8"))
    hasher.update(
        json.dumps(manifest_meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    return hasher.hexdigest()


def _guess_content_type(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "application/x-ndjson"
    if path.suffix.lower() == ".parquet":
        return "application/x-parquet"
    if path.suffix.lower() == ".json":
        return "application/json"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    _fsync_file(tmp)
    os.replace(tmp, path)
    _fsync_file(path)


def _write_text_atomic(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    _fsync_file(tmp)
    os.replace(tmp, path)
    _fsync_file(path)


def _fsync_file(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except FileNotFoundError:
        return


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
