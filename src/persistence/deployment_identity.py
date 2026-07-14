"""Stable local identity for DocMind-owned persistent resources."""

from __future__ import annotations

import os
import re
import stat
from contextlib import suppress
from pathlib import Path
from uuid import UUID, uuid4

DEPLOYMENT_ID_FILENAME = ".deployment-id"
_SNAPSHOT_VERSION_RE = re.compile(r"\A\d{8}T\d{6}-[0-9a-f]{8}\Z")


class DeploymentIdentityError(RuntimeError):
    """Raised when the local deployment identity cannot be trusted."""


def read_deployment_id(data_dir: Path) -> str:
    """Read and validate the deployment UUID stored beneath ``data_dir``."""
    path = _identity_path(data_dir)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise DeploymentIdentityError("DocMind deployment identity is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DeploymentIdentityError(
            "DocMind deployment identity must be a regular, non-symlink file"
        )
    try:
        raw_value = path.read_text(encoding="ascii").strip()
        parsed = UUID(raw_value)
    except (OSError, UnicodeError, ValueError) as exc:
        raise DeploymentIdentityError(
            "DocMind deployment identity is unreadable or invalid"
        ) from exc
    canonical = str(parsed)
    if raw_value != canonical:
        raise DeploymentIdentityError(
            "DocMind deployment identity must use canonical UUID syntax"
        )
    return canonical


def get_or_create_deployment_id(data_dir: Path) -> str:
    """Return a stable deployment UUID, creating it atomically when absent."""
    root = Path(data_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        return read_deployment_id(root)
    except DeploymentIdentityError:
        if (root / DEPLOYMENT_ID_FILENAME).exists() or (
            root / DEPLOYMENT_ID_FILENAME
        ).is_symlink():
            raise

    if _has_durable_snapshot_state(root):
        raise DeploymentIdentityError(
            "DocMind deployment identity is missing while durable snapshots exist"
        )

    deployment_id = str(uuid4())
    temporary = root / f".{DEPLOYMENT_ID_FILENAME}.{uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        payload = f"{deployment_id}\n".encode("ascii")
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise DeploymentIdentityError(
                "DocMind deployment identity could not be written completely"
            )
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, root / DEPLOYMENT_ID_FILENAME)
        except FileExistsError:
            return read_deployment_id(root)
        _fsync_directory(root)
        return deployment_id
    except DeploymentIdentityError:
        raise
    except OSError as exc:
        raise DeploymentIdentityError(
            "DocMind deployment identity could not be created"
        ) from exc
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _identity_path(data_dir: Path) -> Path:
    return Path(data_dir).expanduser().resolve() / DEPLOYMENT_ID_FILENAME


def _has_durable_snapshot_state(data_dir: Path) -> bool:
    """Return whether identity bootstrapping would rotate retained ownership."""
    storage = data_dir / "storage"
    if storage.is_symlink():
        return True
    if not storage.exists():
        return False
    if not storage.is_dir():
        return True
    current = storage / "CURRENT"
    if current.exists() or current.is_symlink():
        return True
    try:
        return any(
            _SNAPSHOT_VERSION_RE.fullmatch(candidate.name) is not None
            for candidate in storage.iterdir()
        )
    except OSError:
        return True


def _fsync_directory(path: Path) -> None:
    """Persist a newly linked identity entry on filesystems that support it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        with suppress(OSError):
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DEPLOYMENT_ID_FILENAME",
    "DeploymentIdentityError",
    "get_or_create_deployment_id",
    "read_deployment_id",
]
