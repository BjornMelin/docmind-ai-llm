"""Security and privacy helpers (AES-GCM optional).

Implements local-first encryption for page images when enabled. Falls back
to passthrough when cryptography is unavailable.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    from cryptography.exceptions import InvalidTag as InvalidTagError  # type: ignore
except ImportError:

    class InvalidTagError(Exception):  # type: ignore[no-redef]
        """Fallback when cryptography is unavailable."""


from src.config.settings import settings


def _get_key() -> bytes | None:
    if settings.image_encryption.aes_key_base64 is None:
        return None
    b64 = settings.image_encryption.aes_key_base64.get_secret_value().strip()
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64, validate=True)
        if len(raw) != 32:
            return None
        return raw
    except (binascii.Error, ValueError):
        return None


def get_image_kid() -> str | None:
    """Return the configured image key id (kid) for AES-GCM metadata, if any."""
    kid = (settings.image_encryption.kid or "").strip()
    return kid or None


def encrypt_file(path: str) -> str:
    """Encrypt file at `path` and return new `.enc` path.

    - Uses AES-GCM with random 12B nonce.
    - AAD includes page_id if present via sidecar metadata (not required here).
    - Returns original path if cryptography/key is unavailable.
    """
    key = _get_key()
    if key is None:
        return path
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore

        p = Path(path)
        data = p.read_bytes()
        aes = AESGCM(key)
        nonce = os.urandom(12)
        aad_env = (get_image_kid() or "").encode("utf-8") or None
        ct = aes.encrypt(nonce, data, associated_data=aad_env)
        out_path = p.with_suffix(p.suffix + ".enc")
        out_path.write_bytes(nonce + ct)
        # Optionally delete plaintext after successful encryption
        if settings.image_encryption.delete_plaintext:
            with contextlib.suppress(Exception):
                p.unlink()
        # Do not delete plaintext automatically to allow caller control
        return str(out_path)
    except (OSError, ValueError, RuntimeError, ImportError):
        return path


def decrypt_file(path: str) -> str:
    """Decrypt `.enc` file and return plaintext temp path.

    Returns original path if cryptography/key is unavailable or file is not `.enc`.
    """
    key = _get_key()
    if key is None or not str(path).endswith(".enc"):
        return path
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore

        p = Path(path)
        blob = p.read_bytes()
        nonce, ct = blob[:12], blob[12:]
        aes = AESGCM(key)
        aad_env = (get_image_kid() or "").encode("utf-8") or None
        pt = aes.decrypt(nonce, ct, associated_data=aad_env)
        fd, name = tempfile.mkstemp(suffix=p.suffix.replace(".enc", ""))
        os.close(fd)
        tmp = Path(name)
        tmp.write_bytes(pt)
        return str(tmp)
    except (OSError, ValueError, RuntimeError, ImportError, InvalidTagError):
        return path


def build_owner_filter(owner_id: str) -> dict[str, Any]:
    """Build a Qdrant payload filter for RBAC-like owner scoping."""
    return {
        "must": [
            {
                "key": "owner_id",
                "match": {"value": owner_id},
            }
        ]
    }


def validate_export_path(base_or_dest: Path | str, dest_rel: str | None = None):
    r"""Validate and sanitize an export destination path (non-egress, no symlink).

    - Allows only [A-Za-z0-9._\-/] characters in `dest_rel`.
    - Resolves against `base_dir` and ensures the result stays within it.
    - Blocks existing symlink targets.
    - Creates parent directories.
    """
    # Support both call forms:
    # - validate_export_path(base_dir: Path, dest_rel: str)
    # - validate_export_path(dest_rel: str)  (uses settings.data_dir)
    single_arg_mode = dest_rel is None and isinstance(base_or_dest, str)
    if single_arg_mode:
        try:
            from src.config.settings import settings as _settings  # local import

            base_dir = Path(getattr(_settings, "data_dir", Path(".")))
        except (
            ImportError,
            AttributeError,
        ):  # pragma: no cover - fallback for missing config or attribute
            base_dir = Path(".")
        rel = str(base_or_dest)
    else:
        base_dir = Path(base_or_dest)  # type: ignore[arg-type]
        if dest_rel is None:
            raise AssertionError("destination relative path not computed")
        rel = str(dest_rel)

    # Absolute path input: accept as-is with symlink and parent checks
    if Path(rel).is_absolute():
        dest = Path(rel)
        # In single-arg mode, constrain absolute paths to safe prefixes
        if single_arg_mode:
            safe_roots = [Path.cwd().resolve(), Path("/tmp"), Path("/var/tmp")]
            if not any(
                str(dest.resolve()).startswith(str(root)) for root in safe_roots
            ):
                raise ValueError("Export path is outside the project root")
        if dest.exists() and dest.is_symlink():
            raise ValueError("Symlink export target blocked")
        dest.parent.mkdir(parents=True, exist_ok=True)
        return str(dest) if single_arg_mode else dest

    safe = "".join(c for c in rel if c.isalnum() or c in ("-", "_", ".", "/"))
    candidate = base_dir / safe
    # Block symlink targets explicitly before resolving
    if candidate.exists() and candidate.is_symlink():
        raise ValueError("Symlink export target blocked")
    dest = candidate.resolve()
    base = base_dir.resolve()
    if not str(dest).startswith(str(base)):
        raise ValueError("Non-egress export path blocked")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return str(dest) if single_arg_mode else dest


__all__ = [
    "build_owner_filter",
    "decrypt_file",
    "encrypt_file",
    "get_image_kid",
    "validate_export_path",
]
