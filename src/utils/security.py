"""Security and privacy helpers (AES-GCM optional).

Implements local-first encryption for page images when enabled. Falls back
to passthrough when cryptography is unavailable.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

_ALG = "AES-256-GCM"
_ENV_KEY = "DOCMIND_IMG_AES_KEY_BASE64"
_ENV_KID = "DOCMIND_IMG_KID"


def redact_pii(text: str) -> str:
    """Return text with PII redaction (stub: no-op).

    Implement real patterns/rules as needed per deployment.
    """
    return text


def _get_key() -> bytes | None:
    b64 = os.getenv(_ENV_KEY, "").strip()
    if not b64:
        return None
    try:
        import base64
        import binascii

        raw = base64.b64decode(b64)
        if len(raw) not in (16, 24, 32):
            return None
        return raw
    except (binascii.Error, ValueError):
        return None


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
        aad_env = os.getenv(_ENV_KID, "").encode("utf-8") or None
        ct = aes.encrypt(nonce, data, associated_data=aad_env)
        out_path = p.with_suffix(p.suffix + ".enc")
        out_path.write_bytes(nonce + ct)
        # Optionally delete plaintext after successful encryption
        if os.getenv("DOCMIND_IMG_DELETE_PLAINTEXT", "0") in {"1", "true", "TRUE"}:
            with contextlib.suppress(Exception):
                p.unlink()
        # Do not delete plaintext automatically to allow caller control
        return str(out_path)
    except Exception:
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
        aad_env = os.getenv(_ENV_KID, "").encode("utf-8") or None
        pt = aes.decrypt(nonce, ct, associated_data=aad_env)
        fd, name = tempfile.mkstemp(suffix=p.suffix.replace(".enc", ""))
        os.close(fd)
        tmp = Path(name)
        tmp.write_bytes(pt)
        return str(tmp)
    except (OSError, ValueError, RuntimeError):
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
