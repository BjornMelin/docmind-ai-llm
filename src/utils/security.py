"""Security and privacy helper stubs.

Provides minimal, local-first helpers without external dependencies.
"""

from __future__ import annotations

from typing import Any


def redact_pii(text: str) -> str:
    """Return text with PII redaction (stub: no-op).

    Implement real patterns/rules as needed per deployment.
    """
    return text


def encrypt_file(path: str) -> str:
    """Encrypt file at `path` and return new path (stub: passthrough)."""
    return path


def decrypt_file(path: str) -> str:
    """Decrypt file at `path` and return new path (stub: passthrough)."""
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
