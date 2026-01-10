"""Safe logging helpers that avoid emitting raw PII."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

from src.config import settings
from src.utils.canonicalization import CanonicalizationConfig, compute_hashes


@dataclass(frozen=True)
class RedactionResult:
    """Redacted text + stable fingerprint metadata."""

    redacted: str
    fingerprint: str
    key_id: str | None
    canonicalization_version: str
    hmac_secret_version: str


def _build_canonicalization_config() -> CanonicalizationConfig:
    cfg = settings.hashing
    return CanonicalizationConfig(
        version=cfg.canonicalization_version,
        hmac_secret=cfg.hmac_secret.encode("utf-8"),
        hmac_secret_version=cfg.hmac_secret_version,
        metadata_keys=cfg.metadata_keys,
    )


def _fingerprint_value(value: str, key_id: str | None = None) -> tuple[str, str, str]:
    payload = f"{key_id}:{value}" if key_id else value
    bundle = compute_hashes(
        content=payload.encode("utf-8"),
        metadata={},
        config=_build_canonicalization_config(),
    )
    return (
        bundle.canonical_hmac_sha256,
        bundle.canonicalization_version,
        bundle.hmac_secret_version,
    )


def _format_redacted_string(
    fingerprint: str, canon_version: str, secret_version: str
) -> str:
    return f"[redacted:{fingerprint[:12]}:v{canon_version}:{secret_version}]"


@overload
def redact_pii(
    value: str,
    key_id: str | None = None,
    *,
    return_fingerprint: Literal[False] = False,
) -> str: ...


@overload
def redact_pii(
    value: str,
    key_id: str | None = None,
    *,
    return_fingerprint: Literal[True],
) -> tuple[str, str]: ...


def redact_pii(
    value: str,
    key_id: str | None = None,
    *,
    return_fingerprint: bool = False,
) -> str | tuple[str, str]:
    """Return a deterministic redaction string (and fingerprint when requested).

    Args:
        value: Raw value that must not be logged.
        key_id: Optional key identifier to namespace fingerprints.
        return_fingerprint: When True, return `(redacted, fingerprint)`.
    """
    fingerprint, canon_version, secret_version = _fingerprint_value(value, key_id)
    redacted = _format_redacted_string(fingerprint, canon_version, secret_version)
    if return_fingerprint:
        return redacted, fingerprint
    return redacted


def build_pii_log_entry(value: str, key_id: str | None = None) -> RedactionResult:
    """Return a metadata-only log payload for PII values."""
    fingerprint, canon_version, secret_version = _fingerprint_value(value, key_id)
    redacted = _format_redacted_string(fingerprint, canon_version, secret_version)
    return RedactionResult(
        redacted=redacted,
        fingerprint=fingerprint,
        key_id=key_id,
        canonicalization_version=canon_version,
        hmac_secret_version=secret_version,
    )


__all__ = ["RedactionResult", "build_pii_log_entry", "redact_pii"]
