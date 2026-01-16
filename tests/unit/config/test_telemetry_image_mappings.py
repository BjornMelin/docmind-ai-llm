"""Unit tests for flat env bridges into nested settings groups."""

from __future__ import annotations

import base64

import pytest

from src.config.settings import DocMindSettings

pytestmark = pytest.mark.unit


def test_env_maps_telemetry_flat_vars_into_nested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCMIND_TELEMETRY_DISABLED", "true")
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "0.25")
    monkeypatch.setenv("DOCMIND_TELEMETRY_ROTATE_BYTES", "1234")

    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    assert s.telemetry.disabled is True
    assert s.telemetry.sample == 0.25
    assert s.telemetry.rotate_bytes == 1234


def test_env_maps_image_encryption_flat_vars_into_nested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = b"k" * 32
    monkeypatch.setenv(
        "DOCMIND_IMG_AES_KEY_BASE64", base64.b64encode(key).decode("ascii")
    )
    monkeypatch.setenv("DOCMIND_IMG_KID", "kid-1")
    monkeypatch.setenv("DOCMIND_IMG_DELETE_PLAINTEXT", "true")

    s = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    assert s.image_encryption.aes_key_base64 is not None
    assert s.image_encryption.kid == "kid-1"
    assert s.image_encryption.delete_plaintext is True


def test_invalid_image_encryption_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCMIND_IMG_AES_KEY_BASE64", "not-base64")
    with pytest.raises(
        Exception, match="DOCMIND_IMG_AES_KEY_BASE64 must be valid base64"
    ):
        _ = DocMindSettings(_env_file=None)  # type: ignore[arg-type]


def test_hashing_hmac_secret_error_message_points_to_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCMIND_HASHING__HMAC_SECRET", "too-short")
    with pytest.raises(Exception, match="DOCMIND_HASHING__HMAC_SECRET"):
        _ = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
