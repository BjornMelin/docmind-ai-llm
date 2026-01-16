"""Unit tests for image helpers (thumbnail + encrypted open)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _set_test_aes_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config.settings import settings

    # 32 bytes key (AES-256) encoded as base64.
    key = b"k" * 32
    monkeypatch.setattr(
        settings.image_encryption,
        "aes_key_base64",
        base64.b64encode(key).decode("ascii"),
    )
    monkeypatch.setattr(settings.image_encryption, "kid", "test")


def test_ensure_thumbnail_plain_and_encrypted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        from PIL import Image  # type: ignore
    except Exception:  # pragma: no cover
        pytest.skip("Pillow not installed")

    from src.utils.images import ensure_thumbnail, open_image_encrypted

    img_path = tmp_path / "page.webp"
    Image.new("RGB", (64, 32), color=(10, 20, 30)).save(img_path, format="WEBP")

    thumb = ensure_thumbnail(img_path, max_side=32)
    assert thumb.exists()
    assert thumb.suffix == ".webp"

    _set_test_aes_key(monkeypatch)
    enc_thumb = ensure_thumbnail(img_path, max_side=32, encrypt=True)
    assert enc_thumb.exists()
    assert str(enc_thumb).endswith(".webp.enc")

    with open_image_encrypted(str(enc_thumb)) as im:
        assert im.size[0] <= 32
        assert im.size[1] <= 32
