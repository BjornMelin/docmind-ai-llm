"""Tests for image helpers."""

from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from src.utils.images import (
    MAX_UNTRUSTED_IMAGE_PIXELS,
    open_image_encrypted,
    open_untrusted_image,
)


def _write_sample_png(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (8, 8), color=color)
    image.save(path)


def test_open_image_plaintext(tmp_path: Path) -> None:
    """Plain images should open without altering the file."""
    image_path = tmp_path / "sample.png"
    _write_sample_png(image_path, (255, 0, 0))

    with open_image_encrypted(str(image_path)) as handle:
        assert handle.size == (8, 8)


def test_open_image_encrypted_invokes_decryptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Encrypted images should be decrypted and temporary files removed."""
    source_image = tmp_path / "plain.png"
    _write_sample_png(source_image, (0, 255, 0))

    encrypted = tmp_path / "cipher.png.enc"
    encrypted.write_bytes(b"placeholder")

    decrypted = tmp_path / "cipher.png"

    def fake_decrypt(path: str) -> str:
        assert path == str(encrypted)
        shutil.copy2(source_image, decrypted)
        return str(decrypted)

    monkeypatch.setattr("src.utils.security.decrypt_file", fake_decrypt)

    with open_image_encrypted(str(encrypted)) as handle:
        assert handle.size == (8, 8)

    assert not decrypted.exists()


def test_open_untrusted_image_verifies_uploaded_bytes() -> None:
    """Uploaded images should be verified and detached from the source stream."""
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(buffer, format="PNG")
    buffer.seek(0)

    image = open_untrusted_image(buffer)

    assert image.size == (8, 8)
    assert image.mode == "RGB"
    assert Image.MAX_IMAGE_PIXELS == MAX_UNTRUSTED_IMAGE_PIXELS


def test_open_untrusted_image_rejects_malformed_upload() -> None:
    """Malformed uploaded image bytes should be rejected by Pillow verification."""
    with pytest.raises(UnidentifiedImageError):
        open_untrusted_image(BytesIO(b"not an image"))
