"""Tests for image helpers."""

from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from src.utils.images import (
    MAX_IMAGE_BYTES,
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


@pytest.mark.unit
def test_open_untrusted_image_verifies_uploaded_bytes() -> None:
    """Uploaded images should be verified and detached from the source stream."""
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(buffer, format="PNG")
    buffer.seek(0)
    previous_limit = Image.MAX_IMAGE_PIXELS

    image = open_untrusted_image(buffer)

    assert image.size == (8, 8)
    assert image.mode == "RGB"
    assert previous_limit == Image.MAX_IMAGE_PIXELS


@pytest.mark.unit
def test_open_untrusted_image_restores_pillow_pixel_limit() -> None:
    """Untrusted validation should not leak Pillow pixel limits globally."""
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(buffer, format="PNG")
    buffer.seek(0)
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_UNTRUSTED_IMAGE_PIXELS * 2
    try:
        open_untrusted_image(buffer)

        assert Image.MAX_IMAGE_PIXELS == MAX_UNTRUSTED_IMAGE_PIXELS * 2
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


@pytest.mark.unit
def test_open_untrusted_image_rejects_malformed_upload() -> None:
    """Malformed uploaded image bytes should be rejected by Pillow verification."""
    with pytest.raises(UnidentifiedImageError):
        open_untrusted_image(BytesIO(b"not an image"))


@pytest.mark.unit
def test_open_untrusted_image_rejects_oversized_upload() -> None:
    """Oversized uploaded image bytes should be rejected before Pillow opens them."""
    upload = BytesIO(b"0" * (MAX_IMAGE_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds the maximum allowed size"):
        open_untrusted_image(upload)

    assert upload.tell() == 0
