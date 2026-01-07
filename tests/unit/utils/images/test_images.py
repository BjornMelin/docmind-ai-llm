"""Tests for image helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from src.utils.images import open_image_encrypted


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
