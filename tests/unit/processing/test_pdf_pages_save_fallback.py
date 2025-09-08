"""Tests for image save fallback and encryption metadata in pdf_pages helpers."""

import types

import pytest

pytestmark = pytest.mark.unit


def test_save_with_format_jpeg_fallback_and_encryption(monkeypatch, tmp_path):
    """Force WebP failure and verify JPEG fallback with `.enc` suffix.

    Creates a fake Pixmap-like object and patches PIL.Image.Image.save to raise
    on WebP, then succeed on JPEG. Patches `encrypt_file` to append `.enc`.
    """
    from src.processing import pdf_pages as pp

    # Fake pixmap with RGB samples (n < 4)
    class _Pix:
        n = 3
        width = 2
        height = 2
        samples = b"\x00\x00\x00" * 4  # 2x2 RGB

    # Patch encrypt to append .enc
    monkeypatch.setattr(pp, "encrypt_file", lambda p: str(p) + ".enc")

    # Patch settings flag
    monkeypatch.setattr(
        pp.settings,
        "processing",
        types.SimpleNamespace(encrypt_page_images=True),
        raising=False,
    )

    calls = {"webp": 0, "jpeg": 0}

    # Wrap Image.save behavior
    import PIL.Image

    original_save = PIL.Image.Image.save

    def _save(self, path, fmt=None, **kwargs):
        if str(path).endswith(".webp"):
            calls["webp"] += 1
            raise OSError("webp not supported")
        if str(path).endswith(".jpg"):
            calls["jpeg"] += 1
            return original_save(self, path)
        return original_save(self, path)

    monkeypatch.setattr(PIL.Image.Image, "save", _save, raising=False)

    out_dir = tmp_path
    path, ph = pp._save_with_format(_Pix(), out_dir / "img")
    assert str(path).endswith(".jpg.enc")
    assert isinstance(ph, str)
    assert ph
    assert calls["webp"] >= 1
    assert calls["jpeg"] >= 1
