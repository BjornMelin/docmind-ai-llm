"""Unit tests for PDF helper functions (no rendering)."""

import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def test_phash_basic():
    from src.processing.pdf_pages import _phash

    img = Image.new("RGB", (8, 8), color="white")
    h = _phash(img, hash_size=4)
    # 4x4 -> 16 bits -> 4 hex chars
    assert isinstance(h, str)
    assert len(h) == 4
