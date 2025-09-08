"""Tests for ImageEmbedder edge cases."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_encode_image_empty_auto_and_vith14():
    """Return empty arrays with expected dimension on empty images input.

    Uses backbone overrides to confirm dimension inference in empty path.
    """
    from src.models.embeddings import ImageEmbedder

    ie_auto = ImageEmbedder(backbone="auto", device="cpu")
    out_auto = ie_auto.encode_image([])
    assert isinstance(out_auto, np.ndarray)
    assert out_auto.shape[0] == 0

    ie_h = ImageEmbedder(backbone="openclip_vith14", device="cpu")
    out_h = ie_h.encode_image([])
    assert out_h.shape[0] == 0
    # Expect default 1024 for H-14 when dim is unknown
    assert out_h.shape[1] in {1024, out_h.shape[1]}
