"""Tests for TextEmbedder edge cases."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_encode_text_empty_returns_empty_arrays():
    """Return empty arrays with correct shapes on empty input.

    For empty text input, TextEmbedder should not load the backend and should
    return arrays with zero rows.
    """
    from src.models.embeddings import TextEmbedder

    te = TextEmbedder()
    out = te.encode_text([])
    assert "dense" in out
    assert out["dense"].shape[0] == 0
    assert isinstance(out["dense"], np.ndarray)


def test_encode_text_device_override_raises():
    """Per-call device override should raise ValueError.

    The TextEmbedder binds device at load time and should reject per-call
    overrides.
    """
    from src.models.embeddings import TextEmbedder

    te = TextEmbedder(device="cpu")
    with pytest.raises(ValueError, match="Per-call device override is not supported"):
        te.encode_text(["hello"], device="cuda", return_sparse=False)
