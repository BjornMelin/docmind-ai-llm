"""Siglip adapter device selection and error/shape behavior."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_siglip_adapter_cpu_when_no_torch(monkeypatch):
    # Simulate no torch by removing from sys.modules
    import sys

    from src.utils.siglip_adapter import SiglipEmbedding

    monkeypatch.setitem(sys.modules, "torch", None)

    emb = SiglipEmbedding(device=None)
    assert emb.device == "cpu"


def test_siglip_adapter_returns_zero_on_error(monkeypatch):
    from src.utils.siglip_adapter import SiglipEmbedding

    # Force loader to think model/proc loaded but inference fails
    s = SiglipEmbedding()
    s._model = object()  # type: ignore[attr-defined]
    s._proc = object()  # type: ignore[attr-defined]
    s._dim = 16
    vec = s.get_image_embedding(image=object())
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (16,)
