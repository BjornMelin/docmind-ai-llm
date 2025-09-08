"""UnifiedEmbedder routing and normalization tests (lightweight)."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_unified_embedder_routes_text_and_images(monkeypatch):
    from src.models.embeddings import UnifiedEmbedder

    # Patch underlying embedders to avoid heavy imports
    class _T:
        def encode_text(self, texts, **_):
            arr = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[: len(texts)]
            return {"dense": arr}

    class _I:
        def get_image_embedding(self, _img):
            return np.asarray([0.0, 2.0], dtype=np.float32)

    ue = UnifiedEmbedder(text=_T(), image=_I())
    res = ue.encode(["hello", {"image": "x"}])
    assert "dense" in res
    assert "image_dense" in res
    assert res["dense"].shape[0] == 1
    assert res["image_dense"].shape[0] == 1
