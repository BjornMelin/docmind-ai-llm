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
        def encode_image(self, images, **_):
            return np.asarray([[0.0, 2.0] for _ in images], dtype=np.float32)

    ue = UnifiedEmbedder(text=_T(), image=_I())
    res = ue.encode(["hello", {"image": "x"}])
    assert "dense" in res
    assert "image_dense" in res
    assert res["dense"].shape[0] == 1
    assert res["image_dense"].shape[0] == 1


def test_unified_embedder_strict_image_types_rejects_unknown() -> None:
    from src.models.embeddings import UnifiedEmbedder

    ue = UnifiedEmbedder(strict_image_types=True)
    with pytest.raises(TypeError, match="Unsupported image type"):
        ue.encode(["hello", {"not": "an image"}])


def test_unified_embedder_encode_pair_routes_both_sides() -> None:
    from src.models.embeddings import UnifiedEmbedder

    class _T:
        def encode_text(self, texts, **_):  # type: ignore[no-untyped-def]
            return {"dense": np.asarray([[1.0] for _ in texts], dtype=np.float32)}

    class _I:
        def encode_image(self, images, **_):  # type: ignore[no-untyped-def]
            return np.asarray([[2.0] for _ in images], dtype=np.float32)

    ue = UnifiedEmbedder(text=_T(), image=_I(), strict_image_types=True)
    out = ue.encode_pair(["a"], [np.zeros((1,), dtype=np.float32)])
    assert out["dense"].shape == (1, 1)
    assert out["image_dense"].shape == (1, 1)
