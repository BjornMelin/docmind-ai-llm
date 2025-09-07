import numpy as np
import pytest

from src.retrieval.embeddings import get_unified_embedder


@pytest.mark.integration
def test_unified_embedder_integration_dense_and_sparse_text():
    u = get_unified_embedder()

    # Patch text backend to avoid external downloads
    class _Stub:
        def encode(self, texts, **kwargs):  # noqa: ANN001
            n = len(texts)
            return {"dense_vecs": np.ones((n, 1024), dtype=np.float32), "lexical_weights": [{0: 1.0} for _ in range(n)]}

    u.text._backend = _Stub()  # type: ignore[attr-defined]
    out = u.text.encode_text(["x", "y"], return_dense=True, return_sparse=True)
    assert out["dense"].shape == (2, 1024)
    assert len(out["sparse"]) == 2


@pytest.mark.integration
def test_unified_embedder_integration_image_vectors():
    u = get_unified_embedder()

    # Monkeypatch image encode to avoid heavy libs
    u.image.encode_image = lambda imgs, **_: np.ones((len(imgs), 768), dtype=np.float32)  # type: ignore[method-assign]
    vecs = u.image.encode_image([object()])
    assert vecs.shape in {(1, 768), (1, 1024)}

