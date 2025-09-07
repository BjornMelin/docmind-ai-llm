import numpy as np
import pytest

from src.models.embeddings import TextEmbedder


@pytest.mark.unit
def test_text_embedder_dense_and_sparse_monkeypatched():
    class _Stub:
        def encode(self, texts, **kwargs):  # noqa: ANN001
            n = len(texts)
            dense = np.random.randn(n, 1024).astype(np.float32)
            sparse = [{1: 0.5, 3: 0.3} for _ in range(n)]
            return {"dense_vecs": dense, "lexical_weights": sparse}

    t = TextEmbedder(device="cpu")
    # Avoid importing FlagEmbedding: patch backend directly
    t._backend = _Stub()  # type: ignore[attr-defined]
    out = t.encode_text(["a", "b"], return_dense=True, return_sparse=True, normalize=True)

    assert "dense" in out and isinstance(out["dense"], np.ndarray)
    assert out["dense"].shape == (2, 1024)
    # L2 normalized rows
    norms = np.linalg.norm(out["dense"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert "sparse" in out and isinstance(out["sparse"], list)
    assert isinstance(out["sparse"][0], dict)


@pytest.mark.unit
def test_text_embedder_empty_inputs():
    t = TextEmbedder(device="cpu")
    out = t.encode_text([], return_dense=True, return_sparse=True)
    assert out["dense"].shape == (0, 1024)
    assert out["sparse"] == []

