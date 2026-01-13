"""Unit tests for TextEmbedder (BGE-M3).

These tests remain fully offline by stubbing the backend.
"""

import numpy as np
import pytest

from src.models.embeddings import TextEmbedder


@pytest.mark.unit
def test_text_embedder_dense_and_sparse_monkeypatched():
    """Verify shape, normalization, and sparse structure using a stub."""

    class _Stub:
        def encode(self, texts, **kwargs):
            n = len(texts)
            dense = np.random.randn(n, 1024).astype(np.float32)
            sparse = [{1: 0.5, 3: 0.3} for _ in range(n)]
            return {"dense_vecs": dense, "lexical_weights": sparse}

    t = TextEmbedder(device="cpu")
    # Avoid importing FlagEmbedding: patch backend directly
    t._backend = _Stub()  # type: ignore[attr-defined]
    out = t.encode_text(
        ["a", "b"], return_dense=True, return_sparse=True, normalize=True
    )

    assert "dense" in out
    assert isinstance(out["dense"], np.ndarray)
    assert out["dense"].shape == (2, 1024)
    # L2 normalized rows
    norms = np.linalg.norm(out["dense"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert "sparse" in out
    assert isinstance(out["sparse"], list)
    assert isinstance(out["sparse"][0], dict)


@pytest.mark.unit
def test_text_embedder_empty_inputs():
    """Verify empty input returns empty arrays and lists."""
    t = TextEmbedder(device="cpu")
    out = t.encode_text([], return_dense=True, return_sparse=True)
    assert out["dense"].shape == (0, 1024)
    assert out["sparse"] == []


@pytest.mark.unit
def test_text_embedder_lazy_loads_flagembedding_backend(monkeypatch):
    """Exercise the FlagEmbedding import path without importing the real backend."""
    import sys
    from types import ModuleType

    class _StubBGEM3:
        def __init__(
            self, _model_name: str, *, use_fp16: bool, devices: list[str]
        ) -> None:
            self.use_fp16 = bool(use_fp16)
            self.devices = list(devices)

        def encode(self, texts: list[str], **_kwargs):
            n = len(texts)
            dense = np.zeros((n, 8), dtype=np.float32)
            sparse = [{1: 0.5} for _ in range(n)]
            return {"dense_vecs": dense, "lexical_weights": sparse}

    stub_mod = ModuleType("FlagEmbedding")
    stub_mod.BGEM3FlagModel = _StubBGEM3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "FlagEmbedding", stub_mod)

    t = TextEmbedder(device="cpu")
    out = t.encode_text(
        ["hello"], return_dense=True, return_sparse=True, normalize=False
    )
    assert out["dense"].shape == (1, 8)
    assert out["sparse"] == [{1: 0.5}]
