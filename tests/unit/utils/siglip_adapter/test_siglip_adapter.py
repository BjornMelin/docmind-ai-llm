"""Unit tests for src.utils.siglip_adapter.SiglipEmbedding.

The tests avoid importing heavy libraries by stubbing internals. They focus on
device selection, lazy-loading behavior, and failure fallbacks returning
deterministic zero vectors.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np


def test_siglip_embedding_returns_zeros_when_not_loaded(monkeypatch):
    """If model/proc are absent after _ensure_loaded, returns zero vector of _dim."""
    from src.utils.siglip_adapter import SiglipEmbedding

    emb = SiglipEmbedding(model_id="dummy/none", device="cpu")

    # Prevent any real loading; set a dimension hint
    monkeypatch.setattr(emb, "_ensure_loaded", lambda: None)
    emb._model = None
    emb._proc = None
    emb._dim = 256

    vec = emb.get_image_embedding(image=None)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (256,)
    assert np.allclose(vec, 0.0)


def test_siglip_embedding_choose_device_cpu_when_no_torch(monkeypatch):
    """_choose_device falls back to 'cpu' when torch import fails."""
    from src.utils import siglip_adapter as mod

    # Remove 'torch' from sys.modules to force ImportError path
    monkeypatch.setitem(__import__("sys").modules, "torch", None)

    emb = mod.SiglipEmbedding(model_id="x/y", device=None)
    assert emb.device == "cpu"


def test_siglip_embedding_forward_error_returns_zero(monkeypatch):
    """If forward pass raises, adapter returns zeros with known dimension."""
    from src.utils.siglip_adapter import SiglipEmbedding

    emb = SiglipEmbedding(model_id="dummy/ok", device="cpu")

    # Stub _ensure_loaded to set model/proc without importing transformers
    class _Model:
        def get_image_features(self, *a, **k):
            raise RuntimeError("boom")

    class _Proc:
        def __call__(self, *a, **k):
            return {"pixel_values": object()}

    def _fake_ensure():
        emb._model = _Model()
        emb._proc = _Proc()
        emb._dim = 128

    monkeypatch.setattr(emb, "_ensure_loaded", _fake_ensure)

    # Provide a tiny stub torch module for no_grad
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_stub = ModuleType("torch")
    torch_stub.no_grad = _NoGrad
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_stub)

    out = emb.get_image_embedding(image=None)
    assert out.shape == (128,)
    assert np.all(out == 0)


def test_siglip_adapter_dim_inference_from_config(monkeypatch):
    """_dim should be inferred from model.config.projection_dim when present."""
    from src.utils.siglip_adapter import SiglipEmbedding

    s = SiglipEmbedding(model_id="dummy/ok", device="cpu")

    class _Model:
        class _Cfg:
            projection_dim = 640

        config = _Cfg()

    class _Proc:
        def __call__(self, *a, **k):
            return {"pixel_values": object()}

    def _fake_ensure():
        s._model = _Model()
        s._proc = _Proc()
        s._dim = getattr(s._model.config, "projection_dim", None)

    monkeypatch.setattr(s, "_ensure_loaded", _fake_ensure)
    _ = s.get_image_embedding(image=None)
    assert s._dim == 640
