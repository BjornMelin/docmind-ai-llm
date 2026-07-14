"""Unit tests for src.utils.siglip_adapter.SiglipEmbedding.

The tests avoid importing heavy libraries by stubbing internals. They focus on
device selection, lazy-loading behavior, and fail-closed inference errors.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest


def test_siglip_embedding_rejects_missing_loaded_model(monkeypatch):
    """Missing model state is an explicit failure, never a synthetic embedding."""
    from src.utils.siglip_adapter import SiglipEmbedding

    emb = SiglipEmbedding(model_id="dummy/none", device="cpu")

    # Prevent any real loading; set a dimension hint
    monkeypatch.setattr(emb, "_ensure_loaded", lambda: None)
    emb._model = None
    emb._proc = None
    emb._dim = 256

    with pytest.raises(RuntimeError, match="image model did not initialize"):
        emb.get_image_embedding(image=None)


def test_siglip_embedding_choose_device_cpu_when_no_torch(monkeypatch):
    """_choose_device falls back to 'cpu' when torch import fails."""
    from src.utils import siglip_adapter as mod

    # Remove 'torch' from sys.modules to force ImportError path
    monkeypatch.setitem(__import__("sys").modules, "torch", None)

    emb = mod.SiglipEmbedding(model_id="x/y", device=None)
    assert emb.device == "cpu"


def test_siglip_embedding_forward_error_raises(monkeypatch):
    """Inference errors fail closed rather than returning a plausible vector."""
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

    torch_stub: Any = ModuleType("torch")
    torch_stub.no_grad = _NoGrad
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_stub)

    with pytest.raises(RuntimeError, match="image embedding failed") as exc_info:
        emb.get_image_embedding(image=None)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom"


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
    assert s._try_ensure_loaded() is True
    assert s._dim == 640


@pytest.mark.unit
def test_siglip_adapter_try_ensure_loaded_propagates_unexpected_errors(monkeypatch):
    """Load-time errors propagate instead of producing a zero-vector fallback."""
    from src.utils.siglip_adapter import SiglipEmbedding

    s = SiglipEmbedding(model_id="dummy/ok", device="cpu")

    def _raise():
        raise KeyError("boom")

    monkeypatch.setattr(s, "_ensure_loaded", _raise)

    with pytest.raises(KeyError, match="boom"):
        s._try_ensure_loaded()


@pytest.mark.unit
def test_siglip_adapter_uses_config_revision_for_explicit_model(monkeypatch):
    """Explicit model IDs still inherit the configured SigLIP revision."""
    from types import SimpleNamespace

    from src.config import settings as app_settings
    from src.utils.siglip_adapter import SiglipEmbedding

    monkeypatch.setattr(
        app_settings,
        "embedding",
        SimpleNamespace(
            siglip_model_id="google/siglip-base-patch16-224",
            siglip_model_revision="configured-revision",
        ),
        raising=False,
    )

    emb = SiglipEmbedding(model_id="example/custom-siglip", device="cpu")
    assert emb.model_id == "example/custom-siglip"
    assert emb.revision == "configured-revision"
