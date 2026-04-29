"""Tests for src.utils.vision_siglip loader caching behavior."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.utils import vision_siglip


@pytest.mark.unit
def test_load_siglip_uses_cached_loader(monkeypatch):
    """Verify load_siglip caches model/processor instances per model id."""
    call_count = {"model": 0, "processor": 0}

    def _select(device: str) -> str:
        return device

    monkeypatch.setattr(vision_siglip, "select_device", _select)

    def _model_loader(model_id: str, revision: str):
        assert revision == vision_siglip.DEFAULT_SIGLIP_MODEL_REVISION
        call_count["model"] += 1
        return types.SimpleNamespace(to=lambda device: None)

    def _proc_loader(model_id: str, revision: str):
        assert revision == vision_siglip.DEFAULT_SIGLIP_MODEL_REVISION
        call_count["processor"] += 1
        return object()

    transformers = types.SimpleNamespace(
        SiglipModel=types.SimpleNamespace(from_pretrained=_model_loader),
        SiglipProcessor=types.SimpleNamespace(from_pretrained=_proc_loader),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model1, proc1, device1 = vision_siglip.load_siglip("siglip-test", "cpu")
    model2, proc2, device2 = vision_siglip.load_siglip("siglip-test", "cpu")

    assert call_count == {"model": 1, "processor": 1}
    assert model1 is model2
    assert proc1 is proc2
    assert device1 == device2 == "cpu"


@pytest.mark.unit
def test_load_siglip_moves_model_to_mps(monkeypatch):
    """Verify canonical loader preserves Apple Silicon MPS placement."""
    moved_to: list[str] = []

    def _select(_device: str) -> str:
        return "mps"

    class _Model:
        def to(self, device: str):
            moved_to.append(device)
            return self

    monkeypatch.setattr(vision_siglip, "select_device", _select)

    transformers = types.SimpleNamespace(
        SiglipModel=types.SimpleNamespace(
            from_pretrained=lambda _model_id, revision: _Model()
        ),
        SiglipProcessor=types.SimpleNamespace(
            from_pretrained=lambda _model_id, revision: object()
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    vision_siglip._cached.cache_clear()
    _model, _proc, device = vision_siglip.load_siglip("siglip-test", "auto")

    assert device == "mps"
    assert moved_to == ["mps"]


@pytest.mark.unit
def test_siglip_features_accepts_v5_pooler_output() -> None:
    """Verify Transformers v5 output objects normalize like v4 tensors."""
    torch = pytest.importorskip("torch")

    direct = torch.ones((1, 4), dtype=torch.float32)
    wrapped = types.SimpleNamespace(
        pooler_output=torch.ones((1, 4), dtype=torch.float32)
    )

    direct_out = vision_siglip.siglip_features(direct).detach().numpy()
    wrapped_out = vision_siglip.siglip_features(wrapped).detach().numpy()

    assert direct_out.shape == (1, 4)
    assert wrapped_out.shape == (1, 4)
    assert np.linalg.norm(direct_out[0]) == pytest.approx(1.0)
    assert np.linalg.norm(wrapped_out[0]) == pytest.approx(1.0)
