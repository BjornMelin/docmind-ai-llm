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
    vision_siglip._cached.cache_clear()

    model1, proc1, device1 = vision_siglip.load_siglip(
        vision_siglip.DEFAULT_SIGLIP_MODEL_ID, "cpu"
    )
    model2, proc2, device2 = vision_siglip.load_siglip(
        vision_siglip.DEFAULT_SIGLIP_MODEL_ID, "cpu"
    )

    assert call_count == {"model": 1, "processor": 1}
    assert model1 is model2
    assert proc1 is proc2
    assert device1 == device2 == "cpu"


@pytest.mark.unit
def test_load_siglip_does_not_apply_default_revision_to_custom_model(monkeypatch):
    """Verify custom model IDs do not inherit the default model commit pin."""
    revisions: list[str | None] = []

    def _select(device: str) -> str:
        return device

    def _record_revision(_model_id: str, revision: str | None = None):
        revisions.append(revision)
        return object()

    monkeypatch.setattr(vision_siglip, "select_device", _select)

    transformers = types.SimpleNamespace(
        SiglipModel=types.SimpleNamespace(from_pretrained=_record_revision),
        SiglipProcessor=types.SimpleNamespace(from_pretrained=_record_revision),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    vision_siglip._cached.cache_clear()

    vision_siglip.load_siglip(
        "example/custom-siglip",
        "cpu",
    )

    assert revisions == [None, None]


@pytest.mark.unit
def test_load_siglip_preserves_explicit_custom_revision(monkeypatch):
    """Verify custom model IDs can still opt into a matching revision pin."""
    revisions: list[str | None] = []

    monkeypatch.setattr(vision_siglip, "select_device", lambda device: device)

    def _record_revision(_model_id: str, revision: str | None = None):
        revisions.append(revision)
        return object()

    transformers = types.SimpleNamespace(
        SiglipModel=types.SimpleNamespace(from_pretrained=_record_revision),
        SiglipProcessor=types.SimpleNamespace(from_pretrained=_record_revision),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    vision_siglip._cached.cache_clear()

    vision_siglip.load_siglip(
        "example/custom-siglip",
        "cpu",
        revision=vision_siglip.DEFAULT_SIGLIP_MODEL_REVISION,
    )

    assert revisions == [
        vision_siglip.DEFAULT_SIGLIP_MODEL_REVISION,
        vision_siglip.DEFAULT_SIGLIP_MODEL_REVISION,
    ]


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
            from_pretrained=lambda _model_id, revision=None: _Model()
        ),
        SiglipProcessor=types.SimpleNamespace(
            from_pretrained=lambda _model_id, revision=None: object()
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


@pytest.mark.unit
def test_siglip_features_normalizes_zero_vectors_without_nan() -> None:
    """Verify zero feature rows remain finite when normalization is enabled."""
    torch = pytest.importorskip("torch")

    out = vision_siglip.siglip_features(torch.zeros((1, 4), dtype=torch.float32))

    assert np.all(np.isfinite(out.detach().numpy()))
    assert np.all(out.detach().numpy() == 0.0)
