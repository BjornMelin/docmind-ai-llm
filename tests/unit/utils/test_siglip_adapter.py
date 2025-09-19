"""Unit tests for SiglipEmbedding covering unified and fallback paths."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.utils.siglip_adapter import SiglipEmbedding


@pytest.mark.unit
def test_ensure_loaded_prefers_unified_loader(monkeypatch):
    """Ensure SiglipEmbedding uses the unified loader when available."""
    sentinel_model = object()
    sentinel_proc = object()

    stub = types.SimpleNamespace(
        load_siglip=lambda model_id, device: (sentinel_model, sentinel_proc, "cuda")
    )
    monkeypatch.setitem(sys.modules, "src.utils.vision_siglip", stub)

    emb = SiglipEmbedding(model_id="test", device="cpu")
    emb._ensure_loaded()

    assert emb._model is sentinel_model
    assert emb._proc is sentinel_proc
    assert emb.device == "cuda"


@pytest.mark.unit
def test_ensure_loaded_falls_back_to_transformers(monkeypatch):
    """Verify SiglipEmbedding falls back to transformers when unified fails."""
    stub = types.SimpleNamespace(
        load_siglip=lambda *_: (_ for _ in ()).throw(RuntimeError("fail"))
    )
    monkeypatch.setitem(sys.modules, "src.utils.vision_siglip", stub)

    fake_model = types.SimpleNamespace(config=types.SimpleNamespace(projection_dim=384))
    fake_proc = object()
    transformers = types.SimpleNamespace(
        SiglipModel=types.SimpleNamespace(from_pretrained=lambda _mid: fake_model),
        SiglipProcessor=types.SimpleNamespace(from_pretrained=lambda _mid: fake_proc),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    emb = SiglipEmbedding(model_id="test", device="cpu")
    emb._ensure_loaded()

    assert emb._model is fake_model
    assert emb._proc is fake_proc
    assert emb._dim == 384


@pytest.mark.unit
def test_get_image_embedding_returns_zero_vector(monkeypatch):
    """Confirm get_image_embedding returns zeros when model is unavailable."""
    emb = SiglipEmbedding(model_id="test", device="cpu")
    emb._dim = 256

    monkeypatch.setattr(emb, "_ensure_loaded", lambda: None)
    emb._model = None
    emb._proc = None

    vec = emb.get_image_embedding(object())
    assert vec.shape == (256,)
    assert np.all(vec == 0.0)
