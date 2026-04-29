"""Unit tests for SiglipEmbedding canonical loader behavior."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.siglip_adapter import SiglipEmbedding


@pytest.mark.unit
def test_ensure_loaded_prefers_unified_loader(monkeypatch):
    sentinel_model = object()
    sentinel_proc = object()

    monkeypatch.setattr(
        "src.utils.siglip_adapter.load_siglip",
        lambda model_id, device, revision=None: (
            sentinel_model,
            sentinel_proc,
            "cuda",
        ),
    )

    emb = SiglipEmbedding(model_id="test", device="cpu")
    emb._ensure_loaded()

    assert emb._model is sentinel_model
    assert emb._proc is sentinel_proc
    assert emb.device == "cuda"


@pytest.mark.unit
def test_get_image_embedding_fails_open_when_loader_fails(monkeypatch):
    monkeypatch.setattr(
        "src.utils.siglip_adapter.load_siglip",
        lambda *_: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    emb = SiglipEmbedding(model_id="test", device="cpu")
    emb._dim = 384
    vec = emb.get_image_embedding(object())

    assert vec.shape == (384,)
    assert np.all(vec == 0.0)


@pytest.mark.unit
def test_get_image_embedding_returns_zero_vector(monkeypatch):
    emb = SiglipEmbedding(model_id="test", device="cpu")
    emb._dim = 256

    monkeypatch.setattr(emb, "_ensure_loaded", lambda: None)
    emb._model = None
    emb._proc = None

    vec = emb.get_image_embedding(object())
    assert vec.shape == (256,)
    assert np.all(vec == 0.0)
