"""Unit tests for SiglipEmbedding canonical loader behavior."""

from __future__ import annotations

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
def test_get_image_embedding_fails_closed_when_loader_fails(monkeypatch):
    def _fail_loader(*_args, **_kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(
        "src.utils.siglip_adapter.load_siglip",
        _fail_loader,
    )

    emb = SiglipEmbedding(model_id="test", device="cpu")
    emb._dim = 384
    with pytest.raises(RuntimeError, match="fail"):
        emb.get_image_embedding(object())


@pytest.mark.unit
def test_get_image_embedding_rejects_uninitialized_loader_state(monkeypatch):
    emb = SiglipEmbedding(model_id="test", device="cpu")
    emb._dim = 256

    monkeypatch.setattr(emb, "_ensure_loaded", lambda: None)
    emb._model = None
    emb._proc = None

    with pytest.raises(RuntimeError, match="image model did not initialize"):
        emb.get_image_embedding(object())
