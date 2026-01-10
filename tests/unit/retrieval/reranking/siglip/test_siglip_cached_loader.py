"""Test SigLIP loader caching without production hooks.

We inject a fake ``transformers`` module via ``sys.modules`` so the loader
imports our stubs, exercising the cache without modifying production code.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.unit
def test_siglip_loader_cached(monkeypatch):
    from src.retrieval import reranking as rr

    vision = importlib.import_module("src.utils.vision_siglip")

    calls = {"model": 0, "proc": 0}

    class _FakeModel:
        @staticmethod
        def from_pretrained(_id):  # type: ignore[no-untyped-def]
            calls["model"] += 1
            return SimpleNamespace(
                get_text_features=lambda **_: 0,
                get_image_features=lambda **_: 0,
            )

    class _FakeProcessor:
        @staticmethod
        def from_pretrained(_id):  # type: ignore[no-untyped-def]
            calls["proc"] += 1
            return SimpleNamespace()

    monkeypatch.setattr(
        rr,
        "settings",
        SimpleNamespace(
            embedding=SimpleNamespace(siglip_model_id="google/siglip-base-patch16-224"),
            retrieval=SimpleNamespace(
                text_rerank_timeout_ms=10, siglip_timeout_ms=10, colpali_timeout_ms=10
            ),
        ),
        raising=False,
    )

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    # Inject fake transformers module used by the loader
    fake_tf = ModuleType("transformers")
    fake_tf.SiglipModel = _FakeModel
    fake_tf.SiglipProcessor = _FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    assert hasattr(vision, "_cached"), "vision_siglip is expected to define _cached"
    assert hasattr(
        vision._cached, "cache_clear"
    ), "vision_siglip._cached should expose cache_clear"
    vision._cached.cache_clear()

    # First load
    _ = rr._load_siglip()
    # Second load should not increment counters
    _ = rr._load_siglip()

    assert calls["model"] == 1
    assert calls["proc"] == 1
