"""Test SigLIP loader caching without production hooks.

We inject a fake ``transformers`` module via ``sys.modules`` so the loader
imports our stubs, exercising the cache without modifying production code.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


def test_siglip_loader_cached(monkeypatch):
    from src.retrieval import reranking as rr

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

    monkeypatch.setitem(rr.__dict__, "TEXT_TRUNCATION_LIMIT", 9999)
    monkeypatch.setitem(rr.__dict__, "SIGLIP_TIMEOUT_MS", 10)
    monkeypatch.setitem(rr.__dict__, "COLPALI_TIMEOUT_MS", 10)
    monkeypatch.setitem(
        rr.__dict__,
        "settings",
        SimpleNamespace(
            embedding=SimpleNamespace(siglip_model_id="google/siglip-base-patch16-224")
        ),
    )

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    # Inject fake transformers module used by the loader
    fake_tf = ModuleType("transformers")
    fake_tf.SiglipModel = _FakeModel
    fake_tf.SiglipProcessor = _FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    # First load
    _ = rr._load_siglip()
    # Second load should not increment counters
    _ = rr._load_siglip()

    assert calls["model"] == 1
    assert calls["proc"] == 1
