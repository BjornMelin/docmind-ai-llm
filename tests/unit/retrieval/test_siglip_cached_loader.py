"""Test that SigLIP model/processor loader is cached (single initialization).

This test injects a fake 'transformers' module into sys.modules so that
_load_siglip's import-on-demand is intercepted cleanly.
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

    # Create a fake transformers module
    fake_tf = ModuleType("transformers")
    fake_tf.SiglipModel = _FakeModel
    fake_tf.SiglipProcessor = _FakeProcessor

    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    # Neutralize torch usage in loader
    monkeypatch.setitem(rr.__dict__, "TORCH", None)
    monkeypatch.setitem(
        rr.__dict__,
        "settings",
        SimpleNamespace(
            embedding=SimpleNamespace(siglip_model_id="google/siglip-base-patch16-224")
        ),
    )

    # First load
    _ = rr._load_siglip()
    # Second load should not increment counters
    _ = rr._load_siglip()

    assert calls["model"] == 1
    assert calls["proc"] == 1
