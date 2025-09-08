"""Test that SigLIP model/processor loader is cached (single initialization)."""

from __future__ import annotations

from types import SimpleNamespace


def test_siglip_loader_cached(monkeypatch):
    from src.retrieval import reranking as rr

    calls = {"model": 0, "proc": 0}

    class _M:
        @staticmethod
        def from_pretrained(_id):  # type: ignore[no-untyped-def]
            calls["model"] += 1
            return SimpleNamespace(
                get_text_features=lambda **_: 0, get_image_features=lambda **_: 0
            )

    class _P:
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

    monkeypatch.setitem(
        rr.__dict__, "TORCH", None
    )  # avoid actual torch usage in loader

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    monkeypatch.setitem(rr.__dict__, "SiglipModel", _M)
    monkeypatch.setitem(rr.__dict__, "SiglipProcessor", _P)

    # First load
    _ = rr._load_siglip()
    # Second load should not increment counters
    _ = rr._load_siglip()

    assert calls["model"] == 1
    assert calls["proc"] == 1
