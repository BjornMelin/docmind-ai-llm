"""Ensure SigLIP processor is called with padding='max_length'."""

from types import SimpleNamespace

from src.retrieval import reranking as rr


def test_siglip_uses_padding_max_length(monkeypatch):
    # Build tiny node with image metadata
    node = SimpleNamespace()
    node.node = SimpleNamespace(
        metadata={"modality": "pdf_page_image", "image_path": "nonexistent.png"}
    )
    node.score = 0.0
    nodes = [node]

    # Track processor call kwargs
    calls = {"kwargs": None}

    class _FakeModel:
        def get_text_features(self, **kwargs):  # pragma: no cover - simplified
            import torch

            return torch.ones((1, 4))

        def get_image_features(self, **kwargs):  # pragma: no cover - simplified
            import torch

            return torch.ones((1, 4))

    class _FakeProc:
        def __call__(self, *args, **kwargs):
            calls["kwargs"] = kwargs
            # Return minimal tensors
            import torch

            return {"input_ids": torch.ones((1, 4), dtype=torch.long)}

    def fake_load():  # force-inject model+processor into closure scope
        return _FakeModel(), _FakeProc(), "cpu"

    monkeypatch.setattr(rr, "_load_siglip", fake_load)

    out = rr._siglip_rescore("hello", nodes, rr.SIGLIP_TIMEOUT_MS)
    assert out
    # Ensure padding was provided
    assert calls["kwargs"] is not None
    assert calls["kwargs"].get("padding") == "max_length"
