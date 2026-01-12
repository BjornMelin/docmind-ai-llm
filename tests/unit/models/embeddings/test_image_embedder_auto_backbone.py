"""Auto-backbone selection behavior for ImageEmbedder."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_image_embedder_auto_backbone_cpu(monkeypatch):
    from src.models.embeddings import ImageEmbedder

    # Ensure core.has_cuda_vram returns False to emulate low/CPU policy
    monkeypatch.setattr(
        "src.utils.core.has_cuda_vram", lambda *_a, **_k: False, raising=False
    )

    ie = ImageEmbedder(backbone="auto", device="cpu")
    assert ie._choose_auto_backbone() == "openclip_vitl14"


@pytest.mark.unit
def test_image_embedder_auto_backbone_cuda_indexed(monkeypatch):
    from src.models.embeddings import ImageEmbedder

    # Force resolve_device to return a non-zero index and has_cuda_vram to True
    monkeypatch.setattr(
        "src.utils.core.resolve_device", lambda *_a, **_k: ("cuda:2", 2), raising=False
    )
    monkeypatch.setattr(
        "src.utils.core.has_cuda_vram",
        lambda _min, device_index=0: device_index == 2,
        raising=False,
    )

    ie = ImageEmbedder(backbone="auto", device="cuda")
    # Private method call for deterministic test of policy
    assert ie._choose_auto_backbone() == "openclip_vith14"
