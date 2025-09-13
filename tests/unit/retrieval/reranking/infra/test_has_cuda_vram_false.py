"""Tests for `_has_cuda_vram` branch conditions."""

import pytest

pytestmark = pytest.mark.unit


def test_has_cuda_vram_false_when_no_torch(monkeypatch):
    """Return False when core helper import path fails (simulating no torch)."""
    from src.retrieval import reranking as rr

    # Simulate ImportError from core helper
    def _raise_import(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ImportError("torch not available")

    monkeypatch.setattr("src.utils.core.has_cuda_vram", _raise_import, raising=False)
    assert rr._has_cuda_vram(8.0) is False


def test_has_cuda_vram_false_when_cuda_unavailable(monkeypatch):
    """Return False when core helper reports insufficient VRAM or no CUDA."""
    from src.retrieval import reranking as rr

    monkeypatch.setattr(
        "src.utils.core.has_cuda_vram", lambda _min: False, raising=False
    )
    assert rr._has_cuda_vram(8.0) is False
