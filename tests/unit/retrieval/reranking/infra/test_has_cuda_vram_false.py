"""Tests for `_has_cuda_vram` branch conditions."""

import sys

import pytest

pytestmark = pytest.mark.unit


def test_has_cuda_vram_false_when_no_torch(monkeypatch):
    """Return False when torch cannot be imported."""
    from src.retrieval import reranking as rr

    monkeypatch.setitem(sys.modules, "torch", None)
    assert rr._has_cuda_vram(8.0) is False


def test_has_cuda_vram_false_when_cuda_unavailable(monkeypatch):
    """Return False when torch reports CUDA not available."""
    from src.retrieval import reranking as rr

    class _T:
        class cuda:  # noqa: N801
            @staticmethod
            def is_available():
                return False

    monkeypatch.setitem(sys.modules, "torch", _T)
    assert rr._has_cuda_vram(8.0) is False
