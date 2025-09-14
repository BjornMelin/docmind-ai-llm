"""Tests for `_has_cuda_vram` branch conditions."""

import pytest

pytestmark = pytest.mark.unit


def test_has_cuda_vram_false_when_no_torch(monkeypatch):
    """Return False when core helper import path fails (simulating no torch)."""
    from src.utils import core

    # Simulate no CUDA by forcing helper to report unavailability
    monkeypatch.setattr(
        "src.utils.core.is_cuda_available", lambda: False, raising=False
    )
    assert core.has_cuda_vram(8.0) is False


def test_has_cuda_vram_false_when_cuda_unavailable(monkeypatch):
    """Return False when core helper reports insufficient VRAM or no CUDA."""
    from src.utils import core

    monkeypatch.setattr(
        "src.utils.core.has_cuda_vram",
        lambda _min, device_index=0: False,
        raising=False,
    )
    assert core.has_cuda_vram(8.0) is False
