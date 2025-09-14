"""Unit tests for resolve_device helper in src.utils.core.

These tests use lightweight monkeypatching to simulate torch backends without
importing heavy dependencies or requiring real GPUs.
"""

from __future__ import annotations

import types

import pytest


@pytest.mark.unit
def test_resolve_device_auto_cpu_when_no_accelerator(monkeypatch):
    import src.utils.core as core

    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False), backends=None
    )
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)

    dev, idx = core.resolve_device("auto")
    assert dev == "cpu"
    assert idx is None


@pytest.mark.unit
def test_resolve_device_auto_cuda_uses_current_device(monkeypatch):
    import src.utils.core as core

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 1

    dummy_torch = types.SimpleNamespace(cuda=_Cuda)
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)

    dev, idx = core.resolve_device("auto")
    assert dev == "cuda:1"
    assert idx == 1


@pytest.mark.unit
def test_resolve_device_explicit_cuda_index(monkeypatch):
    import src.utils.core as core

    dev, idx = core.resolve_device("cuda:2")
    assert dev == "cuda:2"
    assert idx == 2


@pytest.mark.unit
def test_resolve_device_mps(monkeypatch):
    import src.utils.core as core

    class _MPS:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Backends:
        mps = _MPS()

    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False), backends=_Backends()
    )
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)

    dev, idx = core.resolve_device("auto")
    assert dev == "mps"
    assert idx is None
