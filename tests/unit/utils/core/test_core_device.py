"""Unit tests for centralized device/VRAM policy helpers in src.utils.core."""


import types

import pytest


@pytest.mark.unit
def test_select_device_explicit_cpu_returns_cpu(monkeypatch):
    import src.utils.core as core

    # Ensure any CUDA checks are ignored by providing a minimal torch stub
    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)

    assert core.select_device("cpu") == "cpu"


@pytest.mark.unit
def test_select_device_auto_prefers_cuda_when_available(monkeypatch):
    import src.utils.core as core

    class _Cuda:
        @staticmethod
        def is_available() -> bool:  # pragma: no cover - deterministic
            return True

    dummy_torch = types.SimpleNamespace(cuda=_Cuda)
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)

    assert core.select_device("auto") == "cuda"
    assert core.select_device("cuda") == "cuda"


@pytest.mark.unit
def test_select_device_auto_uses_mps_when_no_cuda(monkeypatch):
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

    assert core.select_device("auto") == "mps"
    assert core.select_device("mps") == "mps"


@pytest.mark.unit
def test_has_cuda_vram_true_and_false(monkeypatch):
    import src.utils.core as core

    class _Props:
        def __init__(self, total_gb: float) -> None:
            self.total_memory = int(total_gb * (1024**3))

    class _Cuda:
        def __init__(self, avail: bool, gb: float) -> None:
            self._avail = avail
            self._gb = gb

        def is_available(self) -> bool:
            return self._avail

        def get_device_properties(self, _idx: int):
            return _Props(self._gb)

    # Case 1: CUDA unavailable → False
    dummy_torch = types.SimpleNamespace(cuda=_Cuda(False, 0.0))
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)
    assert core.has_cuda_vram(1.0) is False

    # Case 2: CUDA available with insufficient VRAM → False
    dummy_torch = types.SimpleNamespace(cuda=_Cuda(True, 0.5))
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)
    assert core.has_cuda_vram(1.0) is False

    # Case 3: CUDA available with sufficient VRAM → True
    dummy_torch = types.SimpleNamespace(cuda=_Cuda(True, 24.0))
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)
    assert core.has_cuda_vram(20.0) is True
