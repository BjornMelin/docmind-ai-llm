"""GPU info positive-path tests using a fake torch module."""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def test_gpu_info_with_fake_torch(monkeypatch):
    # Fake torch.cuda API
    class _Props:
        major = 7
        minor = 5
        total_memory = 8 * 1024**3  # 8GB

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(_idx):
            return "FakeGPU"

        @staticmethod
        def get_device_properties(_idx):
            return _Props()

        @staticmethod
        def memory_allocated(_idx=0):
            return 512 * 1024**2  # 512MB

    fake_torch = SimpleNamespace(cuda=_Cuda)

    mod = importlib.import_module("src.utils.storage")
    monkeypatch.setitem(importlib.sys.modules, "torch", fake_torch)
    importlib.reload(mod)

    info = mod.get_safe_gpu_info()
    assert info["cuda_available"] is True
    assert info["device_count"] == 1
    assert info["device_name"] == "FakeGPU"
    assert info["total_memory_gb"] >= 7.5
    assert info["allocated_memory_gb"] >= 0.4

    # Also hit get_safe_vram_usage positive path
    vram = mod.get_safe_vram_usage()
    assert isinstance(vram, float)
    assert vram >= 0.4
