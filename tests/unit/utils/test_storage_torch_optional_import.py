"""Ensure storage module handles optional torch import safely.

Verifies CPU-only environments do not crash and guard paths return safe
defaults when ``torch`` is unavailable.
"""

from __future__ import annotations

import importlib


def test_cpu_only_paths_safe(monkeypatch):  # type: ignore[no-untyped-def]
    """Verify storage helpers remain safe when torch is absent."""
    # Simulate torch not available
    monkeypatch.setitem(importlib.sys.modules, "torch", None)
    # Reload module to apply import-time optional behavior
    mod = importlib.import_module("src.utils.storage")
    importlib.reload(mod)

    # get_safe_vram_usage should not raise and should return 0.0
    vram = mod.get_safe_vram_usage()
    assert isinstance(vram, float)
    assert vram == 0.0

    # get_safe_gpu_info should not raise and return sane defaults
    info = mod.get_safe_gpu_info()
    assert isinstance(info, dict)
    assert info.get("cuda_available") is False
    assert info.get("device_count") == 0
    assert info.get("total_memory_gb") == 0.0


def test_safe_cuda_operation_handles_import_error(monkeypatch):  # type: ignore[no-untyped-def]
    """Ensure safe_cuda_operation returns the default when ImportError occurs."""
    mod = importlib.import_module("src.utils.storage")

    def _op():  # simulate torch access raising ImportError
        raise ImportError("torch not installed")

    val = mod.safe_cuda_operation(_op, operation_name="probe", default_return=123)
    assert val == 123
