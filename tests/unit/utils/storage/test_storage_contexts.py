"""Tests for storage GPU/model/cuda context managers and error context."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.unit
def test_gpu_memory_context_smoke(monkeypatch):
    """Test that gpu_memory_context works with mocked CUDA availability."""
    mod = importlib.import_module("src.utils.storage")

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def empty_cache():
            return None

    # Inject fake torch
    monkeypatch.setitem(importlib.sys.modules, "torch", SimpleNamespace(cuda=_Cuda))
    importlib.reload(mod)

    with mod.gpu_memory_context():
        pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_gpu_memory_context_smoke(monkeypatch):
    """Test that async_gpu_memory_context works with mocked CUDA availability."""
    mod = importlib.import_module("src.utils.storage")

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def empty_cache():
            return None

    monkeypatch.setitem(importlib.sys.modules, "torch", SimpleNamespace(cuda=_Cuda))
    importlib.reload(mod)

    async with mod.async_gpu_memory_context():
        pass


@pytest.mark.unit
def test_model_context_cleanup(monkeypatch):
    """Test that model_context and sync_model_context properly clean up resources."""
    mod = importlib.import_module("src.utils.storage")

    class _M:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    async def _factory():  # async factory
        return _M()

    async def _run():
        async with mod.model_context(_factory, cleanup_method="close") as m:
            assert isinstance(m, _M)

    import asyncio

    asyncio.run(_run())

    # Sync variant
    with mod.sync_model_context(lambda: _M(), cleanup_method="close") as m:
        assert isinstance(m, _M)


@pytest.mark.unit
def test_cuda_error_context_paths():
    """Test that cuda_error_context handles exceptions and returns default values."""
    mod = importlib.import_module("src.utils.storage")

    with mod.cuda_error_context("probe", reraise=False, default_return=7) as ctx:
        # Raising inside context should be swallowed (reraise=False)
        raise RuntimeError("CUDA BOOM")
    # After context, result should be default_return
    assert ctx.get("result") == 7


@pytest.mark.unit
def test_cuda_error_context_reraise_true():
    """Test that cuda_error_context re-raises exceptions when reraise=True."""
    mod = importlib.import_module("src.utils.storage")
    with pytest.raises(RuntimeError), mod.cuda_error_context("probe", reraise=True):
        raise RuntimeError("fail")


@pytest.mark.unit
def test_cuda_error_context_system_and_import_errors():
    """Covers non-RuntimeError branches for cuda_error_context."""
    mod = importlib.import_module("src.utils.storage")

    with mod.cuda_error_context("probe", reraise=False, default_return=9) as ctx:
        raise OSError("disk")
    assert ctx.get("result") == 9
    assert "error" in ctx

    with mod.cuda_error_context("probe", reraise=False, default_return=11) as ctx:
        raise ImportError("missing")
    assert ctx.get("result") == 11
    assert "error" in ctx
