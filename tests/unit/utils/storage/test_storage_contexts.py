"""Tests for storage GPU/model/cuda context managers and error context."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.unit
def test_gpu_memory_context_smoke(monkeypatch):
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
    mod = importlib.import_module("src.utils.storage")

    with mod.cuda_error_context("probe", reraise=False, default_return=7) as ctx:
        # Raising inside context should be swallowed (reraise=False)
        raise RuntimeError("CUDA BOOM")
    # After context, result should be default_return
    assert ctx.get("result") == 7


@pytest.mark.unit
def test_cuda_error_context_reraise_true():
    mod = importlib.import_module("src.utils.storage")
    with pytest.raises(RuntimeError), mod.cuda_error_context("probe", reraise=True):
        raise RuntimeError("fail")
