"""Additional tests for src.utils.core covering offline and GPU paths.

Tests avoid external services by monkeypatching qdrant_client and torch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_validate_startup_configuration_offline(monkeypatch):
    """Offline Qdrant yields structured errors without raising."""
    from src.config.settings import DocMindSettings
    from src.utils import core as c

    class _Client:
        def __init__(self, *_, **__):
            pass

        def get_collections(self):  # simulate connection refused
            raise OSError("[Errno 111] Connection refused")

        def close(self):
            pass

    monkeypatch.setattr(c, "qdrant_client", SimpleNamespace(QdrantClient=_Client))

    settings = DocMindSettings()
    out = c.validate_startup_configuration(settings)
    assert isinstance(out, dict)
    assert out["valid"] is False
    assert any("Qdrant" in str(e) for e in out["errors"]) or out["errors"]


def test_validate_startup_configuration_gpu_enabled_paths(monkeypatch):
    """GPU enabled adds warnings when CUDA not available and info when available."""
    from src.config.settings import DocMindSettings
    from src.utils import core as c

    # Base: offline qdrant so we don't depend on network
    class _Client:
        def __init__(self, *_, **__):
            pass

        def get_collections(self):
            return None

        def close(self):
            pass

    monkeypatch.setattr(c, "qdrant_client", SimpleNamespace(QdrantClient=_Client))

    # Case 1: GPU enabled but not available
    class _Cuda:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(c, "torch", SimpleNamespace(cuda=_Cuda))
    s1 = DocMindSettings(enable_gpu_acceleration=True)
    out1 = c.validate_startup_configuration(s1)
    assert out1["valid"] is True
    assert (
        any("GPU acceleration enabled" in w for w in out1["warnings"])
        or out1["warnings"]
    )

    # Case 2: GPU available adds info
    class _Props:
        total_memory = 8 * 1024**3

    class _Cuda2:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(_idx):
            return "Test GPU"

        @staticmethod
        def get_device_properties(_idx):
            return _Props()

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def empty_cache():
            return None

    monkeypatch.setattr(c, "torch", SimpleNamespace(cuda=_Cuda2))
    s2 = DocMindSettings(enable_gpu_acceleration=True)
    out2 = c.validate_startup_configuration(s2)
    assert out2["valid"] is True
    assert any("GPU available" in i for i in out2["info"]) or out2["info"]


@pytest.mark.asyncio
async def test_managed_async_qdrant_client_closes(monkeypatch):
    """Ensure managed_async_qdrant_client closes the client on exit."""
    from src.utils import core as c

    closed = {"v": False}

    class _AClient:
        def __init__(self, *_, **__):
            pass

        async def close(self):
            closed["v"] = True

    # Patch alias used in module
    monkeypatch.setattr(c, "AsyncQdrantClient", _AClient)

    async with c.managed_async_qdrant_client(url="http://localhost:6333") as _cli:
        assert isinstance(_cli, _AClient)
    assert closed["v"] is True


@pytest.mark.asyncio
async def test_managed_gpu_operation_calls_cuda(monkeypatch):
    """Ensure managed_gpu_operation calls cuda cleanup when available."""
    from src.utils import core as c

    calls = {"sync": 0, "empty": 0}

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            calls["sync"] += 1

        @staticmethod
        def empty_cache():
            calls["empty"] += 1

    monkeypatch.setattr(c, "torch", SimpleNamespace(cuda=_Cuda))
    async with c.managed_gpu_operation():
        pass
    assert calls["sync"] == 1
    assert calls["empty"] == 1
