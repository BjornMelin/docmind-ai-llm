
"""Runtime-oriented tests for src.utils.core covering device resolution and clients."""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest


@pytest.mark.unit
def test_resolve_device_parses_indices(monkeypatch):
    import src.utils.core as core

    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(current_device=lambda: 1)
    )
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)
    monkeypatch.setattr(core, "select_device", lambda prefer: "cuda")

    assert core.resolve_device("cuda:5") == ("cuda:5", 5)
    assert core.resolve_device("cuda") == ("cuda:1", 1)


@pytest.mark.unit
def test_resolve_device_falls_back_on_errors(monkeypatch):
    import src.utils.core as core

    monkeypatch.setattr(core, "TORCH", types.SimpleNamespace(cuda=None))

    def _raise(_prefer: str) -> str:  # pragma: no cover - intentional
        raise RuntimeError("select failed")

    monkeypatch.setattr(core, "select_device", _raise)
    assert core.resolve_device("auto") == ("cpu", None)


@pytest.mark.unit
def test_detect_hardware_reports_cuda(monkeypatch):
    import src.utils.core as core

    monkeypatch.setattr(core, "is_cuda_available", lambda: True)

    class _Cuda:
        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA-Unit-Test"

    dummy_torch = types.SimpleNamespace(cuda=_Cuda)
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)
    monkeypatch.setattr(core, "get_vram_gb", lambda device_index=0: 24.0)

    info = core.detect_hardware()
    assert info["cuda_available"] is True
    assert info["gpu_name"] == "NVIDIA-Unit-Test"
    assert info["vram_total_gb"] == 24.0


@pytest.mark.unit
def test_validate_startup_configuration_handles_qdrant(monkeypatch):
    import src.utils.core as core

    class _Client:
        def __init__(self, url: str) -> None:
            self.url = url
            self.closed = False

        def get_collections(self):  # pragma: no cover - simple stub
            return {"collections": []}

        def close(self) -> None:
            self.closed = True

    created = {}

    def _factory(url: str):
        client = _Client(url)
        created["client"] = client
        return client

    monkeypatch.setattr("qdrant_client.QdrantClient", _factory)

    app_settings = SimpleNamespace(
        database=SimpleNamespace(qdrant_url="http://localhost:6333"),
        enable_gpu_acceleration=False,
    )

    result = core.validate_startup_configuration(app_settings)
    assert result["valid"] is True
    assert created["client"].closed is True

@pytest.mark.unit
def test_validate_startup_configuration_records_errors(monkeypatch):
    import src.utils.core as core

    def _factory(url: str):  # pragma: no cover - intentional failure
        raise ConnectionError("offline")

    monkeypatch.setattr("qdrant_client.QdrantClient", _factory)

    app_settings = SimpleNamespace(
        database=SimpleNamespace(qdrant_url="http://localhost:6333"),
        enable_gpu_acceleration=False,
    )

    result = core.validate_startup_configuration(app_settings)
    assert result["valid"] is False
    assert any("offline" in err for err in result["errors"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_managed_async_qdrant_client_closes(monkeypatch):
    import src.utils.core as core

    class _AsyncClient:
        def __init__(self, url: str) -> None:
            self.url = url
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    created = {}
    monkeypatch.setattr(
        "qdrant_client.AsyncQdrantClient",
        lambda url: created.setdefault("client", _AsyncClient(url)),
    )

    async with core.managed_async_qdrant_client("http://localhost:6333") as client:
        assert client.url.endswith(":6333")
    assert created["client"].closed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_timer_logs(monkeypatch):
    import src.utils.core as core

    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(core.logger, "info", lambda msg, *args: calls.append((msg, args[0])))

    @core.async_timer
    async def _do_work():
        return "done"

    result = await _do_work()
    assert result == "done"
    assert calls and calls[0][0] == "%s completed in %.2fs"
