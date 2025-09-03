"""Unit tests to raise coverage for src.utils.core.

Covers hardware detection, startup configuration validation, RRF verification,
GPU management contexts, and the async timer decorator. All external effects
are boundary-mocked.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.utils import core as core_mod


def test_detect_hardware_cpu_only(monkeypatch):
    """Test hardware detection returns CPU-only configuration."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    monkeypatch.setattr(core_mod, "torch", mock_torch, raising=False)

    info = core_mod.detect_hardware()
    assert info["cuda_available"] is False
    assert info["gpu_name"] == "Unknown"


def test_detect_hardware_gpu(monkeypatch):
    """Test hardware detection returns GPU configuration when available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "RTX-TEST"
    mock_props = SimpleNamespace(total_memory=16 * 1024**3)
    mock_torch.cuda.get_device_properties.return_value = mock_props
    # settings.monitoring.bytes_to_gb_divisor default is 1_073_741_824 (1024**3)
    monkeypatch.setattr(core_mod, "torch", mock_torch, raising=False)

    info = core_mod.detect_hardware()
    assert info["cuda_available"] is True
    assert info["gpu_name"] == "RTX-TEST"
    assert info["vram_total_gb"] in (16, 16.0)


def test_validate_startup_configuration_success(monkeypatch):
    """Test startup configuration validation succeeds with valid Qdrant connection."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = SimpleNamespace(collections=[])
    monkeypatch.setattr(
        core_mod.qdrant_client, "QdrantClient", MagicMock(return_value=mock_client)
    )

    settings = MagicMock()
    settings.database.qdrant_url = "http://localhost:6333"
    settings.enable_gpu_acceleration = False

    result = core_mod.validate_startup_configuration(settings)
    assert result["valid"] is True
    assert any("Qdrant connection successful" in m for m in result["info"])


def test_validate_startup_configuration_connection_error(monkeypatch):
    """Test startup config validation raises RuntimeError on connection failure."""

    # Raise ConnectionError on init to hit error path
    def _raise(*_a, **_kw):
        raise ConnectionError("boom")

    monkeypatch.setattr(core_mod.qdrant_client, "QdrantClient", _raise)

    settings = MagicMock()
    settings.database.qdrant_url = "http://localhost:6333"
    settings.enable_gpu_acceleration = False

    with pytest.raises(RuntimeError):
        core_mod.validate_startup_configuration(settings)


def test_verify_rrf_configuration_in_range():
    """Test RRF configuration verification with valid parameters."""
    cfg = SimpleNamespace(
        retrieval=SimpleNamespace(rrf_alpha=50, rrf_k_constant=60, strategy="hybrid")
    )
    out = core_mod.verify_rrf_configuration(cfg)
    assert out["alpha_in_range"] is True
    assert out["weights_correct"] is True


def test_managed_gpu_operation(monkeypatch):
    """Test managed GPU operation context manager clears cache on exit."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    monkeypatch.setattr(core_mod, "torch", mock_torch, raising=False)

    async def _run():
        async with core_mod.managed_gpu_operation():
            pass

    import asyncio

    asyncio.run(_run())
    assert mock_torch.cuda.empty_cache.called


def test_async_timer_decorator(monkeypatch):
    """Test async timer decorator executes function correctly."""
    calls = {"count": 0}

    @core_mod.async_timer
    async def _fn():
        calls["count"] += 1
        return 42

    import asyncio

    assert asyncio.run(_fn()) == 42
    assert calls["count"] == 1
