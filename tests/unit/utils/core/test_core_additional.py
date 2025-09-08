"""Additional unit tests for src.utils.core to raise coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.utils import core as core_mod


@pytest.mark.unit
@pytest.mark.asyncio
async def test_managed_async_qdrant_client_closes() -> None:
    """Async client context yields client and awaits close on exit."""
    fake_client = AsyncMock()
    with patch.object(core_mod, "AsyncQdrantClient", return_value=fake_client):
        async with core_mod.managed_async_qdrant_client("http://qdrant:6333") as cli:
            assert cli is fake_client
        fake_client.close.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_managed_gpu_operation_calls_empty_cache_when_available() -> None:
    """GPU context calls synchronize and empty_cache when CUDA is available."""
    with (
        patch("src.utils.core.torch.cuda.is_available", return_value=True),
        patch("src.utils.core.torch.cuda.empty_cache") as empty_cache,
        patch("src.utils.core.torch.cuda.synchronize") as sync,
    ):
        async with core_mod.managed_gpu_operation():
            pass
        empty_cache.assert_called_once()
        sync.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_timer_decorator_raises_and_logs() -> None:
    """async_timer lets exceptions through while still logging timing."""
    calls = {"count": 0}

    @core_mod.async_timer
    async def boom() -> None:
        calls["count"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await boom()
    assert calls["count"] == 1
