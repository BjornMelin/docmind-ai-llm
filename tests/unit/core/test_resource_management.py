"""Unit tests for critical resource management fixes (P0).

Tests to verify that the memory leaks and resource management issues
identified in the code review have been properly fixed.

Test Coverage:
    - Async context managers for Qdrant clients
    - GPU memory cleanup after operations
    - Basic resource management validation

Note: Many functions previously tested here were removed during utils/ cleanup.
Only tests for functions that still exist in src.utils are included.
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_managed_async_qdrant_client_cleanup():
    """Test that managed_async_qdrant_client properly cleans up resources."""
    from src.utils.core import managed_async_qdrant_client

    with patch("qdrant_client.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Test normal operation
        async with managed_async_qdrant_client("http://localhost:6333") as client:
            assert client == mock_client

        # Verify client was closed
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_managed_async_qdrant_client_cleanup_on_exception():
    """Test that client is cleaned up even when exceptions occur."""
    from src.utils.core import managed_async_qdrant_client

    with patch("qdrant_client.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Test exception handling
        with pytest.raises(ValueError, match="Test exception"):
            async with managed_async_qdrant_client("http://localhost:6333") as _:
                raise ValueError("Test exception")

        # Verify client was still closed despite exception
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_managed_gpu_operation_cleanup():
    """Test that GPU operations are properly cleaned up."""
    from src.utils.core import managed_gpu_operation

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.synchronize") as mock_sync,
        patch("torch.cuda.empty_cache") as mock_empty_cache,
        patch("gc.collect") as mock_gc,
    ):
        # Test normal operation
        async with managed_gpu_operation():
            pass  # Simulate GPU operation

        # Verify cleanup was called
        mock_sync.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_gc.assert_called_once()


@pytest.mark.asyncio
async def test_managed_gpu_operation_cleanup_on_exception():
    """Test that GPU cleanup occurs even when operations fail."""
    from src.utils.core import managed_gpu_operation

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.synchronize") as mock_sync,
        patch("torch.cuda.empty_cache") as mock_empty_cache,
        patch("gc.collect") as mock_gc,
    ):
        # Test exception handling
        with pytest.raises(RuntimeError):
            async with managed_gpu_operation():
                raise RuntimeError("GPU operation failed")

        # Verify cleanup still occurred
        mock_sync.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_gc.assert_called_once()
