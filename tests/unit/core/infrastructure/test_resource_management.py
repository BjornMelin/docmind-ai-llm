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
import torch


@pytest.mark.asyncio
async def test_managed_async_qdrant_client_cleanup():
    """Test that managed_async_qdrant_client properly cleans up resources."""
    from src.utils.core import managed_async_qdrant_client

    with patch("src.utils.core.AsyncQdrantClient") as mock_client_class:
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

    with patch("src.utils.core.AsyncQdrantClient") as mock_client_class:
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_memory_cleanup_integration():
    """Integration test for GPU memory cleanup (requires CUDA)."""
    import torch

    # Get initial GPU memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()

        # Create some tensors to use GPU memory
        large_tensor = torch.randn(1000, 1000, device="cuda")
        after_allocation = torch.cuda.memory_allocated()

        # Verify memory increased
        assert after_allocation > initial_memory

        # Cleanup
        del large_tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        final_memory = torch.cuda.memory_allocated()

        # Memory should be cleaned up (may not be exactly equal due to fragmentation)
        assert final_memory <= after_allocation


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
