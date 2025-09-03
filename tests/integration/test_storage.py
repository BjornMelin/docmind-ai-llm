"""Storage integration tests (consolidated).

Consolidates storage utility tests into a single file with deterministic
checks for client configuration and context managers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.utils.storage import create_async_client, create_sync_client, get_client_config


@pytest.mark.integration
class TestStorageUtils:
    """Test storage utility functions."""

    def test_get_client_config_has_expected_keys(self):
        """Test client config has all expected keys."""
        cfg = get_client_config()
        assert set(cfg.keys()) >= {"url", "timeout", "prefer_grpc"}

    def test_create_sync_client_context(self):
        """Test synchronous client creation context manager."""
        with patch("src.utils.storage.QdrantClient") as mock_client:
            mock_instance = mock_client.return_value
            with create_sync_client() as client:
                assert client == mock_instance
            mock_instance.close.assert_called()

    @pytest.mark.asyncio
    async def test_create_async_client_context(self):
        """Test asynchronous client creation context manager."""
        with patch("src.utils.storage.AsyncQdrantClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.close = AsyncMock()
            async with create_async_client() as client:
                assert client == mock_instance
            mock_instance.close.assert_awaited()
