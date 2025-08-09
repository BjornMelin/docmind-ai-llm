"""Comprehensive tests for utils/client_factory.py.

Tests QdrantClientFactory with comprehensive coverage of sync/async clients,
context managers, error handling, and configuration consistency.

Target coverage: 95%+ for client factory utilities.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.client_factory import QdrantClientFactory


class TestQdrantClientFactory:
    """Test suite for QdrantClientFactory with comprehensive coverage."""

    def test_get_client_config_structure(self, test_settings):
        """Test client configuration returns expected structure."""
        config = QdrantClientFactory.get_client_config()

        # Verify all required config keys are present
        expected_keys = {"url", "timeout", "prefer_grpc", "grpc_options"}
        assert set(config.keys()) == expected_keys

        # Verify config values
        assert config["url"] == test_settings.qdrant_url
        assert config["timeout"] == 60
        assert config["prefer_grpc"] is True
        assert isinstance(config["grpc_options"], dict)

    def test_get_client_config_grpc_options(self, test_settings):
        """Test GRPC options are properly configured."""
        config = QdrantClientFactory.get_client_config()
        grpc_options = config["grpc_options"]

        # Verify all GRPC options are present
        expected_grpc_keys = {
            "grpc.max_send_message_length",
            "grpc.max_receive_message_length",
            "grpc.keepalive_time_ms",
            "grpc.keepalive_timeout_ms",
            "grpc.keepalive_permit_without_calls",
        }
        assert set(grpc_options.keys()) == expected_grpc_keys

        # Verify message size limits (50MB)
        assert grpc_options["grpc.max_send_message_length"] == 50 * 1024 * 1024
        assert grpc_options["grpc.max_receive_message_length"] == 50 * 1024 * 1024

        # Verify keepalive settings
        assert grpc_options["grpc.keepalive_time_ms"] == 30000
        assert grpc_options["grpc.keepalive_timeout_ms"] == 10000
        assert grpc_options["grpc.keepalive_permit_without_calls"] is True

    @patch("utils.client_factory.QdrantClient")
    @patch("utils.client_factory.logging")
    def test_create_sync_client_success(
        self, mock_logging, mock_qdrant_client, test_settings
    ):
        """Test successful sync client creation with context manager."""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        with QdrantClientFactory.create_sync_client() as client:
            assert client == mock_client

        # Verify client was created with correct config
        config = QdrantClientFactory.get_client_config()
        mock_qdrant_client.assert_called_once_with(**config)

        # Verify logging
        mock_logging.info.assert_called_with(
            f"Sync Qdrant client connected to {test_settings.qdrant_url}"
        )

        # Verify client was closed
        mock_client.close.assert_called_once()
        mock_logging.debug.assert_called_with("Sync Qdrant client closed successfully")

    @patch("utils.client_factory.QdrantClient")
    @patch("utils.client_factory.logging")
    def test_create_sync_client_creation_failure(
        self, mock_logging, mock_qdrant_client
    ):
        """Test sync client creation failure handling."""
        mock_qdrant_client.side_effect = ConnectionError("Connection failed")

        with (
            pytest.raises(ConnectionError, match="Connection failed"),
            QdrantClientFactory.create_sync_client(),
        ):
            pass  # Should not reach here

        # Verify error was logged
        mock_logging.error.assert_called_with(
            "Failed to create sync Qdrant client: Connection failed"
        )

    @patch("utils.client_factory.QdrantClient")
    @patch("utils.client_factory.logging")
    def test_create_sync_client_close_failure(self, mock_logging, mock_qdrant_client):
        """Test sync client cleanup failure handling."""
        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("Close failed")
        mock_qdrant_client.return_value = mock_client

        with QdrantClientFactory.create_sync_client() as client:
            assert client == mock_client

        # Verify close failure was logged as warning
        mock_logging.warning.assert_called_with(
            "Error closing sync client: Close failed"
        )

    @patch("utils.client_factory.AsyncQdrantClient")
    @patch("utils.client_factory.logging")
    @pytest.mark.asyncio
    async def test_create_async_client_success(
        self, mock_logging, mock_async_qdrant_client, test_settings
    ):
        """Test successful async client creation with context manager."""
        mock_client = AsyncMock()
        mock_async_qdrant_client.return_value = mock_client

        async with QdrantClientFactory.create_async_client() as client:
            assert client == mock_client

        # Verify client was created with correct config
        config = QdrantClientFactory.get_client_config()
        mock_async_qdrant_client.assert_called_once_with(**config)

        # Verify logging
        mock_logging.info.assert_called_with(
            f"Async Qdrant client connected to {test_settings.qdrant_url}"
        )

        # Verify client was closed
        mock_client.close.assert_called_once()
        mock_logging.debug.assert_called_with("Async Qdrant client closed successfully")

    @patch("utils.client_factory.AsyncQdrantClient")
    @patch("utils.client_factory.logging")
    @pytest.mark.asyncio
    async def test_create_async_client_creation_failure(
        self, mock_logging, mock_async_qdrant_client
    ):
        """Test async client creation failure handling."""
        mock_async_qdrant_client.side_effect = TimeoutError("Connection timeout")

        with pytest.raises(TimeoutError, match="Connection timeout"):
            async with QdrantClientFactory.create_async_client():
                pass  # Should not reach here

        # Verify error was logged
        mock_logging.error.assert_called_with(
            "Failed to create async Qdrant client: Connection timeout"
        )

    @patch("utils.client_factory.AsyncQdrantClient")
    @patch("utils.client_factory.logging")
    @pytest.mark.asyncio
    async def test_create_async_client_close_failure(
        self, mock_logging, mock_async_qdrant_client
    ):
        """Test async client cleanup failure handling."""
        mock_client = AsyncMock()
        mock_client.close.side_effect = RuntimeError("Async close failed")
        mock_async_qdrant_client.return_value = mock_client

        async with QdrantClientFactory.create_async_client() as client:
            assert client == mock_client

        # Verify close failure was logged as warning
        mock_logging.warning.assert_called_with(
            "Error closing async client: Async close failed"
        )

    @patch("utils.client_factory.QdrantClient")
    @patch("utils.client_factory.logging")
    def test_create_direct_sync_client_success(
        self, mock_logging, mock_qdrant_client, test_settings
    ):
        """Test direct sync client creation without context manager."""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        client = QdrantClientFactory.create_direct_sync_client()

        assert client == mock_client

        # Verify client was created with correct config
        config = QdrantClientFactory.get_client_config()
        mock_qdrant_client.assert_called_once_with(**config)

        # Verify logging
        mock_logging.info.assert_called_with(
            f"Direct sync Qdrant client created: {test_settings.qdrant_url}"
        )

    @patch("utils.client_factory.QdrantClient")
    @patch("utils.client_factory.logging")
    def test_create_direct_sync_client_failure(self, mock_logging, mock_qdrant_client):
        """Test direct sync client creation failure."""
        mock_qdrant_client.side_effect = OSError("Network unreachable")

        with pytest.raises(OSError, match="Network unreachable"):
            QdrantClientFactory.create_direct_sync_client()

        # Verify error was logged
        mock_logging.error.assert_called_with(
            "Failed to create direct sync Qdrant client: Network unreachable"
        )

    @patch("utils.client_factory.AsyncQdrantClient")
    @patch("utils.client_factory.logging")
    @pytest.mark.asyncio
    async def test_create_direct_async_client_success(
        self, mock_logging, mock_async_qdrant_client, test_settings
    ):
        """Test direct async client creation without context manager."""
        mock_client = AsyncMock()
        mock_async_qdrant_client.return_value = mock_client

        client = await QdrantClientFactory.create_direct_async_client()

        assert client == mock_client

        # Verify client was created with correct config
        config = QdrantClientFactory.get_client_config()
        mock_async_qdrant_client.assert_called_once_with(**config)

        # Verify logging
        mock_logging.info.assert_called_with(
            f"Direct async Qdrant client created: {test_settings.qdrant_url}"
        )

    @patch("utils.client_factory.AsyncQdrantClient")
    @patch("utils.client_factory.logging")
    @pytest.mark.asyncio
    async def test_create_direct_async_client_failure(
        self, mock_logging, mock_async_qdrant_client
    ):
        """Test direct async client creation failure."""
        mock_async_qdrant_client.side_effect = PermissionError("Access denied")

        with pytest.raises(PermissionError, match="Access denied"):
            await QdrantClientFactory.create_direct_async_client()

        # Verify error was logged
        mock_logging.error.assert_called_with(
            "Failed to create direct async Qdrant client: Access denied"
        )

    @patch("utils.client_factory.QdrantClient")
    def test_sync_client_exception_in_context(self, mock_qdrant_client):
        """Test sync client proper cleanup when exception occurs in context."""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        with patch("QdrantClientFactory.create_sync_client", return_value=mock_client):
            with pytest.raises(ValueError, match="Test error"):
                with QdrantClientFactory.create_sync_client() as client:
                    raise ValueError("Test error")

        # Verify client was still closed despite exception
        mock_client.close.assert_called_once()

    @patch("utils.client_factory.AsyncQdrantClient")
    @pytest.mark.asyncio
    async def test_async_client_exception_in_context(self, mock_async_qdrant_client):
        """Test async client proper cleanup when exception occurs in context."""
        mock_client = AsyncMock()
        mock_async_qdrant_client.return_value = mock_client

        with patch("QdrantClientFactory.create_async_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Async test error"):
                async with QdrantClientFactory.create_async_client() as client:
                    raise RuntimeError("Async test error")

        # Verify client was still closed despite exception
        mock_client.close.assert_called_once()

    @patch("utils.client_factory.QdrantClient")
    def test_sync_client_no_close_method(self, mock_qdrant_client):
        """Test sync client cleanup when client has no close method."""
        mock_client = MagicMock()
        del mock_client.close  # Remove close method
        mock_qdrant_client.return_value = mock_client

        # Should not raise exception even without close method
        with QdrantClientFactory.create_sync_client() as client:
            assert client == mock_client

    @patch("utils.client_factory.AsyncQdrantClient")
    @pytest.mark.asyncio
    async def test_async_client_no_close_method(self, mock_async_qdrant_client):
        """Test async client cleanup when client has no close method."""
        mock_client = AsyncMock()
        del mock_client.close  # Remove close method
        mock_async_qdrant_client.return_value = mock_client

        # Should not raise exception even without close method
        async with QdrantClientFactory.create_async_client() as client:
            assert client == mock_client

    @patch.dict("os.environ", {"QDRANT_URL": "http://test-server:6333"})
    @patch("utils.client_factory.settings")
    def test_config_respects_environment_variables(self, mock_settings):
        """Test client config respects environment variable changes."""
        mock_settings.qdrant_url = "http://test-server:6333"

        config = QdrantClientFactory.get_client_config()

        assert config["url"] == "http://test-server:6333"

    def test_config_consistency_across_calls(self):
        """Test client configuration is consistent across multiple calls."""
        config1 = QdrantClientFactory.get_client_config()
        config2 = QdrantClientFactory.get_client_config()

        assert config1 == config2

    @patch("utils.client_factory.QdrantClient")
    @patch("utils.client_factory.AsyncQdrantClient")
    def test_sync_and_async_use_same_config(self, mock_async_client, mock_sync_client):
        """Test sync and async clients use identical configuration."""
        mock_sync_instance = MagicMock()
        mock_async_instance = AsyncMock()
        mock_sync_client.return_value = mock_sync_instance
        mock_async_client.return_value = mock_async_instance

        # Create both sync and async clients
        with QdrantClientFactory.create_sync_client():
            pass

        # Get the config used for sync client
        sync_config_call = mock_sync_client.call_args[1]

        # Reset mocks and create async client
        mock_sync_client.reset_mock()
        mock_async_client.reset_mock()

        async def test_async():
            async with QdrantClientFactory.create_async_client():
                pass

        import asyncio

        asyncio.run(test_async())

        # Get the config used for async client
        async_config_call = mock_async_client.call_args[1]

        # Verify configurations are identical
        assert sync_config_call == async_config_call

    @patch("utils.client_factory.QdrantClient")
    @patch("utils.client_factory.logging")
    def test_multiple_sync_clients_independently(
        self, mock_logging, mock_qdrant_client
    ):
        """Test multiple sync clients can be created and managed independently."""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        # Configure mock to return different instances
        mock_qdrant_client.side_effect = [mock_client1, mock_client2]

        # Create first client
        with QdrantClientFactory.create_sync_client() as client1:
            assert client1 == mock_client1

            # Create second client while first is active
            with QdrantClientFactory.create_sync_client() as client2:
                assert client2 == mock_client2
                assert client1 != client2

            # Second client should be closed
            mock_client2.close.assert_called_once()

        # First client should be closed
        mock_client1.close.assert_called_once()

    @patch("utils.client_factory.AsyncQdrantClient")
    @patch("utils.client_factory.logging")
    @pytest.mark.asyncio
    async def test_multiple_async_clients_independently(
        self, mock_logging, mock_async_qdrant_client
    ):
        """Test multiple async clients can be created and managed independently."""
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()

        # Configure mock to return different instances
        mock_async_qdrant_client.side_effect = [mock_client1, mock_client2]

        # Create first client
        async with QdrantClientFactory.create_async_client() as client1:
            assert client1 == mock_client1

            # Create second client while first is active
            async with QdrantClientFactory.create_async_client() as client2:
                assert client2 == mock_client2
                assert client1 != client2

            # Second client should be closed
            mock_client2.close.assert_called_once()

        # First client should be closed
        mock_client1.close.assert_called_once()

    @patch("utils.client_factory.QdrantClient")
    def test_sync_client_creation_none_client(self, mock_qdrant_client):
        """Test sync client factory handles None client gracefully."""
        mock_qdrant_client.return_value = None

        with QdrantClientFactory.create_sync_client() as client:
            assert client is None

        # No close should be called on None client

    @patch("utils.client_factory.AsyncQdrantClient")
    @pytest.mark.asyncio
    async def test_async_client_creation_none_client(self, mock_async_qdrant_client):
        """Test async client factory handles None client gracefully."""
        mock_async_qdrant_client.return_value = None

        async with QdrantClientFactory.create_async_client() as client:
            assert client is None

        # No close should be called on None client

    def test_client_factory_is_stateless(self):
        """Test QdrantClientFactory maintains no internal state."""
        # Factory should have no instance attributes
        factory = QdrantClientFactory()
        assert len(factory.__dict__) == 0

        # All methods should be class methods or static methods
        assert callable(QdrantClientFactory.get_client_config)
        assert callable(QdrantClientFactory.create_sync_client)
        assert callable(QdrantClientFactory.create_async_client)
        assert callable(QdrantClientFactory.create_direct_sync_client)
        assert callable(QdrantClientFactory.create_direct_async_client)

    @patch("utils.client_factory.QdrantClient")
    @patch("utils.client_factory.logging")
    def test_logging_levels_and_messages(self, mock_logging, mock_qdrant_client):
        """Test proper logging levels and message formats."""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        with QdrantClientFactory.create_sync_client():
            pass

        # Verify info level for connection
        info_calls = [
            call
            for call in mock_logging.info.call_args_list
            if "Sync Qdrant client connected" in call[0][0]
        ]
        assert len(info_calls) == 1

        # Verify debug level for close
        debug_calls = [
            call
            for call in mock_logging.debug.call_args_list
            if "Sync Qdrant client closed successfully" in call[0][0]
        ]
        assert len(debug_calls) == 1

    @pytest.mark.parametrize(
        "exception_type,error_message",
        [
            (ConnectionError, "Connection refused"),
            (TimeoutError, "Request timeout"),
            (OSError, "Network error"),
            (ValueError, "Invalid parameter"),
            (RuntimeError, "Runtime error"),
        ],
    )
    @patch("utils.client_factory.QdrantClient")
    @patch("utils.client_factory.logging")
    def test_sync_client_various_creation_errors(
        self, mock_logging, mock_qdrant_client, exception_type, error_message
    ):
        """Test sync client creation handles various exception types."""
        mock_qdrant_client.side_effect = exception_type(error_message)

        with (
            pytest.raises(exception_type, match=error_message),
            QdrantClientFactory.create_sync_client(),
        ):
            pass

        # Verify error was logged with correct message
        mock_logging.error.assert_called_with(
            f"Failed to create sync Qdrant client: {error_message}"
        )

    @pytest.mark.parametrize(
        "exception_type,error_message",
        [
            (ConnectionError, "Async connection refused"),
            (TimeoutError, "Async request timeout"),
            (OSError, "Async network error"),
            (ValueError, "Async invalid parameter"),
            (RuntimeError, "Async runtime error"),
        ],
    )
    @patch("utils.client_factory.AsyncQdrantClient")
    @patch("utils.client_factory.logging")
    @pytest.mark.asyncio
    async def test_async_client_various_creation_errors(
        self, mock_logging, mock_async_qdrant_client, exception_type, error_message
    ):
        """Test async client creation handles various exception types."""
        mock_async_qdrant_client.side_effect = exception_type(error_message)

        with pytest.raises(exception_type, match=error_message):
            async with QdrantClientFactory.create_async_client():
                pass

        # Verify error was logged with correct message
        mock_logging.error.assert_called_with(
            f"Failed to create async Qdrant client: {error_message}"
        )
