"""Qdrant client factory for DocMind AI with consistent configuration.

This module provides a centralized factory for creating Qdrant clients with
optimal configuration, connection pooling, and proper resource cleanup.
Eliminates code duplication between sync and async client setups.

Features:
- Consistent client configuration across the application
- Context managers for proper resource cleanup
- Connection timeout and GRPC optimization
- Error handling with detailed logging
- Singleton pattern for configuration consistency

Example:
    Using the client factory::

        from utils.client_factory import QdrantClientFactory

        # Sync usage
        with QdrantClientFactory.create_sync_client() as client:
            collections = client.get_collections()

        # Async usage
        async with QdrantClientFactory.create_async_client() as client:
            collections = await client.get_collections()

Attributes:
    settings (AppSettings): Global application settings for client configuration.
"""

from contextlib import asynccontextmanager, contextmanager

from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient

from models.core import settings


class QdrantClientFactory:
    """Factory for creating Qdrant clients with consistent configuration.

    Provides centralized client creation with optimized settings, proper
    resource management, and consistent configuration across sync/async usage.
    Eliminates code duplication and ensures all clients use best practices.
    """

    @staticmethod
    def get_client_config() -> dict:
        """Get common client configuration for all Qdrant clients.

        Returns optimized configuration settings for both sync and async clients:
        - Connection timeout: 60 seconds for large operations
        - GRPC preference: Enabled for better performance
        - Message limits: 50MB for large document processing

        Returns:
            Dictionary with client configuration parameters.
        """
        return {
            "url": settings.qdrant_url,
            "timeout": 60,
            "prefer_grpc": True,
            "grpc_options": {
                "grpc.max_send_message_length": 50 * 1024 * 1024,  # 50MB
                "grpc.max_receive_message_length": 50 * 1024 * 1024,
                "grpc.keepalive_time_ms": 30000,  # 30 seconds
                "grpc.keepalive_timeout_ms": 10000,  # 10 seconds
                "grpc.keepalive_permit_without_calls": True,
            },
        }

    @classmethod
    @contextmanager
    def create_sync_client(cls):
        """Create synchronous Qdrant client with context manager.

        Provides proper resource management for sync operations with automatic
        cleanup on context exit. Includes error handling and detailed logging.

        Yields:
            QdrantClient: Configured synchronous client ready for use.

        Raises:
            Exception: If client creation fails, logs error and re-raises.

        Example:
            >>> with QdrantClientFactory.create_sync_client() as client:
            ...     collections = client.get_collections()
            ...     print(f"Found {len(collections.collections)} collections")
        """
        config = cls.get_client_config()
        client = None
        try:
            client = QdrantClient(**config)
            logger.info(f"Sync Qdrant client connected to {settings.qdrant_url}")
            yield client
        except Exception as e:
            logger.error(f"Failed to create sync Qdrant client: {e}")
            raise
        finally:
            if client:
                try:
                    client.close()
                    logger.debug("Sync Qdrant client closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing sync client: {e}")

    @classmethod
    @asynccontextmanager
    async def create_async_client(cls):
        """Create async Qdrant client with context manager.

        Provides proper resource management for async operations with automatic
        cleanup on context exit. Optimized for concurrent operations and includes
        comprehensive error handling.

        Yields:
            AsyncQdrantClient: Configured async client ready for use.

        Raises:
            Exception: If client creation fails, logs error and re-raises.

        Example:
            >>> async with QdrantClientFactory.create_async_client() as client:
            ...     collections = await client.get_collections()
            ...     print(f"Found {len(collections.collections)} collections")
        """
        config = cls.get_client_config()
        client = None
        try:
            client = AsyncQdrantClient(**config)
            logger.info(f"Async Qdrant client connected to {settings.qdrant_url}")
            yield client
        except Exception as e:
            logger.error(f"Failed to create async Qdrant client: {e}")
            raise
        finally:
            if client:
                try:
                    await client.close()
                    logger.debug("Async Qdrant client closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing async client: {e}")

    @classmethod
    def create_direct_sync_client(cls) -> QdrantClient:
        """Create sync client without context manager for legacy compatibility.

        Provides direct client creation for cases where context manager usage
        is not feasible. Client cleanup must be handled manually by caller.

        Returns:
            QdrantClient: Configured synchronous client.

        Note:
            Caller is responsible for calling client.close() after usage.
            Prefer using create_sync_client() context manager when possible.

        Example:
            >>> client = QdrantClientFactory.create_direct_sync_client()
            >>> try:
            ...     collections = client.get_collections()
            ... finally:
            ...     client.close()
        """
        config = cls.get_client_config()
        try:
            client = QdrantClient(**config)
            logger.info(f"Direct sync Qdrant client created: {settings.qdrant_url}")
            return client
        except Exception as e:
            logger.error(f"Failed to create direct sync Qdrant client: {e}")
            raise

    @classmethod
    async def create_direct_async_client(cls) -> AsyncQdrantClient:
        """Create async client without context manager for legacy compatibility.

        Provides direct client creation for cases where context manager usage
        is not feasible. Client cleanup must be handled manually by caller.

        Returns:
            AsyncQdrantClient: Configured asynchronous client.

        Note:
            Caller is responsible for calling await client.close() after usage.
            Prefer using create_async_client() context manager when possible.

        Example:
            >>> client = await QdrantClientFactory.create_direct_async_client()
            >>> try:
            ...     collections = await client.get_collections()
            ... finally:
            ...     await client.close()
        """
        config = cls.get_client_config()
        try:
            client = AsyncQdrantClient(**config)
            logger.info(f"Direct async Qdrant client created: {settings.qdrant_url}")
            return client
        except Exception as e:
            logger.error(f"Failed to create direct async Qdrant client: {e}")
            raise


# Library-first approach: Let AsyncQdrantClient handle its own connection management
# Removed 100+ lines of over-engineered connection pooling for single-user local app
# Modern libraries like httpx (used by qdrant-client) handle connections internally
