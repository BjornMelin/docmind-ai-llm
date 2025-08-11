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

import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient

from models import AppSettings

settings = AppSettings()


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


class QdrantConnectionPool:
    """Connection pool for Qdrant clients."""

    def __init__(self, pool_size: int = 5):
        """Initialize a connection pool for Qdrant clients.

        Manages an asynchronous queue of Qdrant clients to optimize connection reuse
        and improve performance for concurrent database operations.

        Args:
            pool_size (int, optional): Maximum number of concurrent connections.
                Defaults to 5. Allows multiple simultaneous clients to distribute load.

        Attributes:
            pool_size (int): Maximum number of clients in the pool.
            _pool (asyncio.Queue): Asynchronous queue managing client connections.
            _initialized (bool): Flag indicating whether the pool has been initialized.
            _lock (asyncio.Lock): Synchronization lock for thread-safe pool operations.
            _stats (dict): Tracks connection and request statistics including:
                - total_connections: Total number of connections created
                - active_connections: Currently active connections
                - total_requests: Total connection requests
                - failed_requests: Requests that could not be fulfilled
        """
        self.pool_size = pool_size
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._initialized = False
        self._lock = asyncio.Lock()
        self._stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_requests": 0,
            "failed_requests": 0,
        }

    async def initialize(self):
        """Initialize connection pool."""
        async with self._lock:
            if self._initialized:
                return

            config = QdrantClientFactory.get_client_config()
            for _ in range(self.pool_size):
                client = AsyncQdrantClient(**config)
                await self._pool.put(client)
                self._stats["total_connections"] += 1

            self._initialized = True
            logger.info(f"Initialized Qdrant pool with {self.pool_size} connections")

    @asynccontextmanager
    async def get_client(self):
        """Get client from pool."""
        if not self._initialized:
            await self.initialize()

        self._stats["total_requests"] += 1
        self._stats["active_connections"] += 1

        try:
            # Wait for available client with timeout
            client = await asyncio.wait_for(self._pool.get(), timeout=30.0)
            yield client
        except TimeoutError:
            self._stats["failed_requests"] += 1
            logger.error("Timeout waiting for Qdrant client from pool")
            raise
        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error(f"Error getting Qdrant client: {e}")
            raise
        finally:
            # Return to pool
            if "client" in locals():
                await self._pool.put(client)
            self._stats["active_connections"] -= 1

    async def close_all(self):
        """Close all connections in pool."""
        async with self._lock:
            while not self._pool.empty():
                try:
                    client = self._pool.get_nowait()
                    await client.close()
                except Exception as e:
                    logger.warning(f"Error closing client: {e}")

            self._initialized = False
            self._stats["total_connections"] = 0
            logger.info("Closed all connections in Qdrant pool")

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return {
            **self._stats,
            "pool_size": self.pool_size,
            "initialized": self._initialized,
            "queue_size": self._pool.qsize(),
        }


# Global pool instance
qdrant_pool = QdrantConnectionPool(pool_size=5)


class QdrantBatchOperations:
    """Batch operations for Qdrant with async optimizations."""

    @staticmethod
    async def batch_upsert_vectors(
        client: AsyncQdrantClient,
        collection_name: str,
        vectors: list,
        payloads: list,
        batch_size: int = 100,
    ):
        """Batch upsert vectors to Qdrant."""
        from qdrant_client.models import PointStruct

        start_time = time.perf_counter()

        batches = []
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i : i + batch_size]
            batch_payloads = payloads[i : i + batch_size]

            points = [
                PointStruct(id=i + j, vector=vec, payload=payload)
                for j, (vec, payload) in enumerate(
                    zip(batch_vectors, batch_payloads, strict=False)
                )
            ]

            batches.append(
                client.upsert(collection_name=collection_name, points=points)
            )

        # Execute all batches in parallel
        await asyncio.gather(*batches)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Upserted {len(vectors)} vectors in {len(batches)} "
            f"batches ({elapsed:.2f}s)"
        )

    @staticmethod
    async def health_check(client: AsyncQdrantClient) -> bool:
        """Check if Qdrant client is healthy."""
        try:
            await client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False


# Enhanced QdrantClientFactory with pooling support
class EnhancedQdrantClientFactory(QdrantClientFactory):
    """Enhanced client factory with connection pooling."""

    @staticmethod
    @asynccontextmanager
    async def create_pooled_client():
        """Create client from connection pool."""
        async with qdrant_pool.get_client() as client:
            yield client

    @staticmethod
    async def get_pool_stats() -> dict[str, Any]:
        """Get connection pool statistics."""
        return qdrant_pool.get_stats()

    @staticmethod
    async def close_pool():
        """Close the connection pool."""
        await qdrant_pool.close_all()
