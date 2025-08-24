"""Database utilities for DocMind AI - Qdrant operations simplified.

This module provides essential Qdrant database operations with simplified
functions replacing factory patterns. Focuses on core functionality needed
for hybrid vector search operations.

Key features:
- Hybrid Qdrant collection setup for dense + sparse vectors
- Basic client creation with optimal settings
- Simple context managers for resource cleanup
- Essential vector store configuration
"""

from contextlib import asynccontextmanager, contextmanager
from typing import Any

from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from src.config.settings import settings


def get_client_config() -> dict[str, Any]:
    """Get standard Qdrant client configuration.

    Returns:
        Dictionary with client configuration
    """
    return {
        "url": settings.qdrant_url,
        "timeout": settings.default_qdrant_timeout,
        "prefer_grpc": True,
    }


@contextmanager
def create_sync_client():
    """Create sync Qdrant client with proper cleanup.

    Yields:
        QdrantClient: Configured sync client
    """
    client = None
    try:
        config = get_client_config()
        client = QdrantClient(**config)
        logger.debug("Created sync Qdrant client: %s", config["url"])
        yield client
    except (
        ResponseHandlingException,
        UnexpectedResponse,
        ConnectionError,
        TimeoutError,
    ) as e:
        logger.error("Failed to create sync Qdrant client: %s", e)
        raise
    finally:
        if client is not None:
            try:
                client.close()
            except (ResponseHandlingException, ConnectionError) as e:
                logger.warning("Error closing sync client: %s", e)


@asynccontextmanager
async def create_async_client():
    """Create async Qdrant client with proper cleanup.

    Yields:
        AsyncQdrantClient: Configured async client
    """
    client = None
    try:
        config = get_client_config()
        client = AsyncQdrantClient(**config)
        logger.debug("Created async Qdrant client: %s", config["url"])
        yield client
    except (
        ResponseHandlingException,
        UnexpectedResponse,
        ConnectionError,
        TimeoutError,
    ) as e:
        logger.error("Failed to create async Qdrant client: %s", e)
        raise
    finally:
        if client is not None:
            try:
                await client.close()
            except (ResponseHandlingException, ConnectionError) as e:
                logger.warning("Error closing async client: %s", e)


async def setup_hybrid_collection_async(
    client: AsyncQdrantClient,
    collection_name: str,
    dense_embedding_size: int = settings.bge_m3_embedding_dim,
    recreate: bool = False,
) -> QdrantVectorStore:
    """Setup Qdrant collection for hybrid search (async).

    Args:
        client: AsyncQdrantClient instance
        collection_name: Name of collection to create/configure
        dense_embedding_size: Size of dense embeddings
        recreate: Whether to recreate if exists

    Returns:
        QdrantVectorStore configured for hybrid search
    """
    logger.info("Setting up hybrid collection: %s", collection_name)

    if recreate and await client.collection_exists(collection_name):
        await client.delete_collection(collection_name)
        logger.info("Deleted existing collection: %s", collection_name)

    if not await client.collection_exists(collection_name):
        # Create collection with both dense and sparse vectors
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": VectorParams(
                    size=dense_embedding_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.success("Created hybrid collection: %s", collection_name)

    # Create sync client for QdrantVectorStore compatibility
    sync_client = QdrantClient(url=settings.qdrant_url)

    return QdrantVectorStore(
        client=sync_client,
        collection_name=collection_name,
        enable_hybrid=True,
        batch_size=settings.default_batch_size,
    )


def setup_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_embedding_size: int = settings.bge_m3_embedding_dim,
    recreate: bool = False,
) -> QdrantVectorStore:
    """Setup Qdrant collection for hybrid search (sync).

    Args:
        client: QdrantClient instance
        collection_name: Name of collection to create/configure
        dense_embedding_size: Size of dense embeddings
        recreate: Whether to recreate if exists

    Returns:
        QdrantVectorStore configured for hybrid search
    """
    logger.info("Setting up hybrid collection: %s", collection_name)

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection: %s", collection_name)

    if not client.collection_exists(collection_name):
        # Create collection with both dense and sparse vectors
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": VectorParams(
                    size=dense_embedding_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.success("Created hybrid collection: %s", collection_name)

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        enable_hybrid=True,
        batch_size=settings.default_batch_size,
    )


def create_vector_store(
    collection_name: str,
    dense_embedding_size: int = settings.bge_m3_embedding_dim,
    enable_hybrid: bool = True,
) -> QdrantVectorStore:
    """Create QdrantVectorStore with standard configuration.

    Args:
        collection_name: Name of the collection
        dense_embedding_size: Size of dense embeddings
        enable_hybrid: Enable hybrid search capabilities

    Returns:
        Configured QdrantVectorStore
    """
    client = QdrantClient(url=settings.qdrant_url)

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        enable_hybrid=enable_hybrid,
        batch_size=settings.default_batch_size,
    )


def get_collection_info(collection_name: str) -> dict[str, Any]:
    """Get information about a Qdrant collection.

    Args:
        collection_name: Name of the collection

    Returns:
        Dictionary with collection information
    """
    try:
        with create_sync_client() as client:
            if not client.collection_exists(collection_name):
                return {"exists": False, "error": "Collection not found"}

            info = client.get_collection(collection_name)
            return {
                "exists": True,
                "points_count": info.points_count,
                "vectors_config": info.config.params.vectors,
                "sparse_vectors_config": info.config.params.sparse_vectors,
                "status": info.status,
            }
    except (
        ResponseHandlingException,
        UnexpectedResponse,
        ConnectionError,
        TimeoutError,
    ) as e:
        logger.error("Failed to get collection info for %s: %s", collection_name, e)
        return {"exists": False, "error": str(e)}


def test_connection() -> dict[str, Any]:
    """Test connection to Qdrant database.

    Returns:
        Dictionary with connection test results
    """
    try:
        with create_sync_client() as client:
            collections = client.get_collections()
            return {
                "connected": True,
                "url": settings.qdrant_url,
                "collections_count": len(collections.collections),
                "collections": [c.name for c in collections.collections],
            }
    except (
        ResponseHandlingException,
        UnexpectedResponse,
        ConnectionError,
        TimeoutError,
    ) as e:
        logger.error("Qdrant connection test failed: %s", e)
        return {
            "connected": False,
            "url": settings.qdrant_url,
            "error": str(e),
        }


def clear_collection(collection_name: str) -> bool:
    """Clear all points from a collection.

    Args:
        collection_name: Name of collection to clear

    Returns:
        True if successful, False otherwise
    """
    try:
        with create_sync_client() as client:
            if not client.collection_exists(collection_name):
                logger.warning("Collection %s does not exist", collection_name)
                return False

            # Delete and recreate collection (fastest way to clear)
            info = client.get_collection(collection_name)
            client.delete_collection(collection_name)

            # Recreate with same configuration
            client.create_collection(
                collection_name=collection_name,
                vectors_config=info.config.params.vectors,
                sparse_vectors_config=info.config.params.sparse_vectors,
            )

            logger.success("Cleared collection: %s", collection_name)
            return True

    except (
        ResponseHandlingException,
        UnexpectedResponse,
        ConnectionError,
        TimeoutError,
    ) as e:
        logger.error("Failed to clear collection %s: %s", collection_name, e)
        return False
