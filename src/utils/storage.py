"""Storage and resource management utilities for DocMind AI.

This module consolidates database operations and resource management utilities:

Database Operations:
- Qdrant client creation and management with proper cleanup
- Hybrid vector collection setup (dense + sparse)
- Basic vector store configuration and testing

Resource Management:
- GPU memory context managers and cleanup utilities
- Model lifecycle management with automatic cleanup
- CUDA error handling and safe operation wrappers
- Comprehensive GPU information and monitoring

Key features:
- Hybrid Qdrant collection setup for dense + sparse vectors
- GPU memory context managers with automatic cleanup
- Model context managers for lifecycle management
- Safe CUDA operations with comprehensive error handling
- Qdrant client context managers for resource cleanup
"""

import asyncio
import gc
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager, suppress
from typing import Any

try:  # Optional torch; CPU-only environments must not fail at import
    import torch  # type: ignore
except (ImportError, OSError):  # pragma: no cover - optional dependency
    torch = None  # type: ignore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from src.config import settings
from src.utils.qdrant_exceptions import (
    QDRANT_SCHEMA_EXCEPTIONS,
    QDRANT_TRANSPORT_EXCEPTIONS,
)
from src.utils.qdrant_utils import get_collection_params

# Preferred sparse models (logging/telemetry; selection handled by fastembed if present)
PREFERRED_SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
FALLBACK_SPARSE_MODEL = "Qdrant/bm25"


def ensure_sparse_idf_modifier(client: QdrantClient, collection_name: str) -> None:
    """Ensure the sparse vector config uses the IDF modifier.

    If the collection exists and the sparse vector modifier is not IDF, this
    updates the collection configuration in-place to set it to IDF. This is a
    no-op when the modifier is already IDF or when the sparse vector head is
    absent.

    Args:
        client (QdrantClient): Qdrant client instance.
        collection_name (str): Name of the collection to check or update.

    Returns:
        None
    """
    try:
        info = client.get_collection(collection_name)
        cfg = getattr(info.config.params, "sparse_vectors", None)
        if isinstance(cfg, dict) and "text-sparse" in cfg:
            cur = cfg["text-sparse"]
            # cur is SparseVectorParams; getattr safe for older clients
            cur_mod = getattr(cur, "modifier", None)
            if cur_mod is None or cur_mod != qmodels.Modifier.IDF:
                logger.info(
                    "Updating sparse modifier to IDF for collection '%s'",
                    collection_name,
                )
                client.update_collection(
                    collection_name=collection_name,
                    sparse_vectors_config={
                        "text-sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                            modifier=qmodels.Modifier.IDF,
                        )
                    },
                )
    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as e:  # pragma: no cover - defensive path
        logger.warning("ensure_sparse_idf_modifier skipped: %s", e)


def _safe_collection_exists(client: QdrantClient, name: str) -> bool:
    """Safely check whether a Qdrant collection exists.

    Args:
        client (QdrantClient): Configured Qdrant client.
        name (str): Target collection name.

    Returns:
        bool: True if the collection exists; False if it does not or if an
        error occurs while checking.
    """
    try:
        return bool(client.collection_exists(name))
    except (
        *QDRANT_TRANSPORT_EXCEPTIONS,
        ValueError,
    ) as exc:  # pragma: no cover - defensive
        logger.warning("collection_exists check failed: %s", exc)
        return False


def _create_hybrid_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    """Create a hybrid collection with named dense and sparse vectors.

    The dense head is created as ``text-dense`` using cosine distance; the
    sparse head is created as ``text-sparse`` with the IDF modifier.

    Args:
        client (QdrantClient): Configured Qdrant client.
        name (str): Collection name to create.
        dense_dim (int): Dense vector dimensionality.

    Returns:
        None

    Raises:
        ResponseHandlingException: If the server returns a malformed response.
        UnexpectedResponse: If the client cannot parse the response.
        ConnectionError: On connection failures.
        TimeoutError: On request timeout.
        OSError: On local system I/O errors.
        ValueError: On invalid parameters.
    """
    client.create_collection(
        collection_name=name,
        vectors_config={
            "text-dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "text-sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=qmodels.Modifier.IDF,
            ),
        },
    )
    logger.info("Created hybrid collection '%s' (dense=%d)", name, dense_dim)


def _compute_hybrid_patches(
    dense_cfg: dict | None, sparse_cfg: dict | None, dense_dim: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Compute schema patches for named vectors and the IDF modifier.

    Args:
        dense_cfg (dict | None): Existing dense vectors configuration (may be
            ``params.vectors`` or ``params.vectors_config`` depending on client
            version). Use ``None`` when unavailable.
        sparse_cfg (dict | None): Existing sparse vectors configuration (may be
            ``params.sparse_vectors`` or ``params.sparse_vectors_config``).
        dense_dim (int): Desired dimensionality for the ``text-dense`` head.

    Returns:
        tuple[dict[str, Any] | None, dict[str, Any] | None]: A pair of optional
        patch dictionaries:
        - First element is the ``vectors_config`` to add/update ``text-dense`` or
          ``None`` when no change is required.
        - Second element is the ``sparse_vectors_config`` to add/update
          ``text-sparse`` with the IDF modifier, or ``None`` when no change is
          required.
    """
    patch_vectors: dict[str, Any] | None = None
    if not isinstance(dense_cfg, dict) or "text-dense" not in dense_cfg:
        patch_vectors = {
            "text-dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
        }

    patch_sparse: dict[str, Any] | None = None
    needs_sparse_patch = True
    if isinstance(sparse_cfg, dict) and "text-sparse" in sparse_cfg:
        # Present: only patch if modifier not IDF
        try:
            cur = sparse_cfg["text-sparse"]
            cur_mod = getattr(cur, "modifier", None)
            needs_sparse_patch = cur_mod != qmodels.Modifier.IDF
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
            needs_sparse_patch = True

    if needs_sparse_patch:
        patch_sparse = {
            "text-sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=qmodels.Modifier.IDF,
            )
        }

    return patch_vectors, patch_sparse


def ensure_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = settings.embedding.dimension,
) -> None:
    """Ensure a hybrid collection schema exists (idempotent).

    This function guarantees the presence of the named dense (``text-dense``)
    and sparse (``text-sparse`` with IDF) vector heads. It creates the
    collection when missing, or patches the schema in-place when needed. It
    logs a warning if the existing ``text-dense`` size differs from ``dense_dim``
    but does not force a dimensionality change.

    Args:
        client (QdrantClient): Configured Qdrant client.
        collection_name (str): Target collection name.
        dense_dim (int): Expected dimension of the dense embedding vector.

    Returns:
        None
    """
    if not _safe_collection_exists(client, collection_name):
        try:
            _create_hybrid_collection(client, collection_name, dense_dim)
        except (
            *QDRANT_TRANSPORT_EXCEPTIONS,
            ValueError,
        ) as exc:  # pragma: no cover - defensive
            logger.warning(
                "create_collection failed for '%s': %s", collection_name, exc
            )
        return

    try:
        params = get_collection_params(client, collection_name)

        dense_cfg = getattr(params, "vectors", None) or getattr(
            params, "vectors_config", None
        )
        sparse_cfg = getattr(params, "sparse_vectors", None) or getattr(
            params, "sparse_vectors_config", None
        )

        # Warn on size mismatch if present
        try:
            if isinstance(dense_cfg, dict) and "text-dense" in dense_cfg:
                cur_size = int(getattr(dense_cfg["text-dense"], "size", dense_dim))
                if cur_size != dense_dim:
                    logger.warning(
                        "Collection '%s' text-dense size mismatch (have=%d, want=%d)",
                        collection_name,
                        cur_size,
                        dense_dim,
                    )
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
        ) as exc:  # pragma: no cover - defensive
            logger.debug("dense size verify skipped: %s", exc)

        patch_vecs, patch_sprs = _compute_hybrid_patches(
            dense_cfg, sparse_cfg, dense_dim
        )
        if patch_vecs or patch_sprs:
            logger.info("Updating collection '%s' schema", collection_name)
            client.update_collection(
                collection_name=collection_name,
                vectors_config=patch_vecs,
                sparse_vectors_config=patch_sprs,
            )
        else:
            logger.debug("Hybrid schema already present for '%s'", collection_name)
    except QDRANT_SCHEMA_EXCEPTIONS as exc:  # pragma: no cover - defensive
        logger.warning("ensure_hybrid_collection skipped: %s", exc)


# =============================================================================
# Database Operations (formerly from database.py)
# =============================================================================


def get_client_config() -> dict[str, Any]:
    """Get standard Qdrant client configuration.

    Returns:
        Dictionary with client configuration
    """
    return {
        "url": settings.database.qdrant_url,
        "timeout": settings.database.qdrant_timeout,
        "prefer_grpc": True,
    }


@contextmanager
def create_sync_client() -> Generator[QdrantClient, None, None]:
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
async def create_async_client() -> AsyncGenerator[AsyncQdrantClient, None]:
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
    dense_embedding_size: int = settings.embedding.dimension,
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
                    index=SparseIndexParams(on_disk=False),
                    modifier=qmodels.Modifier.IDF,
                )
            },
        )
        logger.success("Created hybrid collection: %s", collection_name)
        logger.info(
            "Sparse model preference: %s (fallback %s)",
            PREFERRED_SPARSE_MODEL,
            FALLBACK_SPARSE_MODEL,
        )
        if settings.retrieval.named_vectors_multi_head_enabled:
            logger.info("Named-vectors multi-head feature flag is enabled (no-op)")

    # Create sync client for QdrantVectorStore compatibility
    config = get_client_config()
    sync_client = QdrantClient(**config)

    try:
        return QdrantVectorStore(
            client=sync_client,
            collection_name=collection_name,
            enable_hybrid=True,
            batch_size=settings.monitoring.default_batch_size,
        )
    except ImportError as e:  # fastembed optional for hybrid sparse
        logger.warning(
            "Hybrid vector store requires FastEmbed; falling back to dense-only: %s",
            e,
        )
        return QdrantVectorStore(
            client=sync_client,
            collection_name=collection_name,
            enable_hybrid=False,
            batch_size=settings.monitoring.default_batch_size,
        )


def setup_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_embedding_size: int = settings.embedding.dimension,
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
                    index=SparseIndexParams(on_disk=False),
                    modifier=qmodels.Modifier.IDF,
                )
            },
        )
        logger.success("Created hybrid collection: %s", collection_name)
        logger.info(
            "Sparse model preference: %s (fallback %s)",
            PREFERRED_SPARSE_MODEL,
            FALLBACK_SPARSE_MODEL,
        )
        if settings.retrieval.named_vectors_multi_head_enabled:
            logger.info("Named-vectors multi-head feature flag is enabled (no-op)")

    try:
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            enable_hybrid=True,
            batch_size=settings.monitoring.default_batch_size,
        )
    except ImportError as e:  # fastembed optional for hybrid sparse
        logger.warning(
            "Hybrid vector store requires FastEmbed; falling back to dense-only: %s",
            e,
        )
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            enable_hybrid=False,
            batch_size=settings.monitoring.default_batch_size,
        )


def create_vector_store(
    collection_name: str,
    _dense_embedding_size: int = settings.embedding.dimension,
    enable_hybrid: bool = True,
) -> QdrantVectorStore:
    """Create QdrantVectorStore with standard configuration.

    Args:
        collection_name: Name of the collection
        _dense_embedding_size: Size of dense embeddings (unused for now)
        enable_hybrid: Enable hybrid search capabilities

    Returns:
        Configured QdrantVectorStore
    """
    client = QdrantClient(**get_client_config())
    # Ensure named vectors schema exists when hybrid is enabled (idempotent)
    if enable_hybrid:
        try:
            ensure_hybrid_collection(
                client,
                collection_name,
                dense_dim=_dense_embedding_size,
            )
        except QDRANT_TRANSPORT_EXCEPTIONS:  # pragma: no cover - defensive ensure
            logger.warning(
                "ensure_hybrid_collection failed; proceeding with store creation"
            )

    try:
        store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            enable_hybrid=enable_hybrid,
            batch_size=settings.monitoring.default_batch_size,
        )
        # Ensure sparse IDF modifier on existing collections where supported
        with suppress(Exception):
            ensure_sparse_idf_modifier(client, collection_name)
        return store
    except ImportError as e:  # fastembed optional for hybrid sparse
        logger.warning(
            "Hybrid vector store requires FastEmbed; falling back to dense-only: %s",
            e,
        )
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            enable_hybrid=False,
            batch_size=settings.monitoring.default_batch_size,
        )


def persist_image_metadata(
    client: QdrantClient,
    collection_name: str,
    point_id: str | int,
    metadata: dict[str, Any],
) -> bool:
    """Persist additional image metadata (e.g., phash) to Qdrant payload.

    Args:
        client: Qdrant client instance used to update payload.
        collection_name: Target collection name.
        point_id: Identifier of the point whose payload is updated.
        metadata: Key-value pairs to merge into the point payload.

    Returns:
        bool: True on success; False when the update fails due to client or
        system errors.
    """
    try:
        client.set_payload(
            collection_name=collection_name,
            points=[point_id],
            payload=metadata,
        )
        return True
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
        logger.warning("persist_image_metadata failed for %s: %s", point_id, exc)
        return False


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
                "url": settings.database.qdrant_url,
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
            "url": settings.database.qdrant_url,
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


# =============================================================================
# Resource Management Operations (formerly from resource_management.py)
# =============================================================================


@contextmanager
def gpu_memory_context() -> Generator[None, None, None]:
    """Context manager for GPU memory cleanup.

    Automatically synchronizes and clears GPU cache on exit, regardless
    of whether operations succeeded or failed. Essential for preventing
    VRAM leaks in ML applications.

    Example:
        with gpu_memory_context():
            # GPU operations here
            model.forward(inputs)
            # Automatic cleanup on exit

    Yields:
        None
    """
    try:
        yield
    finally:
        # Always cleanup GPU resources
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("GPU cleanup failed during context exit: %s", e)
        finally:
            # Always run garbage collection
            gc.collect()


@asynccontextmanager
async def async_gpu_memory_context() -> AsyncGenerator[None, None]:
    """Async context manager for GPU memory cleanup.

    Async version of gpu_memory_context() for use with async operations.
    Provides the same automatic cleanup guarantees.

    Example:
        async with async_gpu_memory_context():
            # Async GPU operations here
            embeddings = await model.encode_async(texts)
            # Automatic cleanup on exit

    Yields:
        None
    """
    try:
        yield
    finally:
        # Always cleanup GPU resources
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("GPU cleanup failed during async context exit: %s", e)
        finally:
            # Always run garbage collection
            gc.collect()


@asynccontextmanager
async def model_context(
    model_factory: Callable[..., Any],
    cleanup_method: str | None = None,
    **kwargs: Any,
) -> AsyncGenerator[Any, None]:
    """Generic model context manager with automatic cleanup.

    Manages model lifecycle including creation, usage, and cleanup.
    Supports both async and sync model factories and cleanup methods.

    Args:
        model_factory: Function/method to create the model
        cleanup_method: Name of cleanup method on model (e.g., 'close', 'cleanup')
        **kwargs: Arguments to pass to model_factory

    Example:
        async with model_context(
            create_embedding_model, cleanup_method='cleanup'
        ) as model:
            embeddings = await model.encode(texts)
            # Automatic cleanup on exit

    Yields:
        The created model instance
    """
    model = None
    try:
        # Create model (handle both sync and async factories)
        if asyncio.iscoroutinefunction(model_factory):
            model = await model_factory(**kwargs)
        else:
            model = model_factory(**kwargs)

        yield model

    finally:
        # Always attempt cleanup
        if model is not None:
            await _cleanup_model(model, cleanup_method)


@contextmanager
def sync_model_context(
    model_factory: Callable[..., Any],
    cleanup_method: str | None = None,
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """Synchronous model context manager with automatic cleanup.

    Sync version of model_context() for non-async workflows.

    Args:
        model_factory: Function/method to create the model
        cleanup_method: Name of cleanup method on model (e.g., 'close', 'cleanup')
        **kwargs: Arguments to pass to model_factory

    Example:
        with sync_model_context(create_model, cleanup_method='close') as model:
            result = model.process(data)
            # Automatic cleanup on exit

    Yields:
        The created model instance
    """
    model = None
    try:
        model = model_factory(**kwargs)
        yield model
    finally:
        # Always attempt cleanup
        if model is not None:
            _sync_cleanup_model(model, cleanup_method)


@contextmanager
def cuda_error_context(
    operation_name: str = "CUDA operation",
    reraise: bool = True,
    default_return: Any = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for robust CUDA error handling.

    Provides comprehensive error handling for CUDA operations with
    detailed logging and optional fallback behavior.

    Args:
        operation_name: Name of the operation for logging
        reraise: Whether to reraise exceptions after logging
        default_return: Default value to return on error (if not reraising)

    Example:
        with cuda_error_context("VRAM check", reraise=False, default_return=0.0) as ctx:
            vram = (
                torch.cuda.memory_allocated() / 1024**3
                if (torch is not None and torch.cuda.is_available())
                else 0.0
            )
            ctx['result'] = vram

        vram = ctx.get('result', 0.0)

    Yields:
        Dictionary to store operation results
    """
    result_dict = {}
    try:
        yield result_dict
    except RuntimeError as e:
        if "CUDA" in str(e).upper():
            logger.warning("%s failed with CUDA error: %s", operation_name, e)
        else:
            logger.warning("%s failed with runtime error: %s", operation_name, e)

        if reraise:
            raise
        result_dict["result"] = default_return
        result_dict["error"] = str(e)

    except (OSError, AttributeError) as e:
        logger.warning("%s failed with system error: %s", operation_name, e)

        if reraise:
            raise
        result_dict["result"] = default_return
        result_dict["error"] = str(e)

    except (ImportError, ModuleNotFoundError) as e:
        logger.error("%s failed with import error: %s", operation_name, e)

        if reraise:
            raise
        result_dict["result"] = default_return
        result_dict["error"] = str(e)


def safe_cuda_operation(
    operation: Callable[[], Any],
    operation_name: str = "CUDA operation",
    default_return: Any = None,
    log_errors: bool = True,
) -> Any:
    """Execute CUDA operation with comprehensive error handling.

    Wrapper function for single CUDA operations that need error handling.

    Args:
        operation: Function to execute (should take no arguments)
        operation_name: Name for logging purposes
        default_return: Value to return on error
        log_errors: Whether to log errors

    Returns:
        Result of operation or default_return on error

    Example:
        vram = safe_cuda_operation(
            lambda: torch.cuda.memory_allocated() / 1024**3,
            "VRAM check",
            default_return=0.0
        )
    """
    try:
        return operation()
    except RuntimeError as e:
        if log_errors:
            if "CUDA" in str(e).upper():
                logger.warning("%s failed with CUDA error: %s", operation_name, e)
            else:
                logger.warning("%s failed with runtime error: %s", operation_name, e)
        return default_return
    except (OSError, AttributeError) as e:
        if log_errors:
            logger.warning("%s failed with system error: %s", operation_name, e)
        return default_return
    except (ImportError, ModuleNotFoundError) as e:
        if log_errors:
            logger.error("%s failed with import error: %s", operation_name, e)
        return default_return


def get_safe_vram_usage() -> float:
    """Get current VRAM usage with comprehensive error handling.

    Provides a safe way to check VRAM usage that won't crash on
    CUDA errors or missing hardware.

    Returns:
        VRAM usage in GB (0.0 if CUDA unavailable or error)
    """
    return safe_cuda_operation(
        lambda: (
            torch.cuda.memory_allocated() / settings.monitoring.bytes_to_gb_divisor
            if (torch is not None and torch.cuda.is_available())
            else 0.0
        ),
        "VRAM usage check",
        default_return=0.0,
    )


def get_safe_gpu_info() -> dict[str, Any]:
    """Get GPU information with comprehensive error handling.

    Returns:
        Dictionary with GPU info (safe defaults on error)
    """
    info = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "Unknown",
        "compute_capability": None,
        "total_memory_gb": 0.0,
        "allocated_memory_gb": 0.0,
    }

    try:
        t = torch
        if t is None:
            return info

        info["cuda_available"] = bool(t.cuda.is_available())

        if info["cuda_available"]:
            info["device_count"] = safe_cuda_operation(
                t.cuda.device_count, "device count", 0
            )

            if info["device_count"] > 0:
                info["device_name"] = safe_cuda_operation(
                    lambda: t.cuda.get_device_name(0), "device name", "Unknown"
                )

                # Get device properties safely
                props = safe_cuda_operation(
                    lambda: t.cuda.get_device_properties(0),
                    "device properties",
                    None,
                )

                if props:
                    info["compute_capability"] = f"{props.major}.{props.minor}"
                    info["total_memory_gb"] = (
                        props.total_memory / settings.monitoring.bytes_to_gb_divisor
                    )

                info["allocated_memory_gb"] = safe_cuda_operation(
                    lambda: (
                        t.cuda.memory_allocated(0)
                        / settings.monitoring.bytes_to_gb_divisor
                    ),
                    "allocated memory",
                    0.0,
                )

    except (RuntimeError, OSError) as e:
        logger.warning("Failed to get GPU info: %s", e)

    return info


# =============================================================================
# Internal Helper Functions
# =============================================================================


async def _cleanup_model(model: Any, cleanup_method: str | None) -> None:
    """Internal async cleanup helper for models.

    Args:
        model: Model instance to clean up.
        cleanup_method: Optional name of the cleanup method (e.g., ``"close"``,
            ``"cleanup"``). If not provided or method is absent, no-op.

    Returns:
        None
    """
    if not cleanup_method:
        return

    try:
        if hasattr(model, cleanup_method):
            cleanup_func = getattr(model, cleanup_method)
            if asyncio.iscoroutinefunction(cleanup_func):
                await cleanup_func()
            else:
                cleanup_func()
        else:
            logger.debug("Model has no cleanup method: %s", cleanup_method)
    except (AttributeError, TypeError) as e:
        logger.warning("Model cleanup failed: %s", e)


def _sync_cleanup_model(model: Any, cleanup_method: str | None) -> None:
    """Internal sync cleanup helper for models.

    Args:
        model: Model instance to clean up.
        cleanup_method: Optional name of the cleanup method (e.g., ``"close"``,
            ``"cleanup"``). If not provided or method is absent, no-op.

    Returns:
        None
    """
    if not cleanup_method:
        return

    try:
        if hasattr(model, cleanup_method):
            cleanup_func = getattr(model, cleanup_method)
            cleanup_func()
        else:
            logger.debug("Model has no cleanup method: %s", cleanup_method)
    except (AttributeError, TypeError) as e:
        logger.warning("Model cleanup failed: %s", e)
