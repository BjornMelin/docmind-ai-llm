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
from dataclasses import dataclass
from typing import Any, Literal

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
from src.utils.log_safety import build_pii_log_entry, safe_url_for_log
from src.utils.qdrant_exceptions import (
    QDRANT_SCHEMA_EXCEPTIONS,
)
from src.utils.qdrant_utils import get_collection_params

# Preferred sparse models (logging/telemetry; selection handled by fastembed if present)
PREFERRED_SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
FALLBACK_SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "text-dense"
SPARSE_VECTOR_NAME = "text-sparse"


@dataclass(frozen=True, slots=True)
class CollectionCompatibilityResult:
    """Explicit named-vector collection compatibility result."""

    compatible: bool
    action: Literal["unchanged", "created", "recreated", "blocked", "error"]
    reason: str
    point_count: int | None = None


class QdrantCollectionIncompatibleError(Exception):
    """Raised before indexing into an incompatible Qdrant collection."""

    def __init__(
        self,
        collection_name: str,
        result: CollectionCompatibilityResult,
    ) -> None:
        """Store the incompatible collection name and compatibility result.

        Args:
            collection_name: Qdrant collection that cannot be indexed safely.
            result: Compatibility evidence describing the blocking condition.
        """
        self.collection_name = collection_name
        self.result = result
        super().__init__(
            f"Qdrant collection {collection_name!r} is incompatible: {result.reason}"
        )


def _create_hybrid_collection(
    client: QdrantClient,
    name: str,
    dense_dim: int,
) -> None:
    """Create a collection with canonical named dense and sparse vectors."""
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=qmodels.Modifier.IDF,
            ),
        },
    )
    logger.info("Created hybrid collection {} (dense={})", name, dense_dim)


def check_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = settings.embedding.dimension,
) -> CollectionCompatibilityResult:
    """Inspect named-vector compatibility without mutating Qdrant.

    Args:
        client: Qdrant client used to inspect collection state.
        collection_name: Collection whose named-vector schema is inspected.
        dense_dim: Required dimension for the canonical dense vector.

    Returns:
        CollectionCompatibilityResult: Compatibility evidence and required action.
    """
    try:
        if not client.collection_exists(collection_name):
            return CollectionCompatibilityResult(
                compatible=False,
                action="blocked",
                reason="collection_missing",
            )
        info = client.get_collection(collection_name)
        params = get_collection_params(client, collection_name)
        dense_cfg = getattr(params, "vectors", None) or getattr(
            params, "vectors_config", None
        )
        sparse_cfg = getattr(params, "sparse_vectors", None) or getattr(
            params, "sparse_vectors_config", None
        )
        raw_count = getattr(info, "points_count", None)
        point_count = raw_count if isinstance(raw_count, int) else None
        incompatibility = _hybrid_collection_incompatibility(
            dense_cfg,
            sparse_cfg,
            dense_dim=dense_dim,
        )
        if incompatibility is not None:
            return CollectionCompatibilityResult(
                False, "blocked", incompatibility, point_count
            )
        return CollectionCompatibilityResult(
            True, "unchanged", "compatible", point_count
        )
    except QDRANT_SCHEMA_EXCEPTIONS as exc:
        redaction = build_pii_log_entry(
            str(exc), key_id="storage.check_hybrid_collection"
        )
        logger.warning(
            "Qdrant compatibility check failed (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        return CollectionCompatibilityResult(
            False, "error", "compatibility_check_failed"
        )


def _hybrid_collection_incompatibility(
    dense_cfg: object,
    sparse_cfg: object,
    *,
    dense_dim: int,
) -> str | None:
    """Return the first canonical hybrid-schema incompatibility, if any."""
    if not isinstance(dense_cfg, dict):
        return "legacy_unnamed_dense_vector"
    if DENSE_VECTOR_NAME not in dense_cfg:
        return "text_dense_head_missing"
    if not isinstance(sparse_cfg, dict) or SPARSE_VECTOR_NAME not in sparse_cfg:
        return "text_sparse_head_missing"
    dense_vector = dense_cfg[DENSE_VECTOR_NAME]
    dense_incompatibility = (
        "text_dense_dimension_mismatch"
        if int(dense_vector.size) != dense_dim
        else "text_dense_distance_mismatch"
        if getattr(dense_vector, "distance", None) != Distance.COSINE
        else None
    )
    if dense_incompatibility is not None:
        return dense_incompatibility
    if (
        getattr(sparse_cfg[SPARSE_VECTOR_NAME], "modifier", None)
        != qmodels.Modifier.IDF
    ):
        return "text_sparse_idf_missing"
    return None


def ensure_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = settings.embedding.dimension,
) -> CollectionCompatibilityResult:
    """Create a missing collection and reject every existing mismatch.

    Args:
        client: Qdrant client used to inspect and create collection state.
        collection_name: Collection to validate or create.
        dense_dim: Required dimension for the canonical dense vector.

    Returns:
        CollectionCompatibilityResult: Compatibility evidence and performed action.
    """
    result = check_hybrid_collection(client, collection_name, dense_dim)
    if result.compatible or result.reason != "collection_missing":
        return result
    try:
        _create_hybrid_collection(client, collection_name, dense_dim)
        return CollectionCompatibilityResult(True, "created", "compatible", 0)
    except QDRANT_SCHEMA_EXCEPTIONS as exc:
        redaction = build_pii_log_entry(str(exc), key_id="storage.create_collection")
        logger.warning(
            "Qdrant collection creation failed (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        rechecked = check_hybrid_collection(client, collection_name, dense_dim)
        if rechecked.reason != "collection_missing":
            return rechecked
        return CollectionCompatibilityResult(
            False,
            "error",
            "collection_create_failed",
            result.point_count,
        )


def rebuild_empty_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = settings.embedding.dimension,
) -> CollectionCompatibilityResult:
    """Rebuild an incompatible collection only after an exact empty count.

    Collection writers must be stopped by the operator. Qdrant does not
    atomically lock the exact-count/delete sequence.
    """
    result = check_hybrid_collection(client, collection_name, dense_dim)
    if result.compatible or result.action == "error":
        return result
    if result.reason == "collection_missing":
        return ensure_hybrid_collection(client, collection_name, dense_dim)

    try:
        raw_count = getattr(
            client.count(collection_name=collection_name, exact=True),
            "count",
            None,
        )
    except QDRANT_SCHEMA_EXCEPTIONS as exc:
        redaction = build_pii_log_entry(str(exc), key_id="storage.exact_count")
        logger.warning(
            "Qdrant exact collection count failed (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        result = CollectionCompatibilityResult(
            False,
            "error",
            "exact_count_failed",
        )
    else:
        if type(raw_count) is not int:
            result = CollectionCompatibilityResult(
                False,
                "error",
                "exact_count_unavailable",
            )
        elif raw_count != 0:
            result = CollectionCompatibilityResult(
                False,
                "blocked",
                "collection_nonempty_exact",
                raw_count,
            )
        else:
            try:
                client.delete_collection(collection_name)
                _create_hybrid_collection(client, collection_name, dense_dim)
            except QDRANT_SCHEMA_EXCEPTIONS as exc:
                redaction = build_pii_log_entry(
                    str(exc), key_id="storage.rebuild_collection"
                )
                logger.warning(
                    "Qdrant empty collection rebuild failed (error_type={}, error={})",
                    type(exc).__name__,
                    redaction.redacted,
                )
                result = CollectionCompatibilityResult(
                    False,
                    "error",
                    "empty_collection_rebuild_failed",
                    raw_count,
                )
            else:
                result = CollectionCompatibilityResult(
                    True, "recreated", "compatible", 0
                )
    return result


# =============================================================================
# Database Operations (formerly from database.py)
# =============================================================================


def get_client_config() -> dict[str, Any]:
    """Get standard Qdrant client configuration.

    Returns:
        Dictionary with client configuration
    """
    timeout_s = float(settings.database.qdrant_timeout)
    if settings.agents.enable_deadline_propagation:
        timeout_s = min(timeout_s, float(settings.agents.decision_timeout))
    config: dict[str, Any] = {
        "url": settings.database.qdrant_url,
        "timeout": timeout_s,
        "prefer_grpc": True,
    }
    if settings.database.qdrant_api_key is not None:
        api_key = settings.database.qdrant_api_key.get_secret_value().strip()
        if api_key:
            config["api_key"] = api_key
    return config


@contextmanager
def create_sync_client() -> Generator[QdrantClient]:
    """Create sync Qdrant client with proper cleanup.

    Yields:
        QdrantClient: Configured sync client
    """
    client = None
    try:
        config = get_client_config()
        client = QdrantClient(**config)
        logger.debug(
            "Created sync Qdrant client {}", safe_url_for_log(str(config["url"]))
        )
        yield client
    except (
        ResponseHandlingException,
        UnexpectedResponse,
        ConnectionError,
        TimeoutError,
    ) as e:
        redaction = build_pii_log_entry(str(e), key_id="storage.qdrant_sync_client")
        logger.error(
            "Failed to create sync Qdrant client (error_type={}, error={})",
            type(e).__name__,
            redaction.redacted,
        )
        raise
    finally:
        if client is not None:
            try:
                client.close()
            except (ResponseHandlingException, ConnectionError) as e:
                redaction = build_pii_log_entry(
                    str(e), key_id="storage.qdrant_sync_close"
                )
                logger.warning(
                    "Error closing sync client (error_type={}, error={})",
                    type(e).__name__,
                    redaction.redacted,
                )


@asynccontextmanager
async def create_async_client() -> AsyncGenerator[AsyncQdrantClient]:
    """Create async Qdrant client with proper cleanup.

    Yields:
        AsyncQdrantClient: Configured async client
    """
    client = None
    try:
        config = get_client_config()
        client = AsyncQdrantClient(**config)
        logger.debug(
            "Created async Qdrant client {}", safe_url_for_log(str(config["url"]))
        )
        yield client
    except (
        ResponseHandlingException,
        UnexpectedResponse,
        ConnectionError,
        TimeoutError,
    ) as e:
        redaction = build_pii_log_entry(str(e), key_id="storage.qdrant_async_client")
        logger.error(
            "Failed to create async Qdrant client (error_type={}, error={})",
            type(e).__name__,
            redaction.redacted,
        )
        raise
    finally:
        if client is not None:
            try:
                await client.close()
            except (ResponseHandlingException, ConnectionError) as e:
                redaction = build_pii_log_entry(
                    str(e), key_id="storage.qdrant_async_close"
                )
                logger.warning(
                    "Error closing async client (error_type={}, error={})",
                    type(e).__name__,
                    redaction.redacted,
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
    # QdrantVectorStore uses the named dense vector by default even when sparse
    # hybrid search is disabled, so the schema must exist for both modes.
    compatibility = ensure_hybrid_collection(
        client,
        collection_name,
        dense_dim=_dense_embedding_size,
    )
    if not compatibility.compatible:
        client.close()
        raise QdrantCollectionIncompatibleError(collection_name, compatibility)

    try:
        store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            enable_hybrid=enable_hybrid,
            batch_size=settings.monitoring.default_batch_size,
            dense_vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )
        return store
    except ImportError as e:  # fastembed optional for hybrid sparse
        redaction = build_pii_log_entry(str(e), key_id="storage.fastembed_store")
        logger.warning(
            "Hybrid vector store requires FastEmbed; falling back to dense-only "
            "(error_type={}, error={})",
            type(e).__name__,
            redaction.redacted,
        )
        try:
            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                enable_hybrid=False,
                batch_size=settings.monitoring.default_batch_size,
                dense_vector_name=DENSE_VECTOR_NAME,
                sparse_vector_name=SPARSE_VECTOR_NAME,
            )
        except Exception:
            with suppress(Exception):
                client.close()
            raise
    except Exception:
        with suppress(Exception):
            client.close()
        raise


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
        redaction = build_pii_log_entry(
            str(exc), key_id="storage.persist_image_metadata"
        )
        logger.warning(
            "persist_image_metadata failed (point_id={} error_type={} error={})",
            point_id,
            type(exc).__name__,
            redaction.redacted,
        )
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
        redaction = build_pii_log_entry(str(e), key_id="storage.get_collection_info")
        logger.error(
            "Failed to get collection info for {} (error_type={} error={})",
            collection_name,
            type(e).__name__,
            redaction.redacted,
        )
        return {
            "exists": False,
            "error_type": type(e).__name__,
            "error": redaction.redacted,
        }


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
        redaction = build_pii_log_entry(str(e), key_id="storage.test_connection")
        logger.error(
            "Qdrant connection test failed (error_type={} error={})",
            type(e).__name__,
            redaction.redacted,
        )
        return {
            "connected": False,
            "url": settings.database.qdrant_url,
            "error_type": type(e).__name__,
            "error": redaction.redacted,
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
                logger.warning("Collection {} does not exist", collection_name)
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

            logger.success("Cleared collection: {}", collection_name)
            return True

    except (
        ResponseHandlingException,
        UnexpectedResponse,
        ConnectionError,
        TimeoutError,
    ) as e:
        redaction = build_pii_log_entry(str(e), key_id="storage.clear_collection")
        logger.error(
            "Failed to clear collection {} (error_type={} error={})",
            collection_name,
            type(e).__name__,
            redaction.redacted,
        )
        return False


# =============================================================================
# Resource Management Operations (formerly from resource_management.py)
# =============================================================================


@contextmanager
def gpu_memory_context() -> Generator[None]:
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
            redaction = build_pii_log_entry(str(e), key_id="storage.gpu_cleanup")
            logger.warning(
                "GPU cleanup failed during context exit (error_type={} error={})",
                type(e).__name__,
                redaction.redacted,
            )
        finally:
            # Always run garbage collection
            gc.collect()


@asynccontextmanager
async def async_gpu_memory_context() -> AsyncGenerator[None]:
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
            redaction = build_pii_log_entry(str(e), key_id="storage.gpu_cleanup_async")
            logger.warning(
                "GPU cleanup failed during async context exit (error_type={} error={})",
                type(e).__name__,
                redaction.redacted,
            )
        finally:
            # Always run garbage collection
            gc.collect()


@asynccontextmanager
async def model_context(
    model_factory: Callable[..., Any],
    cleanup_method: str | None = None,
    **kwargs: Any,
) -> AsyncGenerator[Any]:
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
) -> Generator[Any]:
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
) -> Generator[dict[str, Any]]:
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
        redaction = build_pii_log_entry(
            str(e), key_id=f"storage.cuda_error_context:{operation_name}"
        )
        if "CUDA" in str(e).upper():
            logger.warning(
                "{} failed with CUDA error (error_type={} error={})",
                operation_name,
                type(e).__name__,
                redaction.redacted,
            )
        else:
            logger.warning(
                "{} failed with runtime error (error_type={} error={})",
                operation_name,
                type(e).__name__,
                redaction.redacted,
            )

        if reraise:
            raise
        result_dict["result"] = default_return
        result_dict["error_type"] = type(e).__name__
        result_dict["error"] = redaction.redacted

    except (OSError, AttributeError) as e:
        redaction = build_pii_log_entry(
            str(e), key_id=f"storage.cuda_error_context:{operation_name}"
        )
        logger.warning(
            "{} failed with system error (error_type={} error={})",
            operation_name,
            type(e).__name__,
            redaction.redacted,
        )

        if reraise:
            raise
        result_dict["result"] = default_return
        result_dict["error_type"] = type(e).__name__
        result_dict["error"] = redaction.redacted

    except (ImportError, ModuleNotFoundError) as e:
        redaction = build_pii_log_entry(
            str(e), key_id=f"storage.cuda_error_context:{operation_name}"
        )
        logger.error(
            "{} failed with import error (error_type={} error={})",
            operation_name,
            type(e).__name__,
            redaction.redacted,
        )

        if reraise:
            raise
        result_dict["result"] = default_return
        result_dict["error_type"] = type(e).__name__
        result_dict["error"] = redaction.redacted


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
            redaction = build_pii_log_entry(
                str(e), key_id=f"storage.safe_cuda_operation:{operation_name}"
            )
            if "CUDA" in str(e).upper():
                logger.warning(
                    "{} failed with CUDA error (error_type={} error={})",
                    operation_name,
                    type(e).__name__,
                    redaction.redacted,
                )
            else:
                logger.warning(
                    "{} failed with runtime error (error_type={} error={})",
                    operation_name,
                    type(e).__name__,
                    redaction.redacted,
                )
        return default_return
    except (OSError, AttributeError) as e:
        if log_errors:
            redaction = build_pii_log_entry(
                str(e), key_id=f"storage.safe_cuda_operation:{operation_name}"
            )
            logger.warning(
                "{} failed with system error (error_type={} error={})",
                operation_name,
                type(e).__name__,
                redaction.redacted,
            )
        return default_return
    except (ImportError, ModuleNotFoundError) as e:
        if log_errors:
            redaction = build_pii_log_entry(
                str(e), key_id=f"storage.safe_cuda_operation:{operation_name}"
            )
            logger.error(
                "{} failed with import error (error_type={} error={})",
                operation_name,
                type(e).__name__,
                redaction.redacted,
            )
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
        redaction = build_pii_log_entry(str(e), key_id="storage.gpu_info")
        logger.warning(
            "Failed to get GPU info (error_type={} error={})",
            type(e).__name__,
            redaction.redacted,
        )

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
            logger.debug("Model has no cleanup method: {}", cleanup_method)
    except (AttributeError, TypeError) as e:
        redaction = build_pii_log_entry(str(e), key_id="storage.model_cleanup")
        logger.warning(
            "Model cleanup failed (error_type={} error={})",
            type(e).__name__,
            redaction.redacted,
        )


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
            logger.debug("Model has no cleanup method: {}", cleanup_method)
    except (AttributeError, TypeError) as e:
        redaction = build_pii_log_entry(str(e), key_id="storage.model_cleanup_sync")
        logger.warning(
            "Model cleanup failed (error_type={} error={})",
            type(e).__name__,
            redaction.redacted,
        )
