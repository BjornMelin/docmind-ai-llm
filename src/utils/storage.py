"""Qdrant client, collection, and vector-store utilities."""

import asyncio
import inspect
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Literal

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
from src.config.embedding_defaults import (
    DEFAULT_BM42_MODEL_ID,
    DEFAULT_BM42_SOURCE_REPO,
    DEFAULT_BM42_SOURCE_REVISION,
)
from src.config.settings import DocMindSettings
from src.config.settings_utils import (
    endpoint_url_allowed,
    parse_endpoint_allowlist_hosts,
)
from src.persistence.deployment_identity import (
    get_or_create_deployment_id,
    read_deployment_id,
)
from src.retrieval import vector_contract
from src.retrieval.sparse_query import sparse_callbacks
from src.utils.log_safety import build_pii_log_entry, safe_url_for_log
from src.utils.qdrant_exceptions import (
    QDRANT_SCHEMA_EXCEPTIONS,
)
from src.utils.qdrant_utils import get_collection_params

QDRANT_UPSERT_BATCH_SIZE = 20
TEXT_COLLECTION_SCHEMA_VERSION = "2"


def _consume_async_close(task: asyncio.Task[Any]) -> None:
    """Consume a scheduled close error at the asynchronous cleanup boundary."""
    with suppress(Exception):
        task.result()


async def _await_async_close(awaitable: Any) -> None:
    """Await a dynamically typed async-client close result."""
    await awaitable


def close_qdrant_clients(
    client: Any | None,
    async_client: Any | None,
) -> None:
    """Best-effort close both clients backing a vector-store owner."""
    close = getattr(client, "close", None)
    if callable(close):
        with suppress(Exception):
            close()

    async_close = getattr(async_client, "close", None)
    if not callable(async_close):
        return
    try:
        result = async_close()
        if not inspect.isawaitable(result):
            return
        close_coro = _await_async_close(result)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(close_coro)
        else:
            close_task = loop.create_task(close_coro)
            close_task.add_done_callback(_consume_async_close)
    except Exception:
        return


def close_vector_store_clients(store: Any | None) -> None:
    """Best-effort close both clients owned by a Qdrant vector store."""
    close_qdrant_clients(
        getattr(store, "client", None),
        getattr(store, "_aclient", None),
    )


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


def canonical_text_collection_metadata(
    *,
    dense_dim: int,
    sparse_enabled: bool,
    cfg: DocMindSettings = settings,
) -> dict[str, Any]:
    """Return immutable semantic identity for a staged text collection."""
    return _text_collection_metadata(
        deployment_id=get_or_create_deployment_id(cfg.data_dir),
        dense_dim=dense_dim,
        sparse_enabled=sparse_enabled,
        cfg=cfg,
    )


def _expected_text_collection_metadata(
    *,
    dense_dim: int,
    sparse_enabled: bool,
    cfg: DocMindSettings = settings,
) -> dict[str, Any]:
    """Return read-only expected metadata for compatibility checks."""
    return _text_collection_metadata(
        deployment_id=read_deployment_id(cfg.data_dir),
        dense_dim=dense_dim,
        sparse_enabled=sparse_enabled,
        cfg=cfg,
    )


def _text_collection_metadata(
    *,
    deployment_id: str,
    dense_dim: int,
    sparse_enabled: bool,
    cfg: DocMindSettings,
) -> dict[str, Any]:
    """Build text collection metadata from an explicit deployment identity."""
    return {
        "docmind_deployment_id": deployment_id,
        "docmind_owner": "text",
        "docmind_schema_version": TEXT_COLLECTION_SCHEMA_VERSION,
        "dense_model": str(cfg.embedding.model_name),
        "dense_revision": str(cfg.embedding.model_revision or "unpinned"),
        "dense_dimension": int(dense_dim),
        "sparse_enabled": bool(sparse_enabled),
        "sparse_model": DEFAULT_BM42_MODEL_ID,
        "sparse_source_repo": DEFAULT_BM42_SOURCE_REPO,
        "sparse_source_revision": DEFAULT_BM42_SOURCE_REVISION,
        "sparse_encoding_contract": vector_contract.SPARSE_ENCODING_CONTRACT,
    }


def _create_hybrid_collection(
    client: QdrantClient,
    name: str,
    dense_dim: int,
    *,
    sparse_enabled: bool,
) -> None:
    """Create a collection with canonical named dense and sparse vectors."""
    client.create_collection(
        collection_name=name,
        vectors_config={
            vector_contract.DENSE_VECTOR_NAME: VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            vector_contract.SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=qmodels.Modifier.IDF,
            ),
        },
        metadata=canonical_text_collection_metadata(
            dense_dim=dense_dim,
            sparse_enabled=sparse_enabled,
        ),
    )
    logger.info("Created hybrid collection {} (dense={})", name, dense_dim)


def check_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = settings.embedding.dimension,
    *,
    sparse_enabled: bool | None = None,
) -> CollectionCompatibilityResult:
    """Inspect named-vector compatibility without mutating Qdrant.

    Args:
        client: Qdrant client used to inspect collection state.
        collection_name: Collection whose named-vector schema is inspected.
        dense_dim: Required dimension for the canonical dense vector.
        sparse_enabled: Expected sparse-vector indexing policy.

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
        effective_sparse = (
            vector_contract.sparse_retrieval_enabled()
            if sparse_enabled is None
            else bool(sparse_enabled)
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
            collection_metadata=getattr(
                getattr(info, "config", None), "metadata", None
            ),
            dense_dim=dense_dim,
            sparse_enabled=effective_sparse,
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


def _hybrid_collection_incompatibility(  # noqa: PLR0911
    dense_cfg: object,
    sparse_cfg: object,
    *,
    collection_metadata: object,
    dense_dim: int,
    sparse_enabled: bool,
) -> str | None:
    """Return the first canonical hybrid-schema incompatibility, if any."""
    if not isinstance(dense_cfg, dict):
        return "legacy_unnamed_dense_vector"
    if vector_contract.DENSE_VECTOR_NAME not in dense_cfg:
        return "text_dense_head_missing"
    if (
        not isinstance(sparse_cfg, dict)
        or vector_contract.SPARSE_VECTOR_NAME not in sparse_cfg
    ):
        return "text_sparse_head_missing"
    dense_vector = dense_cfg[vector_contract.DENSE_VECTOR_NAME]
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
        getattr(sparse_cfg[vector_contract.SPARSE_VECTOR_NAME], "modifier", None)
        != qmodels.Modifier.IDF
    ):
        return "text_sparse_idf_missing"
    expected_metadata = _expected_text_collection_metadata(
        dense_dim=dense_dim,
        sparse_enabled=sparse_enabled,
    )
    if not isinstance(collection_metadata, dict):
        return "text_collection_metadata_missing"
    if any(
        collection_metadata.get(key) != value
        for key, value in expected_metadata.items()
    ):
        return "text_collection_metadata_mismatch"
    return None


def ensure_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = settings.embedding.dimension,
    *,
    sparse_enabled: bool | None = None,
) -> CollectionCompatibilityResult:
    """Create a missing collection and reject every existing mismatch.

    Args:
        client: Qdrant client used to inspect and create collection state.
        collection_name: Collection to validate or create.
        dense_dim: Required dimension for the canonical dense vector.
        sparse_enabled: Sparse-vector policy to encode in collection metadata.

    Returns:
        CollectionCompatibilityResult: Compatibility evidence and performed action.
    """
    effective_sparse = (
        vector_contract.sparse_retrieval_enabled()
        if sparse_enabled is None
        else bool(sparse_enabled)
    )
    result = check_hybrid_collection(
        client,
        collection_name,
        dense_dim,
        sparse_enabled=effective_sparse,
    )
    if result.compatible or result.reason != "collection_missing":
        return result
    try:
        _create_hybrid_collection(
            client,
            collection_name,
            dense_dim,
            sparse_enabled=effective_sparse,
        )
        return CollectionCompatibilityResult(True, "created", "compatible", 0)
    except QDRANT_SCHEMA_EXCEPTIONS as exc:
        redaction = build_pii_log_entry(str(exc), key_id="storage.create_collection")
        logger.warning(
            "Qdrant collection creation failed (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        rechecked = check_hybrid_collection(
            client,
            collection_name,
            dense_dim,
            sparse_enabled=effective_sparse,
        )
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
    effective_sparse = vector_contract.sparse_retrieval_enabled()
    result = check_hybrid_collection(
        client,
        collection_name,
        dense_dim,
        sparse_enabled=effective_sparse,
    )
    if result.compatible or result.action == "error":
        return result
    if result.reason == "collection_missing":
        return ensure_hybrid_collection(
            client,
            collection_name,
            dense_dim,
            sparse_enabled=effective_sparse,
        )

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
                _create_hybrid_collection(
                    client,
                    collection_name,
                    dense_dim,
                    sparse_enabled=effective_sparse,
                )
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


def get_client_config(cfg: DocMindSettings | None = None) -> dict[str, Any]:
    """Get standard Qdrant client configuration.

    Args:
        cfg: Optional settings container. Defaults to the process settings.

    Returns:
        Dictionary with client configuration
    """
    active_settings = cfg or settings
    qdrant_url = str(active_settings.database.qdrant_url)
    if not active_settings.security.allow_remote_endpoints:
        allowed_hosts = parse_endpoint_allowlist_hosts(
            active_settings.security.endpoint_allowlist
        )
        if not endpoint_url_allowed(qdrant_url, allowed_hosts=allowed_hosts):
            raise ValueError("Qdrant URL is blocked by the endpoint security policy")
    timeout_s = float(active_settings.database.qdrant_timeout)
    timeout_s = min(timeout_s, float(active_settings.agents.decision_timeout))
    config: dict[str, Any] = {
        "url": qdrant_url,
        "timeout": timeout_s,
        "prefer_grpc": True,
    }
    if active_settings.database.qdrant_api_key is not None:
        api_key = active_settings.database.qdrant_api_key.get_secret_value().strip()
        if api_key:
            config["api_key"] = api_key
    return config


@contextmanager
def create_sync_client(
    cfg: DocMindSettings | None = None,
) -> Generator[QdrantClient]:
    """Create sync Qdrant client with proper cleanup.

    Args:
        cfg: Optional settings container. Defaults to the process settings.

    Yields:
        QdrantClient: Configured sync client
    """
    client = None
    try:
        config = get_client_config(cfg) if cfg is not None else get_client_config()
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
async def create_async_client(
    cfg: DocMindSettings | None = None,
) -> AsyncGenerator[AsyncQdrantClient]:
    """Create async Qdrant client with proper cleanup.

    Args:
        cfg: Optional settings container. Defaults to the process settings.

    Yields:
        AsyncQdrantClient: Configured async client
    """
    client = None
    try:
        config = get_client_config(cfg) if cfg is not None else get_client_config()
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
    enable_hybrid: bool | None = None,
) -> QdrantVectorStore:
    """Create QdrantVectorStore with standard configuration.

    Args:
        collection_name: Name of the collection
        _dense_embedding_size: Size of dense embeddings (unused for now)
        enable_hybrid: Whether to persist canonical sparse vectors. Defaults
            to the unified hybrid-or-keyword feature policy.

    Returns:
        Configured QdrantVectorStore
    """
    client_config = get_client_config()
    client = QdrantClient(**client_config)
    effective_sparse = (
        vector_contract.sparse_retrieval_enabled()
        if enable_hybrid is None
        else bool(enable_hybrid)
    )
    # QdrantVectorStore uses the named dense vector by default even when sparse
    # hybrid search is disabled, so the schema must exist for both modes.
    compatibility = ensure_hybrid_collection(
        client,
        collection_name,
        dense_dim=_dense_embedding_size,
        sparse_enabled=effective_sparse,
    )
    if not compatibility.compatible:
        client.close()
        raise QdrantCollectionIncompatibleError(collection_name, compatibility)

    try:
        async_client = AsyncQdrantClient(**client_config)
    except Exception:
        with suppress(Exception):
            client.close()
        raise
    try:
        sparse_doc_fn, sparse_query_fn = sparse_callbacks()
        store = QdrantVectorStore(
            client=client,
            aclient=async_client,
            collection_name=collection_name,
            enable_hybrid=effective_sparse,
            batch_size=QDRANT_UPSERT_BATCH_SIZE,
            dense_vector_name=vector_contract.DENSE_VECTOR_NAME,
            sparse_vector_name=vector_contract.SPARSE_VECTOR_NAME,
            sparse_doc_fn=sparse_doc_fn,
            sparse_query_fn=sparse_query_fn,
        )
        return store
    except Exception:
        close_qdrant_clients(client, async_client)
        raise


def connect_vector_store(
    collection_name: str,
    _dense_embedding_size: int = settings.embedding.dimension,
    enable_hybrid: bool | None = None,
) -> QdrantVectorStore:
    """Connect to an existing compatible collection without creating it."""
    client_config = get_client_config()
    client = QdrantClient(**client_config)
    effective_sparse = (
        vector_contract.sparse_retrieval_enabled()
        if enable_hybrid is None
        else bool(enable_hybrid)
    )
    compatibility = check_hybrid_collection(
        client,
        collection_name,
        dense_dim=_dense_embedding_size,
        sparse_enabled=effective_sparse,
    )
    if not compatibility.compatible:
        client.close()
        raise QdrantCollectionIncompatibleError(collection_name, compatibility)

    try:
        async_client = AsyncQdrantClient(**client_config)
    except Exception:
        close_qdrant_clients(client, None)
        raise
    try:
        sparse_doc_fn, sparse_query_fn = sparse_callbacks()
        return QdrantVectorStore(
            client=client,
            aclient=async_client,
            collection_name=collection_name,
            enable_hybrid=effective_sparse,
            batch_size=QDRANT_UPSERT_BATCH_SIZE,
            dense_vector_name=vector_contract.DENSE_VECTOR_NAME,
            sparse_vector_name=vector_contract.SPARSE_VECTOR_NAME,
            sparse_doc_fn=sparse_doc_fn,
            sparse_query_fn=sparse_query_fn,
        )
    except Exception:
        close_qdrant_clients(client, async_client)
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
