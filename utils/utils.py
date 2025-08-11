"""Core utility functions for DocMind AI.

This module provides essential utilities for the DocMind AI system including
logging configuration, hardware detection for GPU acceleration, optimized
embedding model initialization, and RRF (Reciprocal Rank Fusion) configuration
validation.

The module focuses on:
- Logging setup with file and console handlers
- Hardware detection for CUDA and FastEmbed providers
- GPU-optimized embedding model configuration with torch.compile
- Research-backed RRF parameter validation
- spaCy model management with automatic downloading

Example:
    Basic usage of core utilities::

        from utils.utils import detect_hardware, get_embed_model
        from loguru import logger

        # Loguru is auto-configured
        logger.info("Application starting")

        # Check hardware capabilities
        hardware = detect_hardware()
        print(f"CUDA available: {hardware['cuda_available']}")

        # Get optimized embedding model
        embed_model = get_embed_model()

Attributes:
    settings (AppSettings): Global application settings instance.
"""

import asyncio
import gc
import subprocess
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any

import torch

try:
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
except ImportError:
    FastEmbedEmbedding = None
from loguru import logger
from qdrant_client import AsyncQdrantClient

from models.core import settings

from .exceptions import (
    ConfigurationError,
    handle_embedding_error,
)
from .logging_utils import log_error_with_context, log_performance
from .model_manager import ModelManager
from .retry_utils import (
    embedding_retry,
    safe_execute,
    with_fallback,
)

# settings is now imported from models.core


def detect_hardware() -> dict[str, Any]:
    """Detect hardware capabilities using FastEmbed native detection.

    Performs comprehensive hardware detection including CUDA availability,
    GPU specifications, and FastEmbed execution providers. Uses the model
    manager singleton pattern to prevent redundant model initializations
    and optimize resource usage with structured error handling.

    Returns:
        Dictionary containing hardware information with keys:
        - 'cuda_available' (bool): Whether CUDA is available
        - 'gpu_name' (str): Name of the primary GPU or 'Unknown'
        - 'vram_total_gb' (float | None): Total VRAM in GB
        - 'fastembed_providers' (list[str]): Available FastEmbed providers

    Note:
        Falls back gracefully if FastEmbed detection fails. Always returns
        a dictionary with all expected keys, using safe defaults when
        detection fails.

    Example:
        >>> hardware = detect_hardware()
        >>> if hardware['cuda_available']:
        ...     print(f"GPU: {hardware['gpu_name']} with {hardware['vram_total_gb']}GB")
        ... else:
        ...     print("Running on CPU")
    """
    start_time = time.perf_counter()

    hardware_info = {
        "cuda_available": False,
        "gpu_name": "Unknown",
        "vram_total_gb": None,
        "fastembed_providers": [],
    }

    # Use FastEmbed's native hardware detection with safe execution
    def detect_fastembed_providers():
        test_model = ModelManager.get_text_embedding_model("BAAI/bge-small-en-v1.5")
        try:
            providers = test_model.model.model.get_providers()
            hardware_info["fastembed_providers"] = providers
            hardware_info["cuda_available"] = "CUDAExecutionProvider" in providers
            logger.info(
                "FastEmbed providers detected",
                extra={
                    "providers": providers,
                    "cuda_available": hardware_info["cuda_available"],
                },
            )
        except (AttributeError, RuntimeError, ImportError) as e:
            logger.warning(f"FastEmbed provider detection fallback: {e}")
            hardware_info["cuda_available"] = torch.cuda.is_available()
        finally:
            del test_model  # Cleanup

    # Safe execution with fallback
    safe_execute(
        detect_fastembed_providers,
        default_value=None,
        operation_name="fastembed_detection",
    )

    # If FastEmbed detection failed, use basic PyTorch detection
    if not hardware_info["fastembed_providers"]:
        hardware_info["cuda_available"] = torch.cuda.is_available()

    # Get detailed GPU information with error handling
    if hardware_info["cuda_available"] and torch.cuda.is_available():

        def get_gpu_info():
            hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            hardware_info["vram_total_gb"] = round(vram_gb, 1)

        safe_execute(
            get_gpu_info, default_value=None, operation_name="gpu_info_detection"
        )

    # Log performance and results
    duration = time.perf_counter() - start_time
    log_performance(
        "hardware_detection",
        duration,
        cuda_available=hardware_info["cuda_available"],
        gpu_name=hardware_info["gpu_name"],
        providers_count=len(hardware_info["fastembed_providers"]),
    )

    return hardware_info


@embedding_retry
def get_embed_model() -> FastEmbedEmbedding:
    """Create optimized embedding model with GPU acceleration and error recovery.

    Initializes a FastEmbedEmbedding model with optimal configuration for
    both CPU and GPU environments. Applies torch.compile optimization when
    available and uses research-backed model settings for best performance.
    Includes structured error handling and automatic retries.

    GPU optimizations include:
    - torch.compile with 'reduce-overhead' mode
    - CUDA execution provider prioritization
    - Automatic batch size configuration
    - Mixed precision support

    Returns:
        FastEmbedEmbedding: Fully configured and optimized embedding model.

    Raises:
        EmbeddingError: If model initialization fails after retries.

    Note:
        The model is configured with:
        - Model: BGE-Large (BAAI/bge-large-en-v1.5) from settings
        - Max length: 512 tokens for optimal performance
        - Cache directory: './embeddings_cache'
        - Dynamic batching for variable input sizes

    Example:
        >>> embed_model = get_embed_model()
        >>> embeddings = embed_model.get_text_embedding("Hello world")
        >>> print(f"Embedding dimension: {len(embeddings)}")
    """
    start_time = time.perf_counter()

    logger.info(
        "Creating embedding model",
        extra={
            "model": settings.dense_embedding_model,
            "gpu_enabled": settings.gpu_acceleration,
            "batch_size": settings.embedding_batch_size,
            "max_length": 512,
        },
    )

    try:
        # Determine providers based on hardware availability
        providers = ["CPUExecutionProvider"]
        if settings.gpu_acceleration and torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for embeddings")

        # Use factory for consistent embedding model creation
        from .embedding_factory import EmbeddingFactory

        embed_model = EmbeddingFactory.create_dense_embedding(
            use_gpu=settings.gpu_acceleration
        )

        duration = time.perf_counter() - start_time
        log_performance(
            "embedding_model_creation",
            duration,
            model=settings.dense_embedding_model,
            gpu_enabled=settings.gpu_acceleration and torch.cuda.is_available(),
            providers=providers,
        )

        logger.success(
            f"Embedding model created successfully: {settings.dense_embedding_model}"
        )
        return embed_model

    except Exception as e:
        log_error_with_context(
            e,
            "embedding_model_creation",
            context={
                "model": settings.dense_embedding_model,
                "gpu_requested": settings.gpu_acceleration,
                "cuda_available": torch.cuda.is_available(),
                "batch_size": settings.embedding_batch_size,
            },
        )
        raise handle_embedding_error(
            e,
            operation="get_embed_model",
            model=settings.dense_embedding_model,
            gpu_requested=settings.gpu_acceleration,
            cuda_available=torch.cuda.is_available(),
        ) from e


"""Verify and validate Reciprocal Rank Fusion (RRF) configuration parameters.

This function performs a comprehensive validation of RRF configuration settings
against established research recommendations. It ensures that embedding fusion
weights, alpha parameters, and other critical settings are optimally configured
for hybrid search performance.

Research-backed validation includes:
- Dense and sparse embedding weight distribution
- RRF alpha parameter range
- Prefetch mechanism status
- Embedding dimension compatibility

The function provides a detailed verification report with:
- Weights correctness
- Alpha parameter validation
- Prefetch mechanism status
- Potential configuration issues
- Recommended improvements

Helps prevent misconfiguration and ensures alignment with best practices in
hybrid search and embedding fusion strategies.
"""


def verify_rrf_configuration(settings: AppSettings) -> dict[str, Any]:
    """Verify RRF configuration against research recommendations.

    Validates Reciprocal Rank Fusion (RRF) parameters against established
    research findings to ensure optimal hybrid search performance. Checks
    weight distribution, alpha parameters, and provides recommendations
    for configuration improvements.

    Research-backed requirements:
    - Dense embedding weight: 0.7 (±0.05 tolerance)
    - Sparse embedding weight: 0.3 (±0.05 tolerance)
    - RRF alpha parameter: 10-100 range (60 optimal)
    - Prefetch mechanism: Always enabled

    Args:
        settings: Application settings containing RRF configuration.

    Returns:
        Dictionary containing verification results with keys:
        - 'weights_correct' (bool): Whether weights match research findings
        - 'prefetch_enabled' (bool): Prefetch mechanism status (always True)
        - 'alpha_in_range' (bool): Whether alpha is in valid range
        - 'computed_hybrid_alpha' (float): Calculated hybrid alpha for LlamaIndex
        - 'issues' (list[str]): List of configuration problems found
        - 'recommendations' (list[str]): Suggested fixes for issues

    Note:
        The computed_hybrid_alpha is calculated as:
        dense_weight / (dense_weight + sparse_weight)
        This value is used by LlamaIndex for hybrid query processing.

    Example:
        >>> from models.core import settings
        >>> verification = verify_rrf_configuration(settings)
        >>> if verification['issues']:
        ...     for issue in verification['issues']:
        ...         print(f"Issue: {issue}")
        >>> print(f"Hybrid alpha: {verification['computed_hybrid_alpha']:.3f}")
    """
    verification = {
        "weights_correct": False,
        "prefetch_enabled": True,  # Always enabled in our implementation
        "alpha_in_range": False,
        "computed_hybrid_alpha": 0.0,
        "issues": [],
        "recommendations": [],
    }

    # Check research-backed weights
    expected_dense = 0.7
    expected_sparse = 0.3
    if (
        abs(settings.rrf_fusion_weight_dense - expected_dense) < 0.05
        and abs(settings.rrf_fusion_weight_sparse - expected_sparse) < 0.05
    ):
        verification["weights_correct"] = True
    else:
        verification["issues"].append(
            f"Weights not research-backed: dense={settings.rrf_fusion_weight_dense}, "
            f"sparse={settings.rrf_fusion_weight_sparse} (expected 0.7/0.3)"
        )
        verification["recommendations"].append(
            "Update weights to research-backed values: dense=0.7, sparse=0.3"
        )

    # Check RRF alpha parameter (research suggests 10-100, with 60 as optimal)
    if 10 <= settings.rrf_fusion_alpha <= 100:
        verification["alpha_in_range"] = True
    else:
        verification["issues"].append(
            f"RRF alpha ({settings.rrf_fusion_alpha}) outside research range (10-100)"
        )
        verification["recommendations"].append(
            "Set RRF alpha between 10-100, with 60 as optimal"
        )

    # Calculate hybrid alpha for LlamaIndex
    verification["computed_hybrid_alpha"] = settings.rrf_fusion_weight_dense / (
        settings.rrf_fusion_weight_dense + settings.rrf_fusion_weight_sparse
    )

    logger.info("RRF Configuration Verification: %s", verification)
    return verification


@with_fallback(lambda model_name: None)  # Graceful fallback returns None
def ensure_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Ensure spaCy model is available, download if needed using standard package mgmt.

    Attempts to load a spaCy model and automatically downloads it if not
    found locally. Uses `uv run` when available for better environment
    handling, falls back to standard Python execution. This approach
    follows security best practices by avoiding custom URL dependencies.

    Args:
        model_name: Name of the spaCy model to load. Common options include:
            - 'en_core_web_sm': Small English model (~15MB)
            - 'en_core_web_md': Medium English model (~40MB)
            - 'en_core_web_lg': Large English model (~560MB)
            Defaults to 'en_core_web_sm'.

    Returns:
        Loaded spaCy Language model instance ready for NLP processing,
        or None if loading fails and fallback is used.

    Raises:
        ConfigurationError: If the model cannot be loaded or downloaded, or if
            spaCy is not installed.

    Note:
        Downloads can take several minutes depending on model size and
        network speed. The function prefers `uv run python -m spacy download`
        for consistency with project tooling, but falls back to standard
        `python -m spacy download` if uv is not available.

    Security:
        This function uses standard spaCy model installation which downloads
        models from spaCy's official distribution, avoiding custom URL
        dependencies that bypass security scanning.

    Example:
        >>> nlp = ensure_spacy_model("en_core_web_sm")
        >>> if nlp:
        ...     doc = nlp("This is a test sentence.")
        ...     entities = [(ent.text, ent.label_) for ent in doc.ents]
        ...     print(f"Found entities: {entities}")
    """
    start_time = time.perf_counter()

    logger.info(f"Loading spaCy model: {model_name}")

    try:
        import spacy

        # Try to load existing model first
        try:
            nlp = spacy.load(model_name)
            logger.success(f"spaCy model '{model_name}' loaded successfully")

            duration = time.perf_counter() - start_time
            log_performance(
                "spacy_model_load",
                duration,
                model_name=model_name,
                loaded_from_cache=True,
            )
            return nlp

        except OSError:
            # Model not found locally, try to download
            logger.info(f"spaCy model '{model_name}' not found locally, downloading...")

            try:
                # Download model with timeout and error handling using uv
                # First try with uv run for better environment handling
                import os

                download_cmd = [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "spacy",
                    "download",
                    model_name,
                ]

                # Fallback to regular python if uv is not available
                if not any(
                    os.path.exists(os.path.join(path, "uv"))
                    for path in os.environ.get("PATH", "").split(os.pathsep)
                ):
                    download_cmd = ["python", "-m", "spacy", "download", model_name]

                subprocess.run(
                    download_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout for downloads
                )

                logger.info(f"spaCy model '{model_name}' downloaded successfully")

                # Try loading again after download
                nlp = spacy.load(model_name)
                logger.success(f"spaCy model '{model_name}' loaded after download")

                duration = time.perf_counter() - start_time
                log_performance(
                    "spacy_model_download_and_load",
                    duration,
                    model_name=model_name,
                    download_required=True,
                )
                return nlp

            except subprocess.TimeoutExpired as e:
                raise ConfigurationError(
                    f"spaCy model '{model_name}' download timed out",
                    context={
                        "model_name": model_name,
                        "timeout_seconds": 300,
                        "suggestion": "Try downloading manually or use a smaller model",
                    },
                    operation="spacy_model_download",
                ) from e
            except (subprocess.CalledProcessError, OSError) as e:
                error_context = {
                    "model_name": model_name,
                    "command": download_cmd,
                    "stderr": getattr(e, "stderr", str(e)),
                }

                log_error_with_context(e, "spacy_model_download", context=error_context)

                raise ConfigurationError(
                    f"Failed to download spaCy model '{model_name}'",
                    context=error_context,
                    original_error=e,
                    operation="spacy_model_download",
                ) from e

    except ImportError as e:
        log_error_with_context(
            e,
            "spacy_import",
            context={
                "model_name": model_name,
                "suggestion": "Install spaCy with: pip install spacy",
            },
        )

        raise ConfigurationError(
            "spaCy is not installed",
            context={
                "model_name": model_name,
                "installation_command": "pip install spacy",
            },
            original_error=e,
            operation="spacy_import",
        ) from e

    except Exception as e:
        log_error_with_context(
            e, "spacy_model_ensure", context={"model_name": model_name}
        )

        raise ConfigurationError(
            f"Unexpected error loading spaCy model '{model_name}'",
            context={"model_name": model_name},
            original_error=e,
            operation="spacy_model_ensure",
        ) from e


# Resource Management Utilities for Critical Fix P0


@asynccontextmanager
async def managed_gpu_operation():
    """Context manager for GPU operations with cleanup.

    Provides proper GPU memory management during operations,
    ensuring CUDA cache is cleared and garbage collection is performed
    to prevent memory leaks.

    Usage:
        async with managed_gpu_operation():
            # GPU operations here
            pass
    """
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()


@asynccontextmanager
async def managed_async_qdrant_client(url: str):
    """Context manager for AsyncQdrantClient with proper cleanup.

    Args:
        url: Qdrant server URL

    Yields:
        AsyncQdrantClient: Properly managed client instance

    Usage:
        async with managed_async_qdrant_client(url) as client:
            # Use client here
            pass
    """
    client = None
    try:
        client = AsyncQdrantClient(url=url)
        yield client
    finally:
        if client is not None:
            await client.close()


@asynccontextmanager
async def managed_embedding_model(model_class, model_kwargs):
    """Context manager for embedding models with proper cleanup.

    Ensures embedding models are properly cleaned up after use,
    including GPU memory cleanup for CUDA models.

    Args:
        model_class: The embedding model class to instantiate
        model_kwargs: Keyword arguments for model initialization

    Yields:
        Properly managed embedding model instance

    Usage:
        async with managed_embedding_model(FastEmbedEmbedding, kwargs) as model:
            # Use model here
            pass
    """
    model = None
    try:
        model = model_class(**model_kwargs)
        yield model
    finally:
        # Clean up model resources
        if model is not None:
            # Clear model cache and GPU memory if applicable
            if hasattr(model, "model") and model.model is not None:
                del model.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()


class AsyncQdrantConnectionPool:
    """Production-ready async Qdrant connection pool.

    Provides efficient connection pooling for AsyncQdrantClient instances
    with proper lifecycle management, resource cleanup, and performance
    optimization for concurrent operations.
    """

    def __init__(self, url: str, max_size: int = 10, min_size: int = 2):
        """Initialize connection pool.

        Args:
            url: Qdrant server URL
            max_size: Maximum number of connections in pool
            min_size: Minimum number of connections to maintain
        """
        self.url = url
        self.max_size = max_size
        self.min_size = min_size
        self._pool: asyncio.Queue[AsyncQdrantClient] = asyncio.Queue(max_size)
        self._current_size = 0
        self._lock = asyncio.Lock()
        self._closed = False

    async def _create_client(self) -> AsyncQdrantClient:
        """Create a new client connection."""
        return AsyncQdrantClient(
            url=self.url,
            timeout=30.0,
            prefer_grpc=True,  # Better performance
        )

    async def acquire(self) -> AsyncQdrantClient:
        """Acquire a client from the pool.

        Returns:
            AsyncQdrantClient: Client instance ready for use

        Raises:
            RuntimeError: If connection pool is closed
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        # Try to get from pool first
        try:
            client = self._pool.get_nowait()
            return client
        except asyncio.QueueEmpty:
            pass

        # Create new client if under max size
        async with self._lock:
            if self._current_size < self.max_size:
                client = await self._create_client()
                self._current_size += 1
                return client

        # Wait for available client
        return await self._pool.get()

    async def release(self, client: AsyncQdrantClient):
        """Release a client back to the pool.

        Args:
            client: AsyncQdrantClient to return to pool
        """
        if self._closed:
            await client.close()
            return

        try:
            self._pool.put_nowait(client)
        except asyncio.QueueFull:
            # Pool is full, close excess client
            await client.close()
            async with self._lock:
                self._current_size -= 1

    async def close(self):
        """Close all connections in the pool."""
        self._closed = True

        # Close all clients in pool
        clients_to_close = []
        while not self._pool.empty():
            try:
                client = self._pool.get_nowait()
                clients_to_close.append(client)
            except asyncio.QueueEmpty:
                break

        # Close all clients in parallel
        if clients_to_close:
            await asyncio.gather(
                *[client.close() for client in clients_to_close], return_exceptions=True
            )

        self._current_size = 0


def validate_startup_configuration(settings: AppSettings) -> dict[str, Any]:
    """Perform comprehensive startup configuration validation.

    Args:
        settings: Application settings to validate.

    Returns:
        Dict with validation results and warnings.

    Raises:
        RuntimeError: If critical configuration errors are found.
    """
    results = {"valid": True, "warnings": [], "errors": [], "info": []}

    # Check Qdrant connectivity
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url)
        client.get_collections()
        results["info"].append(f"Qdrant connection successful: {settings.qdrant_url}")
        client.close()
    except Exception as e:
        results["errors"].append(f"Qdrant connection failed: {e}")
        results["valid"] = False

    # Check GPU configuration
    if settings.gpu_acceleration:
        import torch

        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                results["info"].append(f"GPU available: {gpu_name}")
            except Exception as e:
                results["warnings"].append(f"GPU detection issue: {e}")
        else:
            results["warnings"].append("GPU acceleration enabled but no GPU available")

    # Check embedding model dimensions
    expected_dims = {
        "BAAI/bge-large-en-v1.5": 1024,
        "jinaai/jina-embeddings-v3": 1024,
        "jinaai/jina-embeddings-v4": 1024,
    }
    if settings.dense_embedding_model in expected_dims:
        expected = expected_dims[settings.dense_embedding_model]
        if settings.dense_embedding_dimension != expected:
            results["warnings"].append(
                f"Model {settings.dense_embedding_model} typically uses "
                f"{expected}D, but {settings.dense_embedding_dimension}D configured"
            )

    # Check RRF configuration
    if settings.enable_sparse_embeddings:
        # Check alpha parameter
        if not 10 <= settings.rrf_fusion_alpha <= 100:
            results["warnings"].append(
                f"RRF alpha {settings.rrf_fusion_alpha} is outside typical range "
                "[10, 100]"
            )

        # Verify RRF configuration
        verification = verify_rrf_configuration(settings)
        if verification["issues"]:
            results["warnings"].extend(verification["issues"])

    # Check chunk configuration
    if settings.chunk_overlap >= settings.chunk_size:
        results["errors"].append(
            f"Chunk overlap ({settings.chunk_overlap}) must be less than "
            f"chunk size ({settings.chunk_size})"
        )
        results["valid"] = False

    # Check model paths for llamacpp
    if settings.backend == "llamacpp":
        import os

        if not os.path.exists(settings.llamacpp_model_path):
            results["warnings"].append(
                f"Llama.cpp model path does not exist: {settings.llamacpp_model_path}"
            )

    if not results["valid"]:
        error_msg = "Critical configuration errors:\n" + "\n".join(results["errors"])
        raise RuntimeError(error_msg)

    return results


# Global pool instance
_qdrant_pool: AsyncQdrantConnectionPool | None = None


async def get_qdrant_pool() -> AsyncQdrantConnectionPool:
    """Get or create the global Qdrant connection pool.

    Returns:
        AsyncQdrantConnectionPool: Global connection pool instance
    """
    global _qdrant_pool
    if _qdrant_pool is None or _qdrant_pool._closed:
        settings_instance = AppSettings()
        _qdrant_pool = AsyncQdrantConnectionPool(
            url=settings_instance.qdrant_url,
            max_size=getattr(settings_instance, "qdrant_pool_size", 10),
        )
    return _qdrant_pool


def async_timer(func: Callable) -> Callable:
    """Decorator to measure async function execution time.

    Args:
        func: Async function to time

    Returns:
        Wrapped function that logs execution time
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info(f"{func.__name__} completed in {duration:.2f}s")

    return wrapper


# NOTE: PerformanceMonitor implementations consolidated to utils.monitoring.py
# Use: from utils.monitoring import get_performance_monitor
