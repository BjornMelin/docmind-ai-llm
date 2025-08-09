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

        from utils.utils import setup_logging, detect_hardware, get_embed_model

        # Setup logging
        setup_logging("INFO")

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
from typing import Any, Optional

import torch
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import AsyncQdrantClient

from models import AppSettings
from utils.error_recovery import (
    embedding_retry,
    safe_execute,
    with_fallback,
)
from utils.exceptions import (
    ConfigurationError,
    handle_embedding_error,
)
from utils.logging_config import log_error_with_context, log_performance, logger
from utils.model_manager import ModelManager

settings = AppSettings()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application logging using structured logging.

    Delegates to the structured logging configuration in logging_config.py
    for comprehensive error tracking and performance monitoring.

    Args:
        log_level: Logging level as string. Must be one of 'DEBUG', 'INFO',
            'WARNING', 'ERROR', or 'CRITICAL'. Defaults to 'INFO'.

    Note:
        This function is deprecated. Use setup_logging from logging_config
        directly for full structured logging capabilities.

    Example:
        >>> setup_logging("DEBUG")
        >>> logger.info("Application started")
    """
    from utils.logging_config import setup_logging as setup_structured_logging

    logger.warning(
        "Using deprecated setup_logging. Use utils.logging_config.setup_logging"
    )
    setup_structured_logging(console_level=log_level)


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
        from utils.embedding_factory import EmbeddingFactory

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
        )


"""
Def documentation for verify_rrf_configuration.

TODO: Add detailed description.
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
        >>> from models import AppSettings
        >>> settings = AppSettings()
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
    """Ensure spaCy model is available, download if needed.

    Attempts to load a spaCy model and automatically downloads it if not
    found locally. Provides robust error handling, structured logging,
    and graceful fallbacks throughout the process.

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
        network speed. The function handles subprocess execution for
        model downloads and provides detailed logging throughout.

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
                # Download model with timeout and error handling
                subprocess.run(
                    ["python", "-m", "spacy", "download", model_name],
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

            except subprocess.TimeoutExpired:
                raise ConfigurationError(
                    f"spaCy model '{model_name}' download timed out",
                    context={
                        "model_name": model_name,
                        "timeout_seconds": 300,
                        "suggestion": "Try downloading manually or use a smaller model",
                    },
                    operation="spacy_model_download",
                )
            except (subprocess.CalledProcessError, OSError) as e:
                error_context = {
                    "model_name": model_name,
                    "command": ["python", "-m", "spacy", "download", model_name],
                    "stderr": getattr(e, "stderr", str(e)),
                }

                log_error_with_context(e, "spacy_model_download", context=error_context)

                raise ConfigurationError(
                    f"Failed to download spaCy model '{model_name}'",
                    context=error_context,
                    original_error=e,
                    operation="spacy_model_download",
                )

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
        )

    except Exception as e:
        log_error_with_context(
            e, "spacy_model_ensure", context={"model_name": model_name}
        )

        raise ConfigurationError(
            f"Unexpected error loading spaCy model '{model_name}'",
            context={"model_name": model_name},
            original_error=e,
            operation="spacy_model_ensure",
        )


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


class QdrantConnectionPool:
    """Legacy connection pool for backward compatibility.

    Maintains existing interface while delegating to AsyncQdrantConnectionPool.
    """

    _instance: Optional["QdrantConnectionPool"] = None
    _pool: AsyncQdrantConnectionPool | None = None
    _lock = asyncio.Lock()

    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def configure(self, url: str, max_size: int = 10):
        """Configure the connection pool.

        Args:
            url: Qdrant server URL
            max_size: Maximum number of connections in pool
        """
        self._pool = AsyncQdrantConnectionPool(url, max_size)

    async def get_client(self) -> AsyncQdrantClient:
        """Get client from pool or create new.

        Returns:
            AsyncQdrantClient: Client instance from pool or newly created

        Raises:
            RuntimeError: If pool is not configured
        """
        if self._pool is None:
            raise RuntimeError(
                "Connection pool not configured. Call configure() first."
            )
        return await self._pool.acquire()

    async def return_client(self, client: AsyncQdrantClient):
        """Return client to pool.

        Args:
            client: AsyncQdrantClient to return to pool
        """
        if self._pool is not None:
            await self._pool.release(client)
        else:
            await client.close()

    async def close_all(self):
        """Close all clients in the pool."""
        if self._pool is not None:
            await self._pool.close()


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


class PerformanceMonitor:
    """Monitor async operation performance with detailed metrics.

    Tracks execution time, memory usage, success rates, and provides
    comprehensive performance analytics for optimization.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}

    async def measure_async_operation(self, name: str, operation: Callable) -> Any:
        """Measure and record async operation performance.

        Args:
            name: Name of the operation for metrics tracking
            operation: Async callable to measure

        Returns:
            Result of the operation

        Note:
            Records timing, memory usage, and success/failure status
        """
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        try:
            result = await operation()
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            result = None
            raise  # Re-raise the exception
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()

            self.metrics[name] = {
                "duration": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "success": success,
                "error": error,
                "timestamp": time.time(),
            }

            logger.info(
                f"Performance [{name}]: {end_time - start_time:.2f}s, "
                f"memory: {(end_memory - start_memory) / 1024 / 1024:.1f}MB, "
                f"success: {success}"
            )

        return result

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes.

        Returns:
            Current memory usage in bytes, or 0 if psutil unavailable
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dictionary with aggregated performance statistics
        """
        if not self.metrics:
            return {"total_operations": 0}

        durations = [m["duration"] for m in self.metrics.values()]
        successes = sum(1 for m in self.metrics.values() if m["success"])

        return {
            "total_operations": len(self.metrics),
            "success_rate": successes / len(self.metrics),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations),
            "metrics": self.metrics,
        }


class EnhancedPerformanceMonitor:
    """Enhanced performance monitoring with detailed async operation tracking."""

    def __init__(self):
        self.metrics = {}
        self.operation_counts = {}
        self.error_counts = {}

    @asynccontextmanager
    async def measure(self, operation_name: str):
        """Measure async operation performance."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        # Track operation counts
        self.operation_counts[operation_name] = (
            self.operation_counts.get(operation_name, 0) + 1
        )

        try:
            yield self
        except Exception as e:
            # Track errors
            self.error_counts[operation_name] = (
                self.error_counts.get(operation_name, 0) + 1
            )
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            memory_delta = self._get_memory_usage() - start_memory

            self.metrics[operation_name] = {
                "duration_seconds": elapsed,
                "memory_delta_mb": memory_delta / 1024 / 1024,
                "timestamp": time.time(),
                "success": operation_name not in self.error_counts
                or self.error_counts[operation_name] == 0,
            }

            logger.info(
                f"{operation_name} completed in {elapsed:.2f}s, "
                f"memory delta: {memory_delta / 1024 / 1024:.2f}MB"
            )

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0

    def get_report(self) -> dict:
        """Get comprehensive performance report."""
        total_operations = sum(self.operation_counts.values())
        total_errors = sum(self.error_counts.values())

        return {
            "summary": {
                "total_operations": total_operations,
                "total_errors": total_errors,
                "success_rate": (total_operations - total_errors) / total_operations
                if total_operations > 0
                else 0,
                "total_time": sum(m["duration_seconds"] for m in self.metrics.values()),
                "avg_time": sum(m["duration_seconds"] for m in self.metrics.values())
                / len(self.metrics)
                if self.metrics
                else 0,
            },
            "operation_counts": self.operation_counts,
            "error_counts": self.error_counts,
            "detailed_metrics": self.metrics,
        }

    def log_performance_summary(self):
        """Log a performance summary."""
        report = self.get_report()
        summary = report["summary"]

        logger.info(
            f"Performance Summary - Operations: {summary['total_operations']}, "
            f"Errors: {summary['total_errors']}, "
            f"Success Rate: {summary['success_rate']:.2%}, "
            f"Total Time: {summary['total_time']:.2f}s, "
            f"Avg Time: {summary['avg_time']:.2f}s"
        )
