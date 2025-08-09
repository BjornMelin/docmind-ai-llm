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
import logging
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
from utils.model_manager import ModelManager

settings = AppSettings()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application logging with console and file handlers.

    Sets up structured logging with timestamped messages, appropriate
    formatting, and dual output to both console and log file for
    comprehensive debugging and monitoring.

    Args:
        log_level: Logging level as string. Must be one of 'DEBUG', 'INFO',
            'WARNING', 'ERROR', or 'CRITICAL'. Defaults to 'INFO'.

    Note:
        Creates a log file named 'docmind.log' in the current directory.
        Both console and file handlers use the same formatting pattern
        for consistency.

    Example:
        >>> setup_logging("DEBUG")
        >>> import logging
        >>> logging.info("Application started")
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("docmind.log")],
    )


def detect_hardware() -> dict[str, Any]:
    """Detect hardware capabilities using FastEmbed native detection.

    Performs comprehensive hardware detection including CUDA availability,
    GPU specifications, and FastEmbed execution providers. Uses the model
    manager singleton pattern to prevent redundant model initializations
    and optimize resource usage.

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
    hardware_info = {
        "cuda_available": False,
        "gpu_name": "Unknown",
        "vram_total_gb": None,
        "fastembed_providers": [],
    }

    # Use FastEmbed's native hardware detection
    try:
        # FastEmbed automatically detects available providers
        test_model = ModelManager.get_text_embedding_model("BAAI/bge-small-en-v1.5")

        # Get detected providers from FastEmbed
        try:
            providers = test_model.model.model.get_providers()
            hardware_info["fastembed_providers"] = providers
            hardware_info["cuda_available"] = "CUDAExecutionProvider" in providers
            logging.info("FastEmbed detected providers: %s", providers)
        except (AttributeError, RuntimeError, ImportError):
            # Fallback detection
            hardware_info["cuda_available"] = torch.cuda.is_available()

        # Basic GPU info if available
        if hardware_info["cuda_available"] and torch.cuda.is_available():
            try:
                hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                hardware_info["vram_total_gb"] = round(vram_gb, 1)
            except (RuntimeError, AttributeError) as e:
                logging.warning("GPU info detection failed: %s", e)

        del test_model  # Cleanup

    except (ImportError, RuntimeError, AttributeError) as e:
        logging.warning("FastEmbed hardware detection failed: %s", e)
        # Ultimate fallback
        hardware_info["cuda_available"] = torch.cuda.is_available()

    return hardware_info


def get_embed_model() -> FastEmbedEmbedding:
    """Create optimized embedding model with GPU acceleration.

    Initializes a FastEmbedEmbedding model with optimal configuration for
    both CPU and GPU environments. Applies torch.compile optimization when
    available and uses research-backed model settings for best performance.

    GPU optimizations include:
    - torch.compile with 'reduce-overhead' mode
    - CUDA execution provider prioritization
    - Automatic batch size configuration
    - Mixed precision support

    Returns:
        FastEmbedEmbedding: Fully configured and optimized embedding model.

    Note:
        The model is configured with:
        - Model: BGE-Large (BAAI/bge-large-en-v1.5) from settings
        - Max length: 512 tokens for optimal performance
        - Cache directory: './embeddings_cache'
        - Dynamic batching for variable input sizes

    Raises:
        Exception: If model initialization fails.

    Example:
        >>> embed_model = get_embed_model()
        >>> embeddings = embed_model.get_text_embedding("Hello world")
        >>> print(f"Embedding dimension: {len(embeddings)}")
    """
    # Initialize embedding model with optimal configuration
    embed_model = FastEmbedEmbedding(
        model_name=settings.dense_embedding_model,
        max_length=512,
        cache_dir="./embeddings_cache",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        if settings.gpu_acceleration and torch.cuda.is_available()
        else ["CPUExecutionProvider"],
        batch_size=settings.embedding_batch_size,
    )

    # Apply GPU optimizations if available
    if settings.gpu_acceleration and torch.cuda.is_available():
        # Enable mixed precision and compilation
        if hasattr(torch, "compile"):
            embed_model = torch.compile(
                embed_model,
                mode="reduce-overhead",  # Best for embedding models
                dynamic=True,  # Handle variable batch sizes
            )
            logging.info(
                "GPU: torch.compile enabled for embeddings with reduce-overhead mode"
            )

        # Log GPU configuration
        try:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(
                "GPU: Using %s with %.1fGB VRAM for embeddings", gpu_name, vram_gb
            )
        except (RuntimeError, AttributeError) as e:
            logging.warning("GPU info detection failed: %s", e)
    else:
        logging.info("Using CPU mode for embeddings")

    return embed_model


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

    logging.info("RRF Configuration Verification: %s", verification)
    return verification


def ensure_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Ensure spaCy model is available, download if needed.

    Attempts to load a spaCy model and automatically downloads it if not
    found locally. Provides robust error handling and informative logging
    throughout the process.

    Args:
        model_name: Name of the spaCy model to load. Common options include:
            - 'en_core_web_sm': Small English model (~15MB)
            - 'en_core_web_md': Medium English model (~40MB)
            - 'en_core_web_lg': Large English model (~560MB)
            Defaults to 'en_core_web_sm'.

    Returns:
        Loaded spaCy Language model instance ready for NLP processing.

    Raises:
        RuntimeError: If the model cannot be loaded or downloaded, or if
            spaCy is not installed.

    Note:
        Downloads can take several minutes depending on model size and
        network speed. The function handles subprocess execution for
        model downloads and provides detailed logging throughout.

    Example:
        >>> nlp = ensure_spacy_model("en_core_web_sm")
        >>> doc = nlp("This is a test sentence.")
        >>> entities = [(ent.text, ent.label_) for ent in doc.ents]
        >>> print(f"Found entities: {entities}")
    """
    try:
        import spacy

        nlp = spacy.load(model_name)
        logging.info("spaCy model '%s' loaded successfully", model_name)
        return nlp
    except OSError:
        # Try to download the model
        try:
            logging.info("Downloading spaCy model '%s'...", model_name)
            subprocess.run(
                ["python", "-m", "spacy", "download", model_name],
                check=True,
                capture_output=True,
                text=True,
            )
            import spacy

            nlp = spacy.load(model_name)
            logging.info(
                "spaCy model '%s' downloaded and loaded successfully", model_name
            )
            return nlp
        except (subprocess.CalledProcessError, OSError) as e:
            error_msg = f"Failed to load or download spaCy model '{model_name}': {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    except ImportError as e:
        error_msg = f"spaCy is not installed: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


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
            logging.info(f"{func.__name__} completed in {duration:.2f}s")

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

            logging.info(
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
