"""Core utilities for DocMind AI - essential functions only.

This module provides the most essential utilities needed by the application:
- Hardware detection for GPU acceleration
- Startup configuration validation
- Basic context managers for resource management
- Performance timing utilities

Follows KISS principle with minimal dependencies and no complex patterns.
"""

import gc
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any

import torch
from loguru import logger
from qdrant_client import AsyncQdrantClient

from src.config.app_settings import DocMindSettings, app_settings

# Constants for validation
WEIGHT_TOLERANCE = 0.05
RRF_ALPHA_MIN = 10
RRF_ALPHA_MAX = 100


def detect_hardware() -> dict[str, Any]:
    """Detect basic hardware capabilities for GPU acceleration.

    Returns:
        Dictionary with cuda_available, gpu_name, vram_total_gb
    """
    hardware_info = {
        "cuda_available": False,
        "gpu_name": "Unknown",
        "vram_total_gb": None,
    }

    try:
        hardware_info["cuda_available"] = torch.cuda.is_available()

        if hardware_info["cuda_available"]:
            hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_gb = (
                torch.cuda.get_device_properties(0).total_memory
                / app_settings.bytes_to_gb_divisor
            )
            hardware_info["vram_total_gb"] = round(vram_gb, 1)

    except RuntimeError as e:
        logger.warning("CUDA error during hardware detection: %s", e)
    except (OSError, AttributeError) as e:
        logger.warning("System error during hardware detection: %s", e)
    except (ImportError, ModuleNotFoundError) as e:
        logger.error("Import error during hardware detection: %s", e)

    return hardware_info


def validate_startup_configuration(settings: DocMindSettings) -> dict[str, Any]:
    """Validate critical startup configuration.

    Args:
        settings: Application settings to validate

    Returns:
        Dict with validation results and any errors

    Raises:
        RuntimeError: If critical configuration errors found
    """
    results = {"valid": True, "warnings": [], "errors": [], "info": []}

    # Check Qdrant connectivity
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=app_settings.qdrant_url)
        client.get_collections()
        results["info"].append(
            f"Qdrant connection successful: {app_settings.qdrant_url}"
        )
        client.close()
    except ConnectionError as e:
        results["errors"].append(f"Qdrant connection failed: {e}")
        results["valid"] = False
    except OSError as e:
        results["errors"].append(f"Qdrant network error: {e}")
        results["valid"] = False

    # Check GPU configuration
    if app_settings.enable_gpu_acceleration:
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                results["info"].append(f"GPU available: {gpu_name}")
            else:
                results["warnings"].append(
                    "GPU acceleration enabled but no GPU available"
                )
        except RuntimeError as e:
            results["warnings"].append(f"CUDA error during GPU detection: {e}")
        except (ImportError, ModuleNotFoundError) as e:
            results["warnings"].append(f"Import error during GPU detection: {e}")

    # Check chunk configuration
    if app_settings.chunk_overlap >= app_settings.chunk_size:
        results["errors"].append(
            f"Chunk overlap ({app_settings.chunk_overlap}) must be less than "
            f"chunk size ({app_settings.chunk_size})"
        )
        results["valid"] = False

    # Check RRF configuration if sparse embeddings enabled
    if app_settings.use_sparse_embeddings and not (
        RRF_ALPHA_MIN <= app_settings.rrf_fusion_alpha <= RRF_ALPHA_MAX
    ):
        results["warnings"].append(
            f"RRF alpha {app_settings.rrf_fusion_alpha} outside optimal range "
            f"[{RRF_ALPHA_MIN}, {RRF_ALPHA_MAX}]"
        )

    if not results["valid"]:
        error_msg = "Critical configuration errors:\n" + "\n".join(results["errors"])
        raise RuntimeError(error_msg)

    return results


def verify_rrf_configuration(settings: DocMindSettings) -> dict[str, Any]:
    """Verify RRF configuration against research recommendations.

    Args:
        settings: Application settings containing RRF configuration

    Returns:
        Dictionary with verification results and recommendations
    """
    verification = {
        "weights_correct": False,
        "alpha_in_range": False,
        "computed_hybrid_alpha": 0.0,
        "issues": [],
        "recommendations": [],
    }

    # Check research-backed weights (0.7 dense, 0.3 sparse)
    expected_dense = app_settings.rrf_fusion_weight_dense
    expected_sparse = app_settings.rrf_fusion_weight_sparse

    if (
        abs(app_settings.rrf_fusion_weight_dense - expected_dense) < WEIGHT_TOLERANCE
        and abs(app_settings.rrf_fusion_weight_sparse - expected_sparse)
        < WEIGHT_TOLERANCE
    ):
        verification["weights_correct"] = True
    else:
        verification["issues"].append(
            f"Weights not research-backed: dense={app_settings.rrf_fusion_weight_dense}, "
            f"sparse={app_settings.rrf_fusion_weight_sparse} (expected 0.7/0.3)"
        )
        verification["recommendations"].append(
            "Update weights to research-backed values: dense=0.7, sparse=0.3"
        )

    # Check RRF alpha parameter
    if RRF_ALPHA_MIN <= app_settings.rrf_fusion_alpha <= RRF_ALPHA_MAX:
        verification["alpha_in_range"] = True
    else:
        verification["issues"].append(
            f"RRF alpha ({app_settings.rrf_fusion_alpha}) outside research range (10-100)"
        )
        verification["recommendations"].append(
            f"Set RRF alpha between {RRF_ALPHA_MIN}-{RRF_ALPHA_MAX}, "
            f"with {app_settings.rrf_k_constant} as optimal"
        )

    # Calculate hybrid alpha for LlamaIndex
    verification["computed_hybrid_alpha"] = app_settings.rrf_fusion_weight_dense / (
        app_settings.rrf_fusion_weight_dense + app_settings.rrf_fusion_weight_sparse
    )

    return verification


@asynccontextmanager
async def managed_gpu_operation() -> AsyncGenerator[None, None]:
    """Context manager for GPU operations with cleanup."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()


@asynccontextmanager
async def managed_async_qdrant_client(
    url: str,
) -> AsyncGenerator[AsyncQdrantClient, None]:
    """Context manager for AsyncQdrantClient with proper cleanup.

    Args:
        url: Qdrant server URL

    Yields:
        AsyncQdrantClient: Properly managed client instance
    """
    client = None
    try:
        client = AsyncQdrantClient(url=url)
        yield client
    finally:
        if client is not None:
            await client.close()


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
            logger.info("%s completed in %.2fs", func.__name__, duration)

    return wrapper
