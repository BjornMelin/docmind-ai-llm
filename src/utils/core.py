"""Core utilities for DocMind AI - essential functions only.

This module provides the most essential utilities needed by the application:
- Hardware detection for GPU acceleration
- Startup configuration validation
- Basic context managers for resource management
- Performance timing utilities
"""

import gc
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any

import qdrant_client
import torch
from loguru import logger

from src.config import settings
from src.config.settings import DocMindSettings

# Compatibility alias so tests can patch src.utils.core.AsyncQdrantClient
AsyncQdrantClient = qdrant_client.AsyncQdrantClient


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
                / settings.monitoring.bytes_to_gb_divisor
            )
            hardware_info["vram_total_gb"] = round(vram_gb, 1)

    except RuntimeError as e:
        logger.warning("CUDA error during hardware detection: %s", e)
    except (OSError, AttributeError) as e:
        logger.warning("System error during hardware detection: %s", e)
    except (ImportError, ModuleNotFoundError) as e:
        logger.error("Import error during hardware detection: %s", e)

    return hardware_info


def validate_startup_configuration(app_settings: DocMindSettings) -> dict[str, Any]:
    """Validate critical startup configuration.

    Args:
        app_settings: Application settings to validate

    Returns:
        Dict with validation results and any errors

    Raises:
        RuntimeError: If critical configuration errors found
    """
    results = {"valid": True, "warnings": [], "errors": [], "info": []}

    # Check Qdrant connectivity
    try:
        client = qdrant_client.QdrantClient(url=app_settings.database.qdrant_url)
        client.get_collections()
        results["info"].append(
            f"Qdrant connection successful: {app_settings.database.qdrant_url}"
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

    # Configuration validation complete

    if not results["valid"]:
        error_msg = "Critical configuration errors:\n" + "\n".join(results["errors"])
        raise RuntimeError(error_msg)

    return results


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
) -> AsyncGenerator[object, None]:
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
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info("%s completed in %.2fs", func.__name__, duration)

    return wrapper
