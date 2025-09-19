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

try:  # Optional torch - allow module import without GPU stack present
    import torch as TORCH  # type: ignore  # noqa: N812
except ImportError:  # pragma: no cover - torch may be unavailable in CI
    TORCH = None  # type: ignore[assignment]
from loguru import logger
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from src.config import settings
from src.config.settings import DocMindSettings


def is_cuda_available() -> bool:
    """Return True when CUDA is available via torch."""
    try:
        return bool(
            TORCH and getattr(TORCH, "cuda", None) and TORCH.cuda.is_available()
        )  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):  # pragma: no cover - conservative
        return False


def _is_mps_available() -> bool:
    """Return True when Apple MPS backend is available."""
    try:
        backends = getattr(TORCH, "backends", None)
        mps = getattr(backends, "mps", None) if backends else None
        return bool(mps) and bool(getattr(mps, "is_available", lambda: False)())
    except (AttributeError, RuntimeError):  # pragma: no cover - conservative
        return False


def get_vram_gb(device_index: int = 0) -> float | None:
    """Return total GPU VRAM in GiB for a CUDA device; else None.

    Args:
        device_index: CUDA device index to query (default: 0).
    """
    if not is_cuda_available():
        return None
    try:
        # type: ignore[attr-defined]
        total_bytes = TORCH.cuda.get_device_properties(int(device_index)).total_memory
        divisor = float(getattr(settings.monitoring, "bytes_to_gb_divisor", 1024**3))
        return round(total_bytes / divisor, 1)
    except (
        OSError,
        AttributeError,
        RuntimeError,
        ValueError,
    ):  # pragma: no cover - conservative
        return None


def resolve_device(prefer: str = "auto") -> tuple[str, int | None]:
    """Resolve a canonical device string and device index when applicable.

    Args:
        prefer: 'auto'|'cpu'|'mps'|'cuda' or a concrete 'cuda:N'.

    Returns:
        Tuple of (device_str, device_index or None for CPU/MPS).
    """
    try:
        p = (prefer or "auto").lower()
        # If explicit cuda:N provided
        if p.startswith("cuda:"):
            try:
                idx = int(p.split(":", 1)[1])
            except (TypeError, ValueError):
                idx = 0
            return (f"cuda:{idx}", idx)
        # Use existing selection logic
        dev = select_device(p)
        if dev == "cuda":
            # Derive an index; prefer current device
            try:
                # type: ignore[attr-defined]
                idx = int(TORCH.cuda.current_device()) if TORCH else 0
            except (AttributeError, RuntimeError, ValueError):
                idx = 0
            return (f"cuda:{idx}", idx)
        if dev == "mps":
            return ("mps", None)
        return ("cpu", None)
    except (AttributeError, ValueError, RuntimeError):
        return ("cpu", None)


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
        cuda_ok = is_cuda_available()
        hardware_info["cuda_available"] = cuda_ok
        if cuda_ok:
            # type: ignore[attr-defined]
            hardware_info["gpu_name"] = TORCH.cuda.get_device_name(0)
            vram = get_vram_gb()
            hardware_info["vram_total_gb"] = vram

    except RuntimeError as e:
        logger.warning("CUDA error during hardware detection: %s", e)
    except (OSError, AttributeError) as e:
        logger.warning("System error during hardware detection: %s", e)
    except (ImportError, ModuleNotFoundError) as e:
        logger.error("Import error during hardware detection: %s", e)

    return hardware_info


def has_cuda_vram(min_gb: float, device_index: int = 0) -> bool:
    """Return True when CUDA is available and total VRAM ≥ ``min_gb`` for a device.

    Args:
        min_gb: Minimum required VRAM (GiB) to consider the device sufficient.
        device_index: CUDA device index to check (default: 0).

    Returns:
        bool: True when CUDA is available and device VRAM meets the threshold;
              False otherwise.
    """
    try:
        if not is_cuda_available():
            return False
        # Compare using raw bytes to avoid rounding edge cases
        # type: ignore[attr-defined]
        props = TORCH.cuda.get_device_properties(int(device_index))
        total_bytes = float(getattr(props, "total_memory", 0.0))
        divisor = float(getattr(settings.monitoring, "bytes_to_gb_divisor", 1024**3))
        required_bytes = float(min_gb) * divisor
        return total_bytes >= required_bytes
    except (RuntimeError, AttributeError, ImportError, TypeError, ValueError):
        return False


def select_device(prefer: str = "auto") -> str:
    """Select an inference device string ('cuda'|'mps'|'cpu').

    Preference order:
    - When ``prefer`` is 'cpu' or 'cuda' or 'mps', honor if available; else fall back.
    - When ``prefer`` is 'auto', choose 'cuda' if available; else 'mps' (Apple Silicon);
      otherwise 'cpu'.

    Returns:
        str: One of 'cuda', 'mps', or 'cpu'.
    """
    p = (prefer or "auto").lower()
    selected = "cpu"
    try:
        if p == "cpu":
            selected = "cpu"
        else:
            cuda_ok = is_cuda_available()
            mps_ok = _is_mps_available()
            if p == "cuda":
                selected = "cuda" if cuda_ok else "cpu"
            elif p == "mps":
                selected = "mps" if mps_ok else ("cuda" if cuda_ok else "cpu")
            else:  # auto
                selected = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    except (RuntimeError, AttributeError, ImportError, TypeError):
        selected = "cpu"
    return selected


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

    # Check Qdrant connectivity (graceful offline)
    try:
        client = qdrant_client.QdrantClient(url=app_settings.database.qdrant_url)
        client.get_collections()
        results["info"].append(
            f"Qdrant connection successful: {app_settings.database.qdrant_url}"
        )
        client.close()
    except (ConnectionError, ResponseHandlingException, UnexpectedResponse) as e:
        results["errors"].append(f"Qdrant connection failed: {e}")
        results["valid"] = False
    except OSError as e:
        results["errors"].append(f"Qdrant network error: {e}")
        results["valid"] = False
    except (RuntimeError, ValueError) as e:  # pragma: no cover - transport-specific
        results["errors"].append(f"Qdrant error: {e}")
        results["valid"] = False

    # Check GPU configuration
    if app_settings.enable_gpu_acceleration:
        try:
            if is_cuda_available():
                # type: ignore[attr-defined]
                gpu_name = TORCH.cuda.get_device_name(0)
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

    # Do not raise: return structured result for callers/tests to handle

    return results


@asynccontextmanager
async def managed_gpu_operation() -> AsyncGenerator[None, None]:
    """Context manager for GPU operations with cleanup."""
    try:
        yield
    finally:
        if is_cuda_available():
            # type: ignore[attr-defined]
            TORCH.cuda.synchronize()
            # type: ignore[attr-defined]
            TORCH.cuda.empty_cache()
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
        client = qdrant_client.AsyncQdrantClient(url=url)
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
