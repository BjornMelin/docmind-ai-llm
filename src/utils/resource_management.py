"""Resource management utilities for DocMind AI.

Provides reusable context managers and utilities for robust resource cleanup,
especially for GPU/VRAM operations and model lifecycle management.

Critical for ML applications to prevent memory leaks and resource exhaustion.
"""

import asyncio
import gc
import logging
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)


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
    except Exception:
        # Re-raise the exception after cleanup
        raise
    finally:
        # Always cleanup GPU resources
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except RuntimeError as e:
            logger.warning(f"GPU cleanup failed during context exit: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during GPU cleanup: {e}")
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
    except Exception:
        # Re-raise the exception after cleanup
        raise
    finally:
        # Always cleanup GPU resources
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except RuntimeError as e:
            logger.warning(f"GPU cleanup failed during async context exit: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during async GPU cleanup: {e}")
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

    except Exception:
        # Re-raise after cleanup
        raise
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
    except Exception:
        # Re-raise after cleanup
        raise
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
            vram = torch.cuda.memory_allocated() / 1024**3
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
            logger.warning(f"{operation_name} failed with CUDA error: {e}")
        else:
            logger.warning(f"{operation_name} failed with runtime error: {e}")

        if reraise:
            raise
        else:
            result_dict["result"] = default_return
            result_dict["error"] = str(e)

    except (OSError, AttributeError) as e:
        logger.warning(f"{operation_name} failed with system error: {e}")

        if reraise:
            raise
        else:
            result_dict["result"] = default_return
            result_dict["error"] = str(e)

    except Exception as e:
        logger.error(f"{operation_name} failed with unexpected error: {e}")

        if reraise:
            raise
        else:
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
                logger.warning(f"{operation_name} failed with CUDA error: {e}")
            else:
                logger.warning(f"{operation_name} failed with runtime error: {e}")
        return default_return
    except (OSError, AttributeError) as e:
        if log_errors:
            logger.warning(f"{operation_name} failed with system error: {e}")
        return default_return
    except Exception as e:
        if log_errors:
            logger.error(f"{operation_name} failed with unexpected error: {e}")
        return default_return


async def _cleanup_model(model: Any, cleanup_method: str | None) -> None:
    """Internal async cleanup helper for models."""
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
            logger.debug(f"Model has no cleanup method: {cleanup_method}")
    except Exception as e:
        logger.warning(f"Model cleanup failed: {e}")


def _sync_cleanup_model(model: Any, cleanup_method: str | None) -> None:
    """Internal sync cleanup helper for models."""
    if not cleanup_method:
        return

    try:
        if hasattr(model, cleanup_method):
            cleanup_func = getattr(model, cleanup_method)
            cleanup_func()
        else:
            logger.debug(f"Model has no cleanup method: {cleanup_method}")
    except Exception as e:
        logger.warning(f"Model cleanup failed: {e}")


def get_safe_vram_usage() -> float:
    """Get current VRAM usage with comprehensive error handling.

    Provides a safe way to check VRAM usage that won't crash on
    CUDA errors or missing hardware.

    Returns:
        VRAM usage in GB (0.0 if CUDA unavailable or error)
    """
    return safe_cuda_operation(
        lambda: torch.cuda.memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0,
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
        info["cuda_available"] = torch.cuda.is_available()

        if info["cuda_available"]:
            info["device_count"] = safe_cuda_operation(
                torch.cuda.device_count, "device count", 0
            )

            if info["device_count"] > 0:
                info["device_name"] = safe_cuda_operation(
                    lambda: torch.cuda.get_device_name(0), "device name", "Unknown"
                )

                # Get device properties safely
                props = safe_cuda_operation(
                    lambda: torch.cuda.get_device_properties(0),
                    "device properties",
                    None,
                )

                if props:
                    info["compute_capability"] = f"{props.major}.{props.minor}"
                    info["total_memory_gb"] = props.total_memory / 1024**3

                info["allocated_memory_gb"] = safe_cuda_operation(
                    lambda: torch.cuda.memory_allocated(0) / 1024**3,
                    "allocated memory",
                    0.0,
                )

    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")

    return info
