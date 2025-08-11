"""Memory monitoring and management utilities for DocMind AI.

This module provides comprehensive memory management capabilities including
memory usage tracking, GPU memory management, and context managers for
resource-efficient operations. Consolidates memory-related functionality
to follow DRY principles and provide consistent memory management patterns.

Key features:
- Memory usage monitoring with detailed statistics
- GPU memory management and cleanup utilities
- Context managers for memory-efficient processing
- Automatic garbage collection integration
- Resource management for embedding models and database connections

Example:
    Memory-efficient processing with monitoring::

        from utils.memory_utils import MemoryMonitor, managed_gpu_operation
        import asyncio

        monitor = MemoryMonitor()

        # Monitor memory usage during operation
        with monitor.memory_managed_processing("model_inference"):
            result = run_expensive_operation()

        # GPU memory management
        async with managed_gpu_operation():
            gpu_result = await gpu_intensive_task()
"""

import gc
import time
from contextlib import asynccontextmanager, contextmanager

import psutil
import torch
from loguru import logger
from qdrant_client import AsyncQdrantClient

from .logging_utils import log_performance


class MemoryMonitor:
    """Enhanced memory usage monitoring and optimization using psutil 7.0.0.

    Provides comprehensive memory tracking and management capabilities for
    both system and GPU memory. Now uses psutil Process.oneshot() for efficient
    multi-attribute access and enhanced memory analysis including USS/PSS metrics.

    This class consolidates memory monitoring functionality and provides
    library-first memory management using modern psutil features.
    """

    def __init__(self):
        """Initialize memory monitor with psutil process handle."""
        self._process = psutil.Process()

    def get_memory_usage(self) -> dict[str, float]:
        """Get comprehensive memory usage statistics using psutil Process.oneshot().

        Returns:
            Dictionary containing detailed memory usage information:
            - 'rss_mb': Resident Set Size in megabytes
            - 'vms_mb': Virtual Memory Size in megabytes
            - 'uss_mb': Unique Set Size in megabytes (real memory usage)
            - 'pss_mb': Proportional Set Size in megabytes
            - 'percent': Memory usage as percentage of system total
            - 'available_mb': Available system memory in megabytes
            - 'total_mb': Total system memory in megabytes
            - 'num_fds': Number of open file descriptors (Linux/macOS)
            - 'num_threads': Number of threads in process
        """
        # Use oneshot context for efficient multi-attribute access
        with self._process.oneshot():
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()
            num_threads = self._process.num_threads()

            # Get enhanced memory info if available
            try:
                memory_full_info = self._process.memory_full_info()
                uss_mb = (
                    memory_full_info.uss / 1024 / 1024
                    if hasattr(memory_full_info, "uss")
                    else 0
                )
                pss_mb = (
                    memory_full_info.pss / 1024 / 1024
                    if hasattr(memory_full_info, "pss")
                    else 0
                )
            except (psutil.AccessDenied, AttributeError):
                uss_mb = 0
                pss_mb = 0

            # Get file descriptors count (Linux/macOS)
            try:
                num_fds = self._process.num_fds()
            except (psutil.AccessDenied, AttributeError):
                num_fds = 0

        # Get system memory info
        system_memory = psutil.virtual_memory()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "uss_mb": uss_mb,  # Real memory usage
            "pss_mb": pss_mb,  # Proportional shared memory
            "percent": memory_percent,
            "available_mb": system_memory.available / 1024 / 1024,
            "total_mb": system_memory.total / 1024 / 1024,
            "num_fds": num_fds,
            "num_threads": num_threads,
        }

    @staticmethod
    def get_gpu_memory_usage() -> dict[str, float]:
        """Get current GPU memory usage statistics.

        Returns:
            Dictionary containing GPU memory information:
            - 'allocated_mb': Currently allocated GPU memory in MB
            - 'reserved_mb': Reserved GPU memory in MB
            - 'free_mb': Free GPU memory in MB
            - 'total_mb': Total GPU memory in MB
            - 'utilization_percent': GPU memory utilization percentage

        Note:
            Returns empty dict if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return {}

        try:
            allocated_bytes = torch.cuda.memory_allocated()
            reserved_bytes = torch.cuda.memory_reserved()
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            free_bytes = total_bytes - reserved_bytes

            return {
                "allocated_mb": allocated_bytes / 1024 / 1024,
                "reserved_mb": reserved_bytes / 1024 / 1024,
                "free_mb": free_bytes / 1024 / 1024,
                "total_mb": total_bytes / 1024 / 1024,
                "utilization_percent": (reserved_bytes / total_bytes) * 100,
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory usage: {e}")
            return {}

    @contextmanager
    def memory_managed_processing(self, operation: str):
        """Context manager for memory-efficient processing with psutil monitoring.

        Args:
            operation: Name of the operation for logging purposes

        Yields:
            None - Context for executing memory-managed operations

        Note:
            - Enhanced with psutil Process.oneshot() for efficient monitoring
            - Tracks comprehensive memory metrics including USS/PSS
            - Forces garbage collection after processing
            - Logs detailed memory and system statistics
            - Provides automatic resource cleanup
        """
        initial_memory = self.get_memory_usage()
        initial_gpu = self.get_gpu_memory_usage()

        # Capture initial system state with psutil
        with self._process.oneshot():
            initial_cpu_times = self._process.cpu_times()
            try:
                initial_io = self._process.io_counters()
            except (psutil.AccessDenied, AttributeError):
                initial_io = None
            try:
                initial_ctx_switches = self._process.num_ctx_switches()
            except (psutil.AccessDenied, AttributeError):
                initial_ctx_switches = None

        logger.info(
            f"Starting {operation}",
            extra={
                "memory_rss_mb": initial_memory["rss_mb"],
                "memory_uss_mb": initial_memory["uss_mb"],  # Real memory usage
                "memory_percent": initial_memory["percent"],
                "num_threads": initial_memory["num_threads"],
                "num_fds": initial_memory["num_fds"],
                "gpu_allocated_mb": initial_gpu.get("allocated_mb", 0),
            },
        )

        start_time = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        end_time = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        start_perf_time = time.perf_counter()

        if start_time:
            start_time.record()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_perf_time

            if end_time:
                end_time.record()
                torch.cuda.synchronize()

            # Force garbage collection for memory cleanup
            collected = gc.collect()

            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            final_memory = self.get_memory_usage()
            final_gpu = self.get_gpu_memory_usage()

            # Capture final system state with psutil
            with self._process.oneshot():
                final_cpu_times = self._process.cpu_times()
                try:
                    final_io = self._process.io_counters()
                except (psutil.AccessDenied, AttributeError):
                    final_io = None
                try:
                    final_ctx_switches = self._process.num_ctx_switches()
                except (psutil.AccessDenied, AttributeError):
                    final_ctx_switches = None

            # Calculate comprehensive deltas
            memory_delta_rss = final_memory["rss_mb"] - initial_memory["rss_mb"]
            memory_delta_uss = (
                final_memory["uss_mb"] - initial_memory["uss_mb"]
            )  # Real usage
            gpu_delta = final_gpu.get("allocated_mb", 0) - initial_gpu.get(
                "allocated_mb", 0
            )
            threads_delta = final_memory["num_threads"] - initial_memory["num_threads"]
            fds_delta = final_memory["num_fds"] - initial_memory["num_fds"]

            # CPU times delta
            cpu_user_delta = final_cpu_times.user - initial_cpu_times.user
            cpu_system_delta = final_cpu_times.system - initial_cpu_times.system

            # I/O deltas
            io_read_delta = 0
            io_write_delta = 0
            if initial_io and final_io:
                io_read_delta = (
                    (final_io.read_bytes - initial_io.read_bytes) / 1024 / 1024
                )  # MB
                io_write_delta = (
                    (final_io.write_bytes - initial_io.write_bytes) / 1024 / 1024
                )  # MB

            # Context switches delta
            ctx_switches_delta = 0
            if initial_ctx_switches and final_ctx_switches:
                ctx_switches_delta = (
                    final_ctx_switches.voluntary + final_ctx_switches.involuntary
                ) - (initial_ctx_switches.voluntary + initial_ctx_switches.involuntary)

            # Calculate GPU time if available
            gpu_time_ms = None
            if start_time and end_time:
                gpu_time_ms = start_time.elapsed_time(end_time)

            logger.info(
                f"Completed {operation}",
                extra={
                    "duration_seconds": duration,
                    "memory_final_rss_mb": final_memory["rss_mb"],
                    "memory_final_uss_mb": final_memory["uss_mb"],
                    "memory_delta_rss_mb": memory_delta_rss,
                    "memory_delta_uss_mb": memory_delta_uss,  # Real memory change
                    "gpu_allocated_mb": final_gpu.get("allocated_mb", 0),
                    "gpu_delta_mb": gpu_delta,
                    "cpu_user_time": cpu_user_delta,
                    "cpu_system_time": cpu_system_delta,
                    "io_read_mb": io_read_delta,
                    "io_write_mb": io_write_delta,
                    "ctx_switches": ctx_switches_delta,
                    "threads_delta": threads_delta,
                    "fds_delta": fds_delta,
                    "gc_collected": collected,
                    "gpu_time_ms": gpu_time_ms,
                },
            )

            # Enhanced performance logging with comprehensive metrics
            log_performance(
                f"{operation}_memory_managed",
                duration,
                memory_delta_mb=memory_delta_uss,  # Use real memory usage
                memory_rss_delta_mb=memory_delta_rss,
                gpu_delta_mb=gpu_delta,
                cpu_user_time=cpu_user_delta,
                cpu_system_time=cpu_system_delta,
                io_read_mb=io_read_delta,
                io_write_mb=io_write_delta,
                ctx_switches=ctx_switches_delta,
                gc_collected=collected,
                gpu_time_ms=gpu_time_ms,
            )


@asynccontextmanager
async def managed_gpu_operation():
    """Context manager for GPU operations with automatic cleanup.

    Provides proper GPU memory management during operations, ensuring
    CUDA cache is cleared and garbage collection is performed to prevent
    memory leaks. Essential for long-running applications with GPU operations.

    Usage:
        async with managed_gpu_operation():
            # GPU-intensive operations here
            result = await gpu_model.process(data)

    Note:
        - Synchronizes GPU operations before cleanup
        - Clears CUDA cache to free unused memory
        - Performs garbage collection for complete cleanup
        - Safe to use even when CUDA is not available
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
    """Context manager for AsyncQdrantClient with automatic cleanup.

    Args:
        url: Qdrant server URL

    Yields:
        AsyncQdrantClient: Properly managed client instance

    Usage:
        async with managed_async_qdrant_client(url) as client:
            collections = await client.get_collections()

    Note:
        - Ensures client is properly closed even on exceptions
        - Prevents connection leaks in long-running applications
        - Handles client lifecycle management automatically
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
    """Context manager for embedding models with automatic cleanup.

    Ensures embedding models are properly cleaned up after use, including
    GPU memory cleanup for CUDA models. Essential for preventing memory
    leaks in applications that create multiple embedding model instances.

    Args:
        model_class: The embedding model class to instantiate
        model_kwargs: Keyword arguments for model initialization

    Yields:
        Properly managed embedding model instance

    Usage:
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        kwargs = {"model_name": "BAAI/bge-base-en", "cache_dir": "./cache"}
        async with managed_embedding_model(FastEmbedEmbedding, kwargs) as model:
            embeddings = await model.embed(texts)

    Note:
        - Cleans up model resources including GPU memory
        - Handles model destruction and cache clearing
        - Performs garbage collection after cleanup
        - Safe for both CPU and GPU models
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


# Library-first approach: Removed duplicate connection pooling implementation
# AsyncQdrantClient handles connection management internally via httpx


def force_memory_cleanup():
    """Force comprehensive memory cleanup with detailed psutil monitoring.

    Performs aggressive memory cleanup including garbage collection
    and GPU cache clearing. Enhanced with detailed psutil metrics
    for comprehensive cleanup analysis.

    Note:
        - Calls garbage collection multiple times for thorough cleanup
        - Clears GPU cache if CUDA is available
        - Uses psutil for detailed memory analysis including USS/PSS
        - Logs comprehensive memory usage before and after cleanup
        - Can cause brief performance impact due to cleanup operations
    """
    monitor = MemoryMonitor()
    initial_memory = monitor.get_memory_usage()
    initial_gpu = monitor.get_gpu_memory_usage()

    # Capture initial system memory state
    initial_system_mem = psutil.virtual_memory()
    start_time = time.perf_counter()

    # Multiple GC passes for thorough cleanup
    collected_total = 0
    for i in range(3):
        collected = gc.collect()
        collected_total += collected
        logger.debug(f"GC pass {i + 1}: collected {collected} objects")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Allow system to settle
    time.sleep(0.1)

    cleanup_duration = time.perf_counter() - start_time
    final_memory = monitor.get_memory_usage()
    final_gpu = monitor.get_gpu_memory_usage()
    final_system_mem = psutil.virtual_memory()

    # Calculate comprehensive deltas
    memory_freed_rss = initial_memory["rss_mb"] - final_memory["rss_mb"]
    memory_freed_uss = (
        initial_memory["uss_mb"] - final_memory["uss_mb"]
    )  # Real memory freed
    gpu_freed = initial_gpu.get("allocated_mb", 0) - final_gpu.get("allocated_mb", 0)
    system_mem_freed = (
        (initial_system_mem.available - final_system_mem.available) / 1024 / 1024
    )
    threads_freed = initial_memory["num_threads"] - final_memory["num_threads"]
    fds_freed = initial_memory["num_fds"] - final_memory["num_fds"]

    logger.info(
        f"Memory cleanup completed in {cleanup_duration:.3f}s: "
        f"freed {memory_freed_uss:.1f}MB real memory ({memory_freed_rss:.1f}MB RSS), "
        f"GPU: {gpu_freed:.1f}MB, collected {collected_total} objects, "
        f"threads: {threads_freed}, FDs: {fds_freed}"
    )

    return {
        "cleanup_duration_seconds": cleanup_duration,
        "memory_freed_rss_mb": memory_freed_rss,
        "memory_freed_uss_mb": memory_freed_uss,  # Real memory freed
        "gpu_freed_mb": gpu_freed,
        "system_memory_freed_mb": system_mem_freed,
        "objects_collected": collected_total,
        "threads_freed": threads_freed,
        "fds_freed": fds_freed,
        "initial_memory_mb": initial_memory["rss_mb"],
        "final_memory_mb": final_memory["rss_mb"],
        "initial_uss_mb": initial_memory["uss_mb"],
        "final_uss_mb": final_memory["uss_mb"],
    }
