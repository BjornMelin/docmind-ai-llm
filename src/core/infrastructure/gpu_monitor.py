"""PyTorch native GPU monitoring for DocMind AI.

This module provides lightweight GPU monitoring using only torch.cuda native APIs.
Implements GPUMetrics dataclass and async context manager for performance monitoring
following KISS principles with <25 lines of clean code.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GPUMetrics:
    """Dataclass representing GPU performance metrics.

    Attributes:
        device_name (str): The name of the GPU device.
        memory_allocated_gb (float): Memory allocated by PyTorch, in gigabytes.
        memory_reserved_gb (float): Total memory reserved by the GPU, in gigabytes.
        utilization_percent (float): GPU memory utilization percentage (0-100).
    """

    device_name: str
    memory_allocated_gb: float
    memory_reserved_gb: float
    utilization_percent: float


@asynccontextmanager
async def gpu_performance_monitor() -> AsyncGenerator[GPUMetrics | None, None]:
    """Async context manager for monitoring GPU performance.

    Returns:
        An async generator yielding GPUMetrics if a CUDA device is available,
        or None if no CUDA device is present.

    Example:
        async with gpu_performance_monitor() as metrics:
            if metrics:
                print(f"GPU Utilization: {metrics.utilization_percent}%")
    """
    if not torch.cuda.is_available():
        yield None
        return

    current_device = torch.cuda.current_device()
    device_props = torch.cuda.get_device_properties(current_device)
    allocated = torch.cuda.memory_allocated(current_device) / 1024**3
    reserved = torch.cuda.memory_reserved(current_device) / 1024**3
    utilization = min((allocated / (device_props.total_memory / 1024**3)) * 100, 100.0)

    yield GPUMetrics(device_props.name, allocated, reserved, utilization)
