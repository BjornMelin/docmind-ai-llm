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
    """GPU metrics using PyTorch native APIs."""

    device_name: str
    memory_allocated_gb: float
    memory_reserved_gb: float
    utilization_percent: float


@asynccontextmanager
async def gpu_performance_monitor() -> AsyncGenerator[GPUMetrics | None, None]:
    """Async context manager for GPU performance monitoring."""
    if not torch.cuda.is_available():
        yield None
        return

    device_props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    utilization = min((allocated / (device_props.total_memory / 1024**3)) * 100, 100.0)

    yield GPUMetrics(device_props.name, allocated, reserved, utilization)
