"""Lightweight hardware utilities for DocMind AI.

This module provides minimal hardware detection using PyTorch native APIs only.
Replaces the deprecated utils/hardware_utils.py with KISS-compliant implementation.

Uses the resource management utilities for robust error handling.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Hardware Detection Constants
BYTES_TO_GB_FACTOR = 1024**3
DEFAULT_CPU_CORES = 1
DEFAULT_BATCH_SIZE_FALLBACK = 8
HIGH_END_VRAM_THRESHOLD = 16
MID_RANGE_VRAM_THRESHOLD = 8
ENTRY_LEVEL_VRAM_THRESHOLD = 4

# Batch Size Configuration by Model Type and VRAM
CPU_BATCH_SIZES = {"embedding": 16, "llm": 1, "vision": 4}
HIGH_END_BATCH_SIZES = {"embedding": 128, "llm": 8, "vision": 32}
HIGH_END_DEFAULT_BATCH = 64
MID_RANGE_BATCH_SIZES = {"embedding": 64, "llm": 4, "vision": 16}
MID_RANGE_DEFAULT_BATCH = 32
ENTRY_LEVEL_BATCH_SIZES = {"embedding": 32, "llm": 2, "vision": 8}
ENTRY_LEVEL_DEFAULT_BATCH = 16
LOW_VRAM_BATCH_SIZES = {"embedding": 16, "llm": 1, "vision": 4}
LOW_VRAM_DEFAULT_BATCH = 8


def detect_hardware() -> dict[str, Any]:
    """Detect hardware capabilities using PyTorch native APIs.

    Detects available hardware resources, including GPU and CPU configurations.

    Returns:
        A dictionary containing hardware information with keys:
        - 'cuda_available': Whether CUDA is available
        - 'gpu_name': Name of the GPU device (if available)
        - 'vram_total_gb': Total GPU memory in GB
        - 'vram_available_gb': Available GPU memory in GB
        - 'gpu_compute_capability': GPU compute capability tuple
        - 'gpu_device_count': Number of available GPU devices
        - 'fastembed_providers': Execution providers
        - 'cpu_cores': Number of CPU cores
        - 'cpu_threads': Number of CPU threads
    """
    hardware_info = {
        "cuda_available": False,
        "gpu_name": "Unknown",
        "vram_total_gb": None,
        "vram_available_gb": None,
        "gpu_compute_capability": None,
        "gpu_device_count": 0,
        "fastembed_providers": ["CPUExecutionProvider"],
        "cpu_cores": 1,
        "cpu_threads": 1,
    }

    try:
        hardware_info["cuda_available"] = torch.cuda.is_available()

        if hardware_info["cuda_available"]:
            hardware_info["gpu_device_count"] = torch.cuda.device_count()

            if hardware_info["gpu_device_count"] > 0:
                device_props = torch.cuda.get_device_properties(0)
                hardware_info["gpu_name"] = device_props.name
                vram_gb = device_props.total_memory / BYTES_TO_GB_FACTOR
                hardware_info["vram_total_gb"] = round(vram_gb, 1)

                try:
                    vram_available = (
                        device_props.total_memory - torch.cuda.memory_allocated(0)
                    )
                    hardware_info["vram_available_gb"] = round(
                        vram_available / BYTES_TO_GB_FACTOR, 1
                    )
                except (RuntimeError, OSError) as e:
                    # Handle CUDA runtime errors or system-level issues
                    logger.warning("Failed to get available VRAM: %s", e)
                    hardware_info["vram_available_gb"] = hardware_info["vram_total_gb"]

                hardware_info["gpu_compute_capability"] = (
                    device_props.major,
                    device_props.minor,
                )
                hardware_info["fastembed_providers"] = [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]

    except RuntimeError as e:
        logger.warning("CUDA hardware detection failed: %s", e)
    except (OSError, ImportError, AttributeError) as e:
        logger.error("Unexpected error during hardware detection: %s", e)

    # CPU info
    import os

    hardware_info["cpu_cores"] = os.cpu_count() or DEFAULT_CPU_CORES
    hardware_info["cpu_threads"] = hardware_info["cpu_cores"]

    return hardware_info


def get_optimal_providers(force_cpu: bool = False) -> list[str]:
    """Determine optimal execution providers for model inference.

    Selects the best execution providers based on available hardware,
    with an option to force CPU-only execution.

    Args:
        force_cpu (bool, optional): Force CPU-only execution even if GPU is available.
            Defaults to False.

    Returns:
        list[str]: List of execution providers in order of preference.
            Prioritizes CUDA if available and not force_cpu is set.
    """
    if force_cpu:
        return ["CPUExecutionProvider"]

    if torch.cuda.is_available():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_recommended_batch_size(model_type: str = "embedding") -> int:
    """Get recommended batch size based on available hardware with error handling.

    Args:
        model_type: Type of model ("embedding", "llm", "vision")

    Returns:
        Recommended batch size for the model type and available hardware
    """
    cpu_defaults = CPU_BATCH_SIZES

    try:
        if not torch.cuda.is_available():
            return cpu_defaults.get(model_type, DEFAULT_BATCH_SIZE_FALLBACK)

        device_props = torch.cuda.get_device_properties(0)
        vram_gb = device_props.total_memory / BYTES_TO_GB_FACTOR

        # GPU batch sizes based on available VRAM
        if vram_gb >= HIGH_END_VRAM_THRESHOLD:  # High-end GPU
            return HIGH_END_BATCH_SIZES.get(model_type, HIGH_END_DEFAULT_BATCH)
        if vram_gb >= MID_RANGE_VRAM_THRESHOLD:  # Mid-range GPU
            return MID_RANGE_BATCH_SIZES.get(model_type, MID_RANGE_DEFAULT_BATCH)
        if vram_gb >= ENTRY_LEVEL_VRAM_THRESHOLD:  # Entry-level GPU
            return ENTRY_LEVEL_BATCH_SIZES.get(model_type, ENTRY_LEVEL_DEFAULT_BATCH)
        # Low VRAM case
        return LOW_VRAM_BATCH_SIZES.get(model_type, LOW_VRAM_DEFAULT_BATCH)

    except (RuntimeError, OSError, ImportError, AttributeError, ValueError) as e:
        if isinstance(e, (RuntimeError, OSError)):
            logger.warning("Failed to get GPU properties for batch size: %s", e)
        else:
            logger.error("Unexpected error determining batch size: %s", e)
        return cpu_defaults.get(model_type, DEFAULT_BATCH_SIZE_FALLBACK)
