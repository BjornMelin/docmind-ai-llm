"""Lightweight hardware utilities for DocMind AI.

This module provides minimal hardware detection using PyTorch native APIs only.
Replaces the deprecated utils/hardware_utils.py with KISS-compliant implementation.
"""

from typing import Any

import torch


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
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": "Unknown",
        "vram_total_gb": None,
        "vram_available_gb": None,
        "gpu_compute_capability": None,
        "gpu_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "fastembed_providers": [],
        "cpu_cores": 1,
        "cpu_threads": 1,
    }

    if hardware_info["cuda_available"]:
        device_props = torch.cuda.get_device_properties(0)
        hardware_info["gpu_name"] = device_props.name
        vram_gb = device_props.total_memory / (1024**3)
        hardware_info["vram_total_gb"] = round(vram_gb, 1)

        try:
            vram_available = device_props.total_memory - torch.cuda.memory_allocated(0)
            hardware_info["vram_available_gb"] = round(vram_available / (1024**3), 1)
        except (RuntimeError, OSError):
            # Handle CUDA runtime errors or system-level issues
            hardware_info["vram_available_gb"] = hardware_info["vram_total_gb"]

        hardware_info["gpu_compute_capability"] = (
            device_props.major,
            device_props.minor,
        )
        hardware_info["fastembed_providers"] = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        hardware_info["fastembed_providers"] = ["CPUExecutionProvider"]

    # CPU info
    import os

    hardware_info["cpu_cores"] = os.cpu_count() or 1
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
    else:
        return ["CPUExecutionProvider"]


def get_recommended_batch_size(model_type: str = "embedding") -> int:
    """Get recommended batch size based on available hardware.

    Args:
        model_type: Type of model ("embedding", "llm", "vision")

    Returns:
        Recommended batch size for the model type and available hardware
    """
    if not torch.cuda.is_available():
        # Conservative CPU batch sizes
        return {"embedding": 16, "llm": 1, "vision": 4}.get(model_type, 8)

    try:
        device_props = torch.cuda.get_device_properties(0)
        vram_gb = device_props.total_memory / (1024**3)
    except (RuntimeError, OSError):
        vram_gb = 4  # Fallback assumption for CUDA errors or system issues

    # GPU batch sizes based on available VRAM
    if vram_gb >= 16:  # High-end GPU
        return {"embedding": 128, "llm": 8, "vision": 32}.get(model_type, 64)
    elif vram_gb >= 8:  # Mid-range GPU
        return {"embedding": 64, "llm": 4, "vision": 16}.get(model_type, 32)
    elif vram_gb >= 4:  # Entry-level GPU
        return {"embedding": 32, "llm": 2, "vision": 8}.get(model_type, 16)
    else:  # Low VRAM or unknown
        return {"embedding": 16, "llm": 1, "vision": 4}.get(model_type, 8)
