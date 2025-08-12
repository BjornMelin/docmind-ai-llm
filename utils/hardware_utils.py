"""Hardware detection and GPU utilities for DocMind AI.

This module provides comprehensive hardware detection capabilities including
CUDA availability, GPU specifications, and FastEmbed execution providers.
Consolidates hardware-related functionality to follow DRY principles and
provide consistent hardware detection across the application.

Key features:
- CUDA and GPU detection with detailed specifications
- FastEmbed execution provider detection
- Hardware capability assessment for optimal model configuration
- Performance-optimized detection with caching
- Graceful fallbacks for unsupported hardware

Example:
    Basic hardware detection::

        from utils.hardware_utils import detect_hardware, get_optimal_providers

        # Get comprehensive hardware information
        hardware = detect_hardware()
        print(f"CUDA available: {hardware['cuda_available']}")
        print(f"GPU: {hardware['gpu_name']} with {hardware['vram_total_gb']}GB")

        # Get optimal execution providers for models
        providers = get_optimal_providers()
        print(f"Available providers: {providers}")
"""

import time
from typing import Any

import torch
from loguru import logger

from .logging_utils import log_performance
from .retry_utils import safe_execute


def detect_hardware() -> dict[str, Any]:
    """Detect hardware capabilities using comprehensive detection methods.

    Performs thorough hardware detection including CUDA availability,
    GPU specifications, and FastEmbed execution providers. Uses safe
    execution patterns to prevent crashes from hardware detection failures
    and provides graceful fallbacks when detection methods are unavailable.

    Returns:
        Dictionary containing comprehensive hardware information:
        - 'cuda_available' (bool): Whether CUDA is available
        - 'gpu_name' (str): Name of the primary GPU or 'Unknown'
        - 'vram_total_gb' (float | None): Total VRAM in GB
        - 'vram_available_gb' (float | None): Available VRAM in GB
        - 'gpu_compute_capability' (tuple | None): GPU compute capability
        - 'gpu_device_count' (int): Number of GPU devices
        - 'fastembed_providers' (list[str]): Available FastEmbed providers
        - 'cpu_cores' (int): Number of CPU cores
        - 'cpu_threads' (int): Number of CPU threads

    Note:
        Falls back gracefully if any detection method fails. Always returns
        a dictionary with all expected keys, using safe defaults when
        detection fails. Hardware information is logged for debugging.

    Example:
        >>> hardware = detect_hardware()
        >>> if hardware['cuda_available']:
        ...     print(f"GPU: {hardware['gpu_name']} with {hardware['vram_total_gb']}GB")
        ...     print(f"Compute capability: {hardware['gpu_compute_capability']}")
        ... else:
        ...     print(f"Running on CPU with {hardware['cpu_cores']} cores")
    """
    start_time = time.perf_counter()

    hardware_info = {
        "cuda_available": False,
        "gpu_name": "Unknown",
        "vram_total_gb": None,
        "vram_available_gb": None,
        "gpu_compute_capability": None,
        "gpu_device_count": 0,
        "fastembed_providers": [],
        "cpu_cores": 1,
        "cpu_threads": 1,
    }

    # Detect CPU information
    def detect_cpu_info():
        import os

        hardware_info["cpu_cores"] = os.cpu_count() or 1
        try:
            import psutil

            hardware_info["cpu_threads"] = psutil.cpu_count(logical=True) or 1
        except ImportError:
            hardware_info["cpu_threads"] = hardware_info["cpu_cores"]

    safe_execute(
        detect_cpu_info,
        default_value=None,
        operation_name="cpu_detection",
    )

    # Basic CUDA detection
    hardware_info["cuda_available"] = torch.cuda.is_available()
    hardware_info["gpu_device_count"] = (
        torch.cuda.device_count() if hardware_info["cuda_available"] else 0
    )

    # Detailed GPU information with error handling
    if hardware_info["cuda_available"] and torch.cuda.is_available():

        def get_detailed_gpu_info():
            device_props = torch.cuda.get_device_properties(0)
            hardware_info["gpu_name"] = device_props.name
            vram_gb = device_props.total_memory / (1024**3)
            hardware_info["vram_total_gb"] = round(vram_gb, 1)

            # Get available VRAM
            try:
                vram_available = torch.cuda.get_device_properties(
                    0
                ).total_memory - torch.cuda.memory_allocated(0)
                hardware_info["vram_available_gb"] = round(
                    vram_available / (1024**3), 1
                )
            except Exception:
                hardware_info["vram_available_gb"] = hardware_info["vram_total_gb"]

            # Get compute capability
            hardware_info["gpu_compute_capability"] = (
                device_props.major,
                device_props.minor,
            )

            logger.info(
                f"GPU detected: {hardware_info['gpu_name']}",
                extra={
                    "vram_total_gb": hardware_info["vram_total_gb"],
                    "vram_available_gb": hardware_info["vram_available_gb"],
                    "compute_capability": hardware_info["gpu_compute_capability"],
                    "device_count": hardware_info["gpu_device_count"],
                },
            )

        safe_execute(
            get_detailed_gpu_info,
            default_value=None,
            operation_name="gpu_info_detection",
        )

    # FastEmbed provider detection with fallback
    def detect_fastembed_providers():
        try:
            from utils.model_manager import ModelManager

            test_model = ModelManager.get_text_embedding_model("BAAI/bge-small-en-v1.5")
            try:
                providers = test_model.model.model.get_providers()
                hardware_info["fastembed_providers"] = providers
                logger.info(
                    "FastEmbed providers detected",
                    extra={
                        "providers": providers,
                        "cuda_in_providers": "CUDAExecutionProvider" in providers,
                    },
                )
            except (AttributeError, RuntimeError, ImportError) as e:
                logger.warning(f"FastEmbed provider detection failed: {e}")
                # Fallback to basic provider list based on CUDA availability
                if hardware_info["cuda_available"]:
                    hardware_info["fastembed_providers"] = [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ]
                else:
                    hardware_info["fastembed_providers"] = ["CPUExecutionProvider"]
            finally:
                del test_model  # Cleanup
        except Exception as e:
            logger.warning(f"FastEmbed model initialization failed: {e}")
            # Ultimate fallback
            if hardware_info["cuda_available"]:
                hardware_info["fastembed_providers"] = [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
            else:
                hardware_info["fastembed_providers"] = ["CPUExecutionProvider"]

    safe_execute(
        detect_fastembed_providers,
        default_value=None,
        operation_name="fastembed_detection",
    )

    # Log performance and results
    duration = time.perf_counter() - start_time
    log_performance(
        "hardware_detection",
        duration,
        cuda_available=hardware_info["cuda_available"],
        gpu_name=hardware_info["gpu_name"],
        gpu_count=hardware_info["gpu_device_count"],
        providers_count=len(hardware_info["fastembed_providers"]),
        cpu_cores=hardware_info["cpu_cores"],
    )

    logger.success(
        "Hardware detection completed",
        extra={
            "cuda_available": hardware_info["cuda_available"],
            "gpu_name": hardware_info["gpu_name"],
            "gpu_count": hardware_info["gpu_device_count"],
            "cpu_cores": hardware_info["cpu_cores"],
            "providers": hardware_info["fastembed_providers"],
        },
    )

    return hardware_info


def get_optimal_providers(force_cpu: bool = False) -> list[str]:
    """Get optimal execution providers based on available hardware.

    Args:
        force_cpu: Force CPU-only execution even if GPU is available

    Returns:
        List of execution providers in order of preference

    Note:
        Returns providers optimized for the current hardware configuration.
        GPU providers are prioritized when available and not forced to CPU.
    """
    if force_cpu:
        return ["CPUExecutionProvider"]

    hardware = detect_hardware()

    if (
        hardware["cuda_available"]
        and "CUDAExecutionProvider" in hardware["fastembed_providers"]
    ):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]


def check_gpu_memory_available(required_gb: float) -> bool:
    """Check if sufficient GPU memory is available.

    Args:
        required_gb: Required GPU memory in gigabytes

    Returns:
        True if sufficient GPU memory is available, False otherwise

    Note:
        Returns False if CUDA is not available or if memory detection fails.
    """
    if not torch.cuda.is_available():
        return False

    try:
        available_memory = torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_allocated(0)
        available_gb = available_memory / (1024**3)
        return available_gb >= required_gb
    except Exception as e:
        logger.warning(f"Failed to check GPU memory availability: {e}")
        return False


def get_recommended_batch_size(model_type: str = "embedding") -> int:
    """Get recommended batch size based on available hardware.

    Args:
        model_type: Type of model ("embedding", "llm", "vision")

    Returns:
        Recommended batch size for the model type and available hardware

    Note:
        Batch sizes are optimized based on GPU memory and model requirements.
        Falls back to conservative sizes when hardware detection fails.
    """
    hardware = detect_hardware()

    if not hardware["cuda_available"]:
        # Conservative CPU batch sizes
        return {"embedding": 16, "llm": 1, "vision": 4}.get(model_type, 8)

    vram_gb = (
        hardware.get("vram_available_gb", 0) or hardware.get("vram_total_gb", 0) or 0
    )

    # GPU batch sizes based on available VRAM
    if vram_gb >= 16:  # High-end GPU
        return {"embedding": 128, "llm": 8, "vision": 32}.get(model_type, 64)
    elif vram_gb >= 8:  # Mid-range GPU
        return {"embedding": 64, "llm": 4, "vision": 16}.get(model_type, 32)
    elif vram_gb >= 4:  # Entry-level GPU
        return {"embedding": 32, "llm": 2, "vision": 8}.get(model_type, 16)
    else:  # Low VRAM or unknown
        return {"embedding": 16, "llm": 1, "vision": 4}.get(model_type, 8)


def is_mixed_precision_supported() -> bool:
    """Check if mixed precision (FP16) is supported by the current GPU.

    Returns:
        True if mixed precision is supported, False otherwise

    Note:
        Mixed precision requires compute capability 7.0+ for optimal performance.
        Returns False if CUDA is not available or detection fails.
    """
    if not torch.cuda.is_available():
        return False

    try:
        compute_capability = torch.cuda.get_device_capability(0)
        # Mixed precision is well-supported on compute capability 7.0+
        return compute_capability[0] >= 7
    except Exception as e:
        logger.warning(f"Failed to check mixed precision support: {e}")
        return False


def optimize_torch_settings():
    """Optimize PyTorch settings based on available hardware.

    Note:
        Applies hardware-specific optimizations for better performance.
        Safe to call multiple times - settings are idempotent.
    """
    if torch.cuda.is_available():
        # Enable optimized attention for modern GPUs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TensorFloat-32 optimizations for CUDA")
        except Exception as e:
            logger.warning(f"Failed to enable TF32 optimizations: {e}")

        # Enable cuDNN benchmark mode for consistent input sizes
        try:
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark mode")
        except Exception as e:
            logger.warning(f"Failed to enable cuDNN benchmark: {e}")

    # Set optimal number of threads for CPU operations
    try:
        import os

        cpu_cores = os.cpu_count() or 1
        torch.set_num_threads(min(cpu_cores, 8))  # Cap at 8 to avoid oversubscription
        logger.info(f"Set PyTorch CPU threads to {torch.get_num_threads()}")
    except Exception as e:
        logger.warning(f"Failed to optimize CPU thread settings: {e}")


def get_hardware_summary() -> dict[str, Any]:
    """Get a comprehensive hardware summary for logging and diagnostics.

    Returns:
        Dictionary with human-readable hardware summary including
        recommendations for optimal configuration.
    """
    hardware = detect_hardware()

    summary = {
        "hardware_type": "GPU" if hardware["cuda_available"] else "CPU",
        "primary_device": hardware["gpu_name"]
        if hardware["cuda_available"]
        else f"{hardware['cpu_cores']} CPU cores",
        "memory_info": f"{hardware['vram_total_gb']}GB VRAM"
        if hardware["cuda_available"]
        else "System RAM",
        "execution_providers": hardware["fastembed_providers"],
        "recommended_batch_size_embedding": get_recommended_batch_size("embedding"),
        "recommended_batch_size_llm": get_recommended_batch_size("llm"),
        "mixed_precision_supported": is_mixed_precision_supported(),
    }

    if hardware["cuda_available"]:
        summary["gpu_details"] = {
            "device_count": hardware["gpu_device_count"],
            "compute_capability": hardware["gpu_compute_capability"],
            "vram_available_gb": hardware["vram_available_gb"],
        }

    return summary
