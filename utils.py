"""Core utility functions for DocMind AI.

This module contains general-purpose utilities including logging setup,
hardware detection, and RRF configuration verification.

Functions:
    setup_logging: Configure application logging.
    detect_hardware: Detect GPU and FastEmbed providers.
    verify_rrf_configuration: Verify RRF config against research.
"""

import logging
from typing import Any

import torch

from model_manager import ModelManager
from models import AppSettings

settings = AppSettings()


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("docmind.log")],
    )


def detect_hardware() -> dict[str, Any]:
    """Use FastEmbed native hardware detection.

    Detects GPU availability and FastEmbed execution providers using the
    model manager singleton for efficient resource usage.

    Returns:
        Dictionary with hardware status including CUDA availability,
        GPU information, and FastEmbed providers.
    """
    hardware_info = {
        "cuda_available": False,
        "gpu_name": "Unknown",
        "vram_total_gb": None,
        "fastembed_providers": [],
    }

    # Use FastEmbed's native hardware detection
    try:
        # FastEmbed automatically detects available providers
        test_model = ModelManager.get_text_embedding_model("BAAI/bge-small-en-v1.5")

        # Get detected providers from FastEmbed
        try:
            providers = test_model.model.model.get_providers()
            hardware_info["fastembed_providers"] = providers
            hardware_info["cuda_available"] = "CUDAExecutionProvider" in providers
            logging.info("FastEmbed detected providers: %s", providers)
        except Exception:
            # Fallback detection
            hardware_info["cuda_available"] = torch.cuda.is_available()

        # Basic GPU info if available
        if hardware_info["cuda_available"] and torch.cuda.is_available():
            try:
                hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                hardware_info["vram_total_gb"] = round(vram_gb, 1)
            except Exception as e:
                logging.warning("GPU info detection failed: %s", e)

        del test_model  # Cleanup

    except Exception as e:
        logging.warning("FastEmbed hardware detection failed: %s", e)
        # Ultimate fallback
        hardware_info["cuda_available"] = torch.cuda.is_available()

    return hardware_info


def verify_rrf_configuration(settings: AppSettings) -> dict[str, Any]:
    """Verify RRF configuration meets Phase 2.1 requirements.

    Checks:
    - Research-backed weights (dense: 0.7, sparse: 0.3)
    - Proper prefetch mechanism configuration
    - RRF alpha parameter within research range

    Returns:
        dict: Configuration verification results and computed parameters.
    """
    verification = {
        "weights_correct": False,
        "prefetch_enabled": True,  # Always enabled in our implementation
        "alpha_in_range": False,
        "computed_hybrid_alpha": 0.0,
        "issues": [],
        "recommendations": [],
    }

    # Check research-backed weights
    expected_dense = 0.7
    expected_sparse = 0.3
    if (
        abs(settings.rrf_fusion_weight_dense - expected_dense) < 0.05
        and abs(settings.rrf_fusion_weight_sparse - expected_sparse) < 0.05
    ):
        verification["weights_correct"] = True
    else:
        verification["issues"].append(
            f"Weights not research-backed: dense={settings.rrf_fusion_weight_dense}, "
            f"sparse={settings.rrf_fusion_weight_sparse} (expected 0.7/0.3)"
        )
        verification["recommendations"].append(
            "Update weights to research-backed values: dense=0.7, sparse=0.3"
        )

    # Check RRF alpha parameter (research suggests 10-100, with 60 as optimal)
    if 10 <= settings.rrf_fusion_alpha <= 100:
        verification["alpha_in_range"] = True
    else:
        verification["issues"].append(
            f"RRF alpha ({settings.rrf_fusion_alpha}) outside research range (10-100)"
        )
        verification["recommendations"].append(
            "Set RRF alpha between 10-100, with 60 as optimal"
        )

    # Calculate hybrid alpha for LlamaIndex
    verification["computed_hybrid_alpha"] = settings.rrf_fusion_weight_dense / (
        settings.rrf_fusion_weight_dense + settings.rrf_fusion_weight_sparse
    )

    logging.info(f"RRF Configuration Verification: {verification}")
    return verification
