"""Configuration validation utilities for DocMind AI.

This module provides comprehensive configuration validation capabilities including
RRF parameter validation, startup configuration checks, and model compatibility
verification. Consolidates validation functionality to follow DRY principles
and ensure consistent configuration validation across the application.

Key features:
- Research-backed RRF configuration validation
- Comprehensive startup configuration validation
- Model compatibility and dimension verification
- Hardware configuration validation
- Detailed validation reporting with recommendations

Example:
    Basic configuration validation::

        from utils.validation_utils import (
    verify_rrf_configuration,
    validate_startup_configuration
)
        from src.models import AppSettings

        settings = AppSettings()

        # Validate RRF configuration
        rrf_result = verify_rrf_configuration(settings)
        if rrf_result['issues']:
            print("RRF issues found:", rrf_result['issues'])

        # Comprehensive startup validation
        startup_result = validate_startup_configuration(settings)
        if not startup_result['valid']:
            print("Critical configuration errors found")
"""

import os
from typing import Any

import torch
from loguru import logger
from qdrant_client import QdrantClient

from src.core.infrastructure.hardware_utils import detect_hardware
from src.models import AppSettings

from .logging_utils import log_error_with_context


def verify_rrf_configuration(settings: AppSettings) -> dict[str, Any]:
    """Verify RRF configuration against research recommendations.

    Validates Reciprocal Rank Fusion (RRF) parameters against established
    research findings to ensure optimal hybrid search performance. Checks
    weight distribution, alpha parameters, and provides recommendations
    for configuration improvements.

    Research-backed requirements:
    - Dense embedding weight: 0.7 (±0.05 tolerance)
    - Sparse embedding weight: 0.3 (±0.05 tolerance)
    - RRF alpha parameter: 10-100 range (60 optimal)
    - Prefetch mechanism: Always enabled

    Args:
        settings: Application settings containing RRF configuration.

    Returns:
        Dictionary containing verification results with keys:
        - 'weights_correct' (bool): Whether weights match research findings
        - 'prefetch_enabled' (bool): Prefetch mechanism status (always True)
        - 'alpha_in_range' (bool): Whether alpha is in valid range
        - 'computed_hybrid_alpha' (float): Calculated hybrid alpha for LlamaIndex
        - 'issues' (list[str]): List of configuration problems found
        - 'recommendations' (list[str]): Suggested fixes for issues

    Note:
        The computed_hybrid_alpha is calculated as:
        dense_weight / (dense_weight + sparse_weight)
        This value is used by LlamaIndex for hybrid query processing.

    Example:
        >>> from src.models import AppSettings
        >>> settings = AppSettings()
        >>> verification = verify_rrf_configuration(settings)
        >>> if verification['issues']:
        ...     for issue in verification['issues']:
        ...         print(f"Issue: {issue}")
        >>> print(f"Hybrid alpha: {verification['computed_hybrid_alpha']:.3f}")
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

    logger.info("RRF Configuration Verification", extra={"verification": verification})
    return verification


def validate_startup_configuration(settings: AppSettings) -> dict[str, Any]:
    """Perform comprehensive startup configuration validation.

    Validates all critical configuration parameters to ensure the application
    can start successfully and operate correctly. Checks database connectivity,
    hardware configuration, model settings, and parameter consistency.

    Args:
        settings: Application settings to validate.

    Returns:
        Dictionary with validation results:
        - 'valid' (bool): Whether configuration is valid for startup
        - 'warnings' (list[str]): Non-critical issues that should be reviewed
        - 'errors' (list[str]): Critical errors that prevent startup
        - 'info' (list[str]): Informational messages about configuration
        - 'hardware_summary' (dict): Summary of detected hardware capabilities

    Raises:
        RuntimeError: If critical configuration errors are found.

    Note:
        This function performs extensive validation including:
        - Database connectivity testing
        - Hardware capability verification
        - Model dimension compatibility
        - Parameter range validation
        - File path existence checks
    """
    results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "info": [],
        "hardware_summary": {},
    }

    logger.info("Starting comprehensive configuration validation")

    # Get hardware information for validation
    try:
        hardware_info = detect_hardware()
        results["hardware_summary"] = hardware_info
        results["info"].append(f"Hardware: {hardware_info.get('gpu_name', 'CPU')}")
    except Exception as e:
        results["warnings"].append(f"Hardware detection failed: {e}")

    # Check Qdrant connectivity
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=10)
        collections = client.get_collections()
        results["info"].append(
            f"Qdrant connection successful: {settings.qdrant_url} "
            f"({len(collections.collections)} collections)"
        )
        client.close()
    except Exception as e:
        results["errors"].append(f"Qdrant connection failed: {e}")
        results["valid"] = False

    # Check GPU configuration
    if settings.gpu_acceleration:
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                results["info"].append(
                    f"GPU available: {gpu_name} ({gpu_memory:.1f}GB VRAM)"
                )

                # Check if GPU has sufficient memory for selected models
                if gpu_memory < 2.0:  # Less than 2GB VRAM
                    results["warnings"].append(
                        f"Low GPU memory: {gpu_memory:.1f}GB - limited perf"
                    )
            except Exception as e:
                results["warnings"].append(f"GPU detection issue: {e}")
        else:
            results["warnings"].append(
                "GPU acceleration enabled but no GPU available - will use CPU"
            )

    # Check embedding model dimensions and compatibility
    expected_dims = {
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "jinaai/jina-embeddings-v3": 1024,
        "jinaai/jina-embeddings-v2-base-en": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    }

    if settings.dense_embedding_model in expected_dims:
        expected = expected_dims[settings.dense_embedding_model]
        if settings.dense_embedding_dimension != expected:
            results["warnings"].append(
                f"Model {settings.dense_embedding_model} typically uses "
                f"{expected}D, but {settings.dense_embedding_dimension}D configured"
            )
    else:
        results["info"].append(
            f"Custom embedding model: {settings.dense_embedding_model} "
            f"({settings.dense_embedding_dimension}D)"
        )

    # Check RRF configuration if sparse embeddings are enabled
    if settings.enable_sparse_embeddings:
        # Check alpha parameter
        if not 10 <= settings.rrf_fusion_alpha <= 100:
            results["warnings"].append(
                f"RRF alpha {settings.rrf_fusion_alpha} - outside recommended range"
            )

        # Verify RRF configuration
        verification = verify_rrf_configuration(settings)
        if verification["issues"]:
            results["warnings"].extend(verification["issues"])
            results["info"].append(
                f"RRF hybrid alpha: {verification['computed_hybrid_alpha']:.3f}"
            )

    # Check chunk configuration
    if settings.chunk_overlap >= settings.chunk_size:
        results["errors"].append(
            f"Chunk overlap ({settings.chunk_overlap}) must be less than "
            f"chunk size ({settings.chunk_size})"
        )
        results["valid"] = False

    if settings.chunk_size < 100:
        results["warnings"].append(
            f"Very small chunk size ({settings.chunk_size}) may reduce context quality"
        )
    elif settings.chunk_size > 2048:
        results["warnings"].append(
            f"Large chunk size ({settings.chunk_size}) may exceed model context limits"
        )

    # Check model paths for llamacpp
    if settings.backend == "llamacpp":
        if not os.path.exists(settings.llamacpp_model_path):
            results["errors"].append(
                f"Llama.cpp model path does not exist: {settings.llamacpp_model_path}"
            )
            results["valid"] = False
        else:
            # Check model file size
            try:
                model_size_gb = os.path.getsize(settings.llamacpp_model_path) / (
                    1024**3
                )
                results["info"].append(
                    f"Llama.cpp: {os.path.basename(settings.llamacpp_model_path)} "
                    f"({model_size_gb:.1f}GB)"
                )
            except Exception as e:
                results["warnings"].append(f"Could not get model file size: {e}")

    # Check batch sizes
    if settings.embedding_batch_size < 1 or settings.embedding_batch_size > 1024:
        results["warnings"].append(
            f"Embedding batch size ({settings.embedding_batch_size}) may be suboptimal"
        )

    # Check memory usage configuration
    if hasattr(settings, "max_memory_usage_gb"):
        try:
            import psutil

            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if settings.max_memory_usage_gb > available_memory_gb * 0.8:
                results["warnings"].append(
                    f"Configured memory limit ({settings.max_memory_usage_gb}GB) "
                    f"exceeds 80% of available memory ({available_memory_gb:.1f}GB)"
                )
        except ImportError:
            results["info"].append("psutil not available for memory validation")

    # Validate API keys and credentials (without exposing values)
    if hasattr(settings, "openai_api_key") and settings.openai_api_key:
        if len(settings.openai_api_key) < 20:
            results["warnings"].append("OpenAI API key appears to be incomplete")
        else:
            results["info"].append("OpenAI API key configured")

    if hasattr(settings, "llamaparse_api_key") and settings.llamaparse_api_key:
        if len(settings.llamaparse_api_key) < 20:
            results["warnings"].append("LlamaParse API key appears to be incomplete")
        else:
            results["info"].append("LlamaParse API key configured")

    # Final validation
    if not results["valid"]:
        error_msg = "Critical configuration errors found:\n" + "\n".join(
            results["errors"]
        )
        logger.error(
            "Configuration validation failed", extra={"errors": results["errors"]}
        )
        raise RuntimeError(error_msg)

    # Log validation summary
    logger.info(
        "Configuration validation completed",
        extra={
            "valid": results["valid"],
            "warnings_count": len(results["warnings"]),
            "errors_count": len(results["errors"]),
            "info_count": len(results["info"]),
        },
    )

    return results


def validate_model_compatibility(
    model_name: str, expected_dimension: int = None, hardware_requirements: dict = None
) -> dict[str, Any]:
    """Validate model compatibility with current hardware and configuration.

    Args:
        model_name: Name of the model to validate
        expected_dimension: Expected embedding dimension (if known)
        hardware_requirements: Minimum hardware requirements

    Returns:
        Dictionary with compatibility validation results
    """
    compatibility = {
        "model_name": model_name,
        "compatible": True,
        "issues": [],
        "recommendations": [],
        "hardware_sufficient": True,
    }

    try:
        # Check hardware requirements
        hardware_info = detect_hardware()

        if hardware_requirements:
            min_vram = hardware_requirements.get("min_vram_gb", 0)
            if hardware_info["cuda_available"]:
                available_vram = hardware_info.get("vram_available_gb", 0)
                if available_vram < min_vram:
                    compatibility["issues"].append(
                        f"Insufficient VRAM: {available_vram}GB available, "
                        f"{min_vram}GB required"
                    )
                    compatibility["hardware_sufficient"] = False
            else:
                # Model requires GPU but none available
                if min_vram > 0:
                    compatibility["issues"].append(
                        f"Model requires {min_vram}GB VRAM - no GPU available"
                    )
                    compatibility["hardware_sufficient"] = False

        # Check if model name follows expected patterns
        known_providers = ["BAAI", "jinaai", "sentence-transformers", "nomic-ai"]
        if not any(provider in model_name for provider in known_providers):
            compatibility["recommendations"].append(
                f"Unknown model provider: {model_name}"
            )

        # Validate expected dimensions if provided
        if expected_dimension:
            known_dimensions = [384, 512, 768, 1024, 1536]
            if expected_dimension not in known_dimensions:
                compatibility["recommendations"].append(
                    f"Unusual embedding dimension: {expected_dimension}"
                )

        compatibility["compatible"] = (
            compatibility["hardware_sufficient"] and len(compatibility["issues"]) == 0
        )

    except Exception as e:
        compatibility["issues"].append(f"Validation failed: {e}")
        compatibility["compatible"] = False
        log_error_with_context(
            e, "model_compatibility_validation", context={"model_name": model_name}
        )

    return compatibility


def validate_search_configuration(settings: AppSettings) -> dict[str, Any]:
    """Validate search and retrieval configuration parameters.

    Args:
        settings: Application settings to validate

    Returns:
        Dictionary with search configuration validation results
    """
    search_validation = {
        "valid": True,
        "issues": [],
        "recommendations": [],
    }

    # Check retrieval parameters
    if hasattr(settings, "similarity_top_k"):
        if settings.similarity_top_k < 1:
            search_validation["issues"].append("similarity_top_k must be at least 1")
            search_validation["valid"] = False
        elif settings.similarity_top_k > 100:
            search_validation["recommendations"].append(
                "High similarity_top_k - potential performance impact"
            )

    # Check reranking configuration
    if (
        hasattr(settings, "enable_reranking")
        and settings.enable_reranking
        and hasattr(settings, "rerank_top_n")
        and settings.rerank_top_n > settings.similarity_top_k
    ):
        search_validation["recommendations"].append(
            f"rerank_top_n ({settings.rerank_top_n}) should not exceed "
            f"similarity_top_k ({settings.similarity_top_k})"
        )

    # Check embedding model settings
    if settings.dense_embedding_dimension <= 0:
        search_validation["issues"].append("dense_embedding_dimension must be positive")
        search_validation["valid"] = False

    return search_validation


def get_configuration_health_score(settings: AppSettings) -> dict[str, Any]:
    """Calculate a health score for the current configuration.

    Args:
        settings: Application settings to score

    Returns:
        Dictionary with health score and detailed breakdown
    """
    health_score = {
        "overall_score": 0.0,
        "category_scores": {},
        "max_score": 100.0,
        "issues_found": 0,
        "recommendations": [],
    }

    categories = {
        "database": 20,  # Qdrant connectivity and configuration
        "hardware": 25,  # GPU/CPU configuration optimality
        "models": 20,  # Model compatibility and settings
        "search": 15,  # Search and retrieval configuration
        "performance": 10,  # Batch sizes and optimization settings
        "security": 10,  # API keys and security configuration
    }

    try:
        # Database health
        db_score = 0
        try:
            startup_validation = validate_startup_configuration(settings)
            if startup_validation["valid"] and not any(
                "Qdrant" in error for error in startup_validation["errors"]
            ):
                db_score = categories["database"]
        except RuntimeError:
            pass  # Database issues already captured
        health_score["category_scores"]["database"] = db_score

        # Hardware optimization
        hardware_score = 0
        hardware_info = detect_hardware()
        if hardware_info["cuda_available"] and settings.gpu_acceleration:
            hardware_score += 15  # GPU available and enabled
        elif not hardware_info["cuda_available"] and not settings.gpu_acceleration:
            hardware_score += 10  # Consistent CPU-only configuration

        # Check if batch sizes are reasonable for hardware
        from src.core.infrastructure.hardware_utils import get_recommended_batch_size

        recommended_batch = get_recommended_batch_size("embedding")
        if abs(settings.embedding_batch_size - recommended_batch) <= 16:
            hardware_score += 10  # Reasonable batch size
        else:
            health_score["recommendations"].append(
                f"Adjust embedding batch to {recommended_batch} for optimal performance"
            )

        health_score["category_scores"]["hardware"] = hardware_score

        # Model configuration
        model_score = 0
        expected_dims = {
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-small-en-v1.5": 384,
        }
        if settings.dense_embedding_model in expected_dims:
            if (
                settings.dense_embedding_dimension
                == expected_dims[settings.dense_embedding_model]
            ):
                model_score += 15  # Correct dimensions
            else:
                model_score += 5  # Known model but wrong dimensions
        else:
            model_score += 10  # Custom model (neutral)

        if settings.enable_sparse_embeddings:
            rrf_validation = verify_rrf_configuration(settings)
            if not rrf_validation["issues"]:
                model_score += 5  # Good RRF configuration

        health_score["category_scores"]["models"] = model_score

        # Search configuration
        search_score = 0
        search_validation = validate_search_configuration(settings)
        if search_validation["valid"]:
            search_score = categories["search"]
        health_score["category_scores"]["search"] = search_score

        # Performance configuration
        perf_score = 0
        if 1 <= settings.embedding_batch_size <= 128:
            perf_score += 5
        if 512 <= settings.chunk_size <= 1024:
            perf_score += 3
        if 0 < settings.chunk_overlap < settings.chunk_size * 0.2:
            perf_score += 2
        health_score["category_scores"]["performance"] = perf_score

        # Security configuration
        security_score = 0
        if (
            hasattr(settings, "openai_api_key")
            and settings.openai_api_key
            and len(settings.openai_api_key) > 20
        ):
            security_score += 5

        if (
            hasattr(settings, "llamaparse_api_key")
            and settings.llamaparse_api_key
            and len(settings.llamaparse_api_key) > 20
        ):
            security_score += 5

        health_score["category_scores"]["security"] = security_score

        # Calculate overall score
        health_score["overall_score"] = sum(health_score["category_scores"].values())

        # Count issues from various validations
        health_score["issues_found"] = len(health_score["recommendations"])

    except Exception as e:
        log_error_with_context(e, "configuration_health_scoring")
        health_score["overall_score"] = 0
        health_score["issues_found"] = 1

    return health_score
