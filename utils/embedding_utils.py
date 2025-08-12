"""Embedding model creation and management utilities for DocMind AI.

This module provides optimized embedding model initialization with GPU acceleration
and error recovery. Consolidates embedding-related functionality to follow DRY
principles and provide consistent embedding model management across the application.

Key features:
- GPU-optimized embedding model configuration
- Automatic hardware detection and provider selection
- Model factory pattern with caching and reuse
- Comprehensive error handling and retry logic
- Research-backed model settings for optimal performance

Example:
    Basic embedding model creation::

        from utils.embedding_utils import get_embed_model, create_optimized_embedding

        # Get standard embedding model with optimal configuration
        embed_model = get_embed_model()
        embeddings = embed_model.get_text_embedding("Hello world")

        # Create custom embedding model
        custom_model = create_optimized_embedding(
            model_name="jinaai/jina-embeddings-v3",
            use_gpu=True
        )
"""

import time
from typing import Any

import torch
from loguru import logger

try:
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
except ImportError:
    FastEmbedEmbedding = None

from src.core.infrastructure.hardware_utils import get_optimal_providers
from src.models.core import settings

from .exceptions import handle_embedding_error
from .logging_utils import log_error_with_context, log_performance
from .retry_utils import embedding_retry


@embedding_retry
def get_embed_model() -> FastEmbedEmbedding:
    """Create optimized embedding model with GPU acceleration and error recovery.

    Initializes a FastEmbedEmbedding model with optimal configuration for
    both CPU and GPU environments. Applies hardware-specific optimizations
    and uses research-backed model settings for best performance. Includes
    comprehensive error handling and automatic retries.

    GPU optimizations include:
    - Automatic provider selection based on hardware detection
    - Optimal batch size configuration
    - Mixed precision support where available
    - Memory management and cleanup

    Returns:
        FastEmbedEmbedding: Fully configured and optimized embedding model.

    Raises:
        EmbeddingError: If model initialization fails after retries.

    Note:
        The model is configured with:
        - Model: From settings.dense_embedding_model (default: BGE-Large)
        - Max length: 512 tokens for optimal performance
        - Cache directory: './embeddings_cache'
        - Dynamic provider selection based on hardware

    Example:
        >>> embed_model = get_embed_model()
        >>> embeddings = embed_model.get_text_embedding("Hello world")
        >>> print(f"Embedding dimension: {len(embeddings)}")
    """
    start_time = time.perf_counter()

    logger.info(
        "Creating embedding model",
        extra={
            "model": settings.dense_embedding_model,
            "gpu_enabled": settings.gpu_acceleration,
            "batch_size": settings.embedding_batch_size,
            "max_length": 512,
        },
    )

    try:
        # Get optimal providers based on hardware detection
        providers = get_optimal_providers(force_cpu=not settings.gpu_acceleration)

        if "CUDAExecutionProvider" in providers:
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for embeddings")

        # Use embedding factory for consistent model creation
        embed_model = create_optimized_embedding(
            model_name=settings.dense_embedding_model,
            use_gpu=settings.gpu_acceleration,
            cache_dir="./embeddings_cache",
            batch_size=settings.embedding_batch_size,
        )

        duration = time.perf_counter() - start_time
        log_performance(
            "embedding_model_creation",
            duration,
            model=settings.dense_embedding_model,
            gpu_enabled=settings.gpu_acceleration and torch.cuda.is_available(),
            providers=providers,
        )

        logger.success(
            f"Embedding model created successfully: {settings.dense_embedding_model}"
        )
        return embed_model

    except Exception as e:
        log_error_with_context(
            e,
            "embedding_model_creation",
            context={
                "model": settings.dense_embedding_model,
                "gpu_requested": settings.gpu_acceleration,
                "cuda_available": torch.cuda.is_available(),
                "batch_size": settings.embedding_batch_size,
            },
        )
        raise handle_embedding_error(
            e,
            operation="get_embed_model",
            model=settings.dense_embedding_model,
            gpu_requested=settings.gpu_acceleration,
            cuda_available=torch.cuda.is_available(),
        ) from e


def create_optimized_embedding(
    model_name: str,
    use_gpu: bool = True,
    cache_dir: str = "./embeddings_cache",
    batch_size: int = 32,
    max_length: int = 512,
) -> FastEmbedEmbedding:
    """Create an optimized FastEmbedEmbedding model with custom configuration.

    Args:
        model_name: Name of the embedding model to load
        use_gpu: Whether to enable GPU acceleration
        cache_dir: Directory for model cache
        batch_size: Batch size for embedding computation
        max_length: Maximum sequence length for embeddings

    Returns:
        Configured FastEmbedEmbedding instance

    Raises:
        ImportError: If FastEmbedEmbedding is not available
        EmbeddingError: If model creation fails

    Note:
        Automatically selects optimal execution providers based on hardware
        and requested GPU settings. Provides consistent model configuration
        across the application.
    """
    if FastEmbedEmbedding is None:
        raise ImportError(
            "FastEmbedEmbedding is not available. Please install fastembed package."
        )

    start_time = time.perf_counter()

    try:
        # Get optimal providers for the requested configuration
        providers = get_optimal_providers(force_cpu=not use_gpu)

        # Create model with optimal configuration
        embed_model = FastEmbedEmbedding(
            model_name=model_name,
            providers=providers,
            cache_dir=cache_dir,
            doc_embed_type="passage",  # Optimized for document embedding
            max_length=max_length,
            batch_size=batch_size,
        )

        duration = time.perf_counter() - start_time
        log_performance(
            "optimized_embedding_creation",
            duration,
            model_name=model_name,
            providers=providers,
            gpu_requested=use_gpu,
            batch_size=batch_size,
        )

        logger.info(
            f"Created optimized embedding model: {model_name}",
            extra={
                "providers": providers,
                "batch_size": batch_size,
                "max_length": max_length,
                "cache_dir": cache_dir,
            },
        )

        return embed_model

    except Exception as e:
        log_error_with_context(
            e,
            "optimized_embedding_creation",
            context={
                "model_name": model_name,
                "use_gpu": use_gpu,
                "cache_dir": cache_dir,
                "batch_size": batch_size,
            },
        )

        raise handle_embedding_error(
            e,
            operation="create_optimized_embedding",
            model=model_name,
            gpu_requested=use_gpu,
            cuda_available=torch.cuda.is_available(),
        ) from e


class EmbeddingModelManager:
    """Singleton manager for embedding models with caching and reuse.

    Provides centralized management of embedding models to prevent redundant
    initialization and optimize memory usage. Supports multiple model types
    with automatic cleanup and resource management.
    """

    _instance = None
    _models = {}

    def __new__(cls) -> "EmbeddingModelManager":
        """Create or return the singleton instance of EmbeddingModelManager.

        Returns:
            EmbeddingModelManager: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self, model_name: str = None, use_gpu: bool = None, force_reload: bool = False
    ) -> FastEmbedEmbedding:
        """Get or create an embedding model with caching.

        Args:
            model_name: Name of the model (uses default if None)
            use_gpu: GPU setting (uses default if None)
            force_reload: Force model recreation even if cached

        Returns:
            Cached or newly created embedding model
        """
        # Use defaults from settings if not specified
        if model_name is None:
            model_name = settings.dense_embedding_model
        if use_gpu is None:
            use_gpu = settings.gpu_acceleration

        model_key = f"{model_name}_{use_gpu}"

        if force_reload or model_key not in self._models:
            logger.info(f"Creating new embedding model: {model_key}")

            self._models[model_key] = create_optimized_embedding(
                model_name=model_name,
                use_gpu=use_gpu,
                batch_size=settings.embedding_batch_size,
            )
        else:
            logger.debug(f"Using cached embedding model: {model_key}")

        return self._models[model_key]

    def clear_cache(self):
        """Clear all cached models to free memory."""
        logger.info(f"Clearing {len(self._models)} cached embedding models")

        for model_key, model in self._models.items():
            try:
                # Clean up model resources
                if hasattr(model, "model") and model.model is not None:
                    del model.model
            except Exception as e:
                logger.warning(f"Error cleaning up model {model_key}: {e}")

        self._models.clear()

        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_info(self) -> dict[str, Any]:
        """Get information about cached models.

        Returns:
            Dictionary with model cache statistics and information
        """
        return {
            "cached_models": list(self._models.keys()),
            "model_count": len(self._models),
            "default_model": settings.dense_embedding_model,
            "gpu_enabled": settings.gpu_acceleration,
        }


# Global model manager instance
_model_manager = EmbeddingModelManager()


def get_embedding_model_manager() -> EmbeddingModelManager:
    """Get the global embedding model manager instance.

    Returns:
        Global EmbeddingModelManager for centralized model management
    """
    return _model_manager


def validate_embedding_model(model_name: str) -> dict[str, Any]:
    """Validate an embedding model and return its specifications.

    Args:
        model_name: Name of the model to validate

    Returns:
        Dictionary with model validation results and specifications

    Note:
        Attempts to load the model briefly to verify it's available and
        get its dimensions and capabilities.
    """
    validation_result = {
        "model_name": model_name,
        "available": False,
        "dimensions": None,
        "max_length": None,
        "providers": None,
        "error": None,
    }

    try:
        # Create test model to validate availability
        test_model = create_optimized_embedding(
            model_name=model_name,
            use_gpu=False,  # Use CPU for validation
            batch_size=1,  # Minimal batch size
        )

        # Test embedding to get dimensions
        test_embedding = test_model.get_text_embedding("test")

        validation_result.update(
            {
                "available": True,
                "dimensions": len(test_embedding),
                "providers": get_optimal_providers(),
            }
        )

        logger.info(f"Model validation successful: {model_name}")

    except Exception as e:
        validation_result["error"] = str(e)
        logger.warning(f"Model validation failed for {model_name}: {e}")

    return validation_result


def get_supported_models() -> list[str]:
    """Get list of supported embedding models.

    Returns:
        List of model names that are known to work with FastEmbed

    Note:
        This is a curated list of models that have been tested and
        work well with the current embedding pipeline.
    """
    return [
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "jinaai/jina-embeddings-v2-base-en",
        "jinaai/jina-embeddings-v3",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "nomic-ai/nomic-embed-text-v1",
        "WhereIsAI/UAE-Large-V1",
    ]


def benchmark_embedding_model(
    model_name: str, test_texts: list[str] = None, batch_sizes: list[int] = None
) -> dict[str, Any]:
    """Benchmark an embedding model's performance.

    Args:
        model_name: Name of the model to benchmark
        test_texts: Custom test texts (uses defaults if None)
        batch_sizes: Batch sizes to test (uses defaults if None)

    Returns:
        Dictionary with benchmark results including timing and throughput

    Note:
        Useful for optimizing embedding performance and selecting
        appropriate batch sizes for different hardware configurations.
    """
    if test_texts is None:
        test_texts = [
            "Short test.",
            "Longer test sentence with more context for benchmarking.",
            "Compact text.",
            "Medium-length text for performance evaluation.",
        ] * 10  # 40 texts total

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    benchmark_results = {
        "model_name": model_name,
        "test_text_count": len(test_texts),
        "batch_results": {},
        "optimal_batch_size": None,
        "error": None,
    }

    try:
        best_throughput = 0
        optimal_batch = 1

        for batch_size in batch_sizes:
            logger.info(f"Benchmarking {model_name} with batch_size={batch_size}")

            start_time = time.perf_counter()

            # Create model with specific batch size
            model = create_optimized_embedding(
                model_name=model_name,
                batch_size=batch_size,
            )

            # Run embeddings
            embeddings = []
            for i in range(0, len(test_texts), batch_size):
                batch = test_texts[i : i + batch_size]
                batch_embeddings = [model.get_text_embedding(text) for text in batch]
                embeddings.extend(batch_embeddings)

            duration = time.perf_counter() - start_time
            throughput = len(test_texts) / duration  # texts per second

            benchmark_results["batch_results"][batch_size] = {
                "duration_seconds": duration,
                "throughput_texts_per_sec": throughput,
                "avg_time_per_text": duration / len(test_texts),
            }

            if throughput > best_throughput:
                best_throughput = throughput
                optimal_batch = batch_size

            logger.info(
                f"Batch {batch_size}: {throughput:.2f} texts/sec, {duration:.2f}s"
            )

        benchmark_results["optimal_batch_size"] = optimal_batch
        logger.success(
            f"Benchmark complete. Optimal batch size: {optimal_batch} "
            f"({best_throughput:.2f} texts/sec)"
        )

    except Exception as e:
        benchmark_results["error"] = str(e)
        log_error_with_context(
            e,
            "embedding_model_benchmark",
            context={"model_name": model_name, "batch_sizes": batch_sizes},
        )

    return benchmark_results
