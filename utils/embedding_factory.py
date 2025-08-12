"""Embedding model factory for DocMind AI with consistent configuration.

This module provides a centralized factory for creating embedding models with
optimal configuration, GPU acceleration, and caching. Eliminates code
duplication across dense, sparse, and multimodal embedding setups.

Features:
- LRU caching for expensive model initialization
- Consistent GPU/CPU provider configuration
- torch.compile optimization when available
- Support for dense, sparse, and multimodal embeddings
- Quantization support for memory efficiency
- Comprehensive error handling and logging

Example:
    Using the embedding factory::

        from utils.embedding_factory import EmbeddingFactory

        # Create dense embedding model
        dense_model = EmbeddingFactory.create_dense_embedding(use_gpu=True)

        # Create sparse embedding model
        sparse_model = EmbeddingFactory.create_sparse_embedding(use_gpu=True)

        # Create both for hybrid search
        dense, sparse = EmbeddingFactory.create_hybrid_embeddings(use_gpu=True)

Attributes:
    settings (AppSettings): Global application settings for embedding configuration.
"""

from functools import lru_cache
from typing import Any

import torch

try:
    from fastembed import SparseTextEmbedding
except ImportError:
    SparseTextEmbedding = None

try:
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
except ImportError:
    FastEmbedEmbedding = None
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from loguru import logger

from models.core import settings


class EmbeddingFactory:
    """Factory for creating embedding models with consistent configuration.

    Provides centralized model creation with optimized settings, caching,
    and consistent configuration across different embedding types.
    Eliminates code duplication and ensures all models use best practices.
    """

    @staticmethod
    def get_providers(use_gpu: bool | None = None) -> list[str]:
        """Get execution providers based on GPU availability.

        Determines optimal execution providers based on GPU availability
        and user preferences. Prioritizes CUDA when available and enabled.

        Args:
            use_gpu: Whether to use GPU acceleration. If None, uses settings value.

        Returns:
            List of execution providers in priority order.

        Example:
            >>> providers = EmbeddingFactory.get_providers(use_gpu=True)
            >>> print(providers)  # ['CUDAExecutionProvider', 'CPUExecutionProvider']
        """
        use_gpu = use_gpu if use_gpu is not None else settings.gpu_acceleration

        if use_gpu and torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU for embeddings: {gpu_name}")
            except (RuntimeError, AttributeError) as e:
                logger.warning(f"GPU info detection failed: {e}")
        else:
            providers = ["CPUExecutionProvider"]
            logger.info("Using CPU for embeddings")

        return providers

    @classmethod
    @lru_cache(maxsize=2)
    def create_dense_embedding(cls, use_gpu: bool | None = None) -> FastEmbedEmbedding:
        """Create dense embedding model with caching.

        Creates and caches a FastEmbedEmbedding model with optimal configuration
        for dense vector search. Applies GPU acceleration and torch.compile
        optimization when available.

        Args:
            use_gpu: Whether to enable GPU acceleration. If None, uses settings.

        Returns:
            FastEmbedEmbedding: Configured and optimized dense embedding model.

        Note:
            Model is cached using LRU cache to prevent redundant initialization.
            torch.compile is applied when available for performance optimization.

        Example:
            >>> model = EmbeddingFactory.create_dense_embedding(use_gpu=True)
            >>> embeddings = model.get_text_embedding("Hello world")
        """
        providers = cls.get_providers(use_gpu)

        if FastEmbedEmbedding is not None:
            embed_model = FastEmbedEmbedding(
                model_name=settings.dense_embedding_model,
                max_length=512,
                providers=providers,
                batch_size=settings.embedding_batch_size,
                cache_dir="./embeddings_cache",
            )
        else:
            # Fallback to HuggingFace if FastEmbed not available
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Apply torch.compile optimization if available and using GPU
        if use_gpu and torch.cuda.is_available() and hasattr(torch, "compile"):
            try:
                embed_model = torch.compile(
                    embed_model,
                    mode="reduce-overhead",  # Optimize for embedding workloads
                    dynamic=True,  # Handle variable batch sizes
                    fullgraph=False,  # Allow partial graph compilation
                )
                logger.info(
                    "torch.compile applied to dense embeddings with reduce-overhead"
                )
            except Exception as e:
                logger.warning(f"torch.compile failed for dense embeddings: {e}")

        logger.info(f"Dense embedding model created: {settings.dense_embedding_model}")
        return embed_model

    @classmethod
    @lru_cache(maxsize=2)
    def create_sparse_embedding(cls, use_gpu: bool | None = None) -> Any:
        """Create sparse embedding model with caching.

        Creates and caches a SparseTextEmbedding model for keyword-based search.
        Returns None if sparse embeddings are disabled in settings.

        Args:
            use_gpu: Whether to enable GPU acceleration. If None, uses settings.

        Returns:
            SparseTextEmbedding or None: Configured sparse embedding model or None
            if sparse embeddings are disabled.

        Note:
            Model is cached using LRU cache to prevent redundant initialization.
            Returns None if settings.enable_sparse_embeddings is False.

        Example:
            >>> model = EmbeddingFactory.create_sparse_embedding(use_gpu=True)
            >>> if model:
            ...     sparse_embeddings = model.embed(["Hello world"])
        """
        if not settings.enable_sparse_embeddings:
            logger.info("Sparse embeddings disabled in settings")
            return None

        providers = cls.get_providers(use_gpu)

        try:
            if SparseTextEmbedding is not None:
                sparse_model = SparseTextEmbedding(
                    model_name=settings.sparse_embedding_model,
                    providers=providers,
                    batch_size=settings.embedding_batch_size,
                    cache_dir="./embeddings_cache",
                )
            else:
                logger.warning("SparseTextEmbedding not available - returning None")
                return None

            logger.info(
                f"Sparse embedding model created: {settings.sparse_embedding_model}"
            )
            return sparse_model

        except Exception as e:
            logger.error(f"Failed to create sparse embedding model: {e}")
            return None

    @classmethod
    def create_multimodal_embedding(
        cls, use_gpu: bool | None = None
    ) -> HuggingFaceEmbedding:
        """Create multimodal embedding model for text and images.

        Creates a HuggingFaceEmbedding model capable of processing both text
        and image content in a unified embedding space. Supports quantization
        for memory efficiency.

        Args:
            use_gpu: Whether to enable GPU acceleration. If None, uses settings.

        Returns:
            HuggingFaceEmbedding: Configured multimodal embedding model.

        Note:
            Uses Jina v3 embeddings for multimodal performance.
            Applies quantization if enabled in settings for memory efficiency.

        Example:
            >>> model = EmbeddingFactory.create_multimodal_embedding(use_gpu=True)
            >>> text_embedding = model.get_text_embedding("Sample text")
        """
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Configure quantization if enabled
        quantization_config = None
        if settings.enable_quantization and use_gpu and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                logger.info("Quantization enabled for multimodal embeddings")
            except ImportError:
                logger.warning(
                    "transformers package not available, quantization disabled"
                )

        # Configure model kwargs
        model_kwargs = {
            "torch_dtype": torch.float16
            if use_gpu and torch.cuda.is_available()
            else torch.float32,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        embed_model = HuggingFaceEmbedding(
            model_name="jinaai/jina-embeddings-v3",  # State-of-the-art multimodal model
            embed_batch_size=settings.embedding_batch_size,
            device=device,
            trust_remote_code=True,  # Required for Jina v3
            model_kwargs=model_kwargs,
        )

        logger.info(f"Multimodal embedding model created on {device}")
        return embed_model

    @classmethod
    def create_hybrid_embeddings(cls, use_gpu: bool | None = None) -> tuple[Any, Any]:
        """Create both dense and sparse embeddings for hybrid search.

        Creates a tuple of dense and sparse embedding models optimized for
        hybrid search with RRF fusion. Handles cases where sparse embeddings
        are disabled gracefully.

        Args:
            use_gpu: Whether to enable GPU acceleration. If None, uses settings.

        Returns:
            Tuple containing:
            - FastEmbedEmbedding: Dense embedding model (always created)
            - SparseTextEmbedding or None: Sparse embedding model (optional)

        Note:
            Both models use consistent provider configuration and caching.
            Sparse model may be None if disabled in settings.

        Example:
            >>> dense, sparse = EmbeddingFactory.create_hybrid_embeddings(use_gpu=True)
            >>> if sparse:
            ...     print("Hybrid search available")
            ... else:
            ...     print("Dense-only search")
        """
        dense_model = cls.create_dense_embedding(use_gpu)
        sparse_model = (
            cls.create_sparse_embedding(use_gpu)
            if settings.enable_sparse_embeddings
            else None
        )

        embedding_types = ["dense"]
        if sparse_model:
            embedding_types.append("sparse")

        logger.info(f"Hybrid embeddings created: {', '.join(embedding_types)}")
        return dense_model, sparse_model

    @classmethod
    def clear_cache(cls):
        """Clear the LRU cache for embedding models.

        Clears cached embedding models to force recreation on next access.
        Useful for testing or when changing embedding configurations.

        Example:
            >>> EmbeddingFactory.clear_cache()
            >>> # Next call will create fresh models
        """
        cls.create_dense_embedding.cache_clear()
        cls.create_sparse_embedding.cache_clear()
        logger.info("Embedding model cache cleared")

    @classmethod
    def get_cache_info(cls) -> dict:
        """Get cache information for debugging.

        Returns cache statistics for both dense and sparse embedding models
        to help with debugging and optimization.

        Returns:
            Dictionary containing cache hit/miss statistics.

        Example:
            >>> cache_info = EmbeddingFactory.get_cache_info()
            >>> print(f"Dense cache hits: {cache_info['dense']['hits']}")
        """
        return {
            "dense": cls.create_dense_embedding.cache_info()._asdict(),
            "sparse": cls.create_sparse_embedding.cache_info()._asdict(),
        }


# Rate-limited wrapper functions for embedding operations


# Rate limiting functions removed - not needed for local document processing app
