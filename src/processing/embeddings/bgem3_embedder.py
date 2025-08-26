"""BGE-M3 8K Context Embedder for Document Processing Pipeline.

This module implements a specialized BGE-M3 embedder optimized for the document
processing pipeline with 8K context window, batch processing, and semantic
caching integration per ADR-009 requirements.

Key Features:
- BGE-M3 unified dense/sparse embeddings with 8K context window
- Optimal batch processing for document chunks
- GPU memory optimization for RTX 4090
- Semantic similarity caching integration
- Async processing with memory management
- Performance targets: <50ms per chunk, <3GB VRAM usage
"""

import asyncio
import gc
import time
from typing import Any

import numpy as np
import torch
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# BGE-M3 model import with fallback
try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    logger.error("FlagEmbedding not available. Install with: uv add FlagEmbedding")
    BGEM3FlagModel = None

from src.config.settings import app_settings
from src.processing.embeddings.models import (
    EmbeddingError,
    EmbeddingParameters,
    EmbeddingResult,
)


class BGEM3Embedder:
    """BGE-M3 8K context embedder optimized for document processing pipeline.

    This embedder is specifically designed for the document processing pipeline
    with 8K context window support, optimal batch processing, and semantic
    caching integration. It provides:

    - BGE-M3 unified dense/sparse embeddings
    - 8K context window for large document chunks
    - GPU memory optimization for RTX 4090
    - Batch processing with automatic memory management
    - Semantic similarity caching support
    - Async processing for non-blocking operations
    """

    def __init__(
        self,
        settings: Any | None = None,
        parameters: EmbeddingParameters | None = None,
    ):
        """Initialize BGEM3Embedder.

        Args:
            settings: DocMind configuration settings. Uses app_settings if None.
            parameters: Embedding parameters. Uses defaults if None.
        """
        self.settings = settings or app_settings
        self.parameters = parameters or EmbeddingParameters()

        # Model instance will be lazily loaded
        self._model: Any | None = None
        self._model_loaded = False

        # Performance tracking
        self._embedding_count = 0
        self._total_processing_time = 0.0
        self._peak_memory_mb = 0.0

        # Device detection and optimization
        self.device = self._detect_optimal_device()
        self.parameters.device = self.device

        # Adjust batch size based on device
        if self.device == "cuda" and torch.cuda.is_available():
            self.parameters.batch_size_gpu = self._optimize_batch_size()

        logger.info(
            "BGEM3Embedder initialized: device={}, max_length={}, batch_size={}",
            self.device,
            self.parameters.max_length,
            self._get_optimal_batch_size(),
        )

    def _detect_optimal_device(self) -> str:
        """Detect optimal device for BGE-M3 processing.

        Returns:
            Device string ('cuda', 'cpu', or specific GPU)
        """
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            return "cpu"

        # Check GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 8.0:
            logger.warning(
                f"GPU has {gpu_memory_gb:.1f}GB memory. BGE-M3 may use CPU fallback."
            )
            return "cpu"

        logger.info(f"Using CUDA with {gpu_memory_gb:.1f}GB GPU memory")
        return "cuda"

    def _optimize_batch_size(self) -> int:
        """Optimize batch size based on available GPU memory.

        Returns:
            Optimal batch size for current GPU
        """
        if self.device == "cpu":
            return self.parameters.batch_size_cpu

        try:
            # Get available GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # BGE-M3 with 8K context requires ~2GB base + ~250MB per batch item
            if gpu_memory_gb >= 16.0:
                return 12  # RTX 4090 optimal
            elif gpu_memory_gb >= 12.0:
                return 8  # RTX 4060 Ti
            elif gpu_memory_gb >= 8.0:
                return 4  # RTX 4060
            else:
                return 2  # Minimal GPU
        except Exception:
            logger.warning("Could not optimize batch size, using default")
            return self.parameters.batch_size_gpu

    def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size for current device.

        Returns:
            Batch size to use for current device
        """
        if self.device == "cuda":
            return self.parameters.batch_size_gpu
        else:
            return self.parameters.batch_size_cpu

    def _load_model(self) -> None:
        """Load BGE-M3 model with error handling and optimization.

        Raises:
            EmbeddingError: If model loading fails
        """
        if self._model_loaded and self._model is not None:
            return

        if BGEM3FlagModel is None:
            raise EmbeddingError(
                "FlagEmbedding not available. Install with: uv add FlagEmbedding>=1.3.5"
            )

        try:
            logger.info(f"Loading BGE-M3 model: {self.settings.bge_m3_model_name}")

            # Load model with optimizations
            self._model = BGEM3FlagModel(
                model_name_or_path=self.settings.bge_m3_model_name,
                use_fp16=self.parameters.use_fp16,
                device=self.device,
            )

            self._model_loaded = True

            # Model successfully loaded - log information

            logger.info(
                "BGE-M3 model loaded successfully: device={}, fp16={}, max_length={}",
                self.device,
                self.parameters.use_fp16,
                self.parameters.max_length,
            )

        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {str(e)}")
            raise EmbeddingError(f"Model loading failed: {e}") from e

    def _track_memory_usage(self) -> float:
        """Track current GPU memory usage.

        Returns:
            Current GPU memory usage in MB
        """
        if self.device == "cuda" and torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024**2)
            self._peak_memory_mb = max(self._peak_memory_mb, memory_mb)
            return memory_mb
        return 0.0

    def _cleanup_memory(self) -> None:
        """Clean up GPU memory after processing."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @retry(
        retry=retry_if_exception_type(
            (EmbeddingError, RuntimeError, torch.cuda.OutOfMemoryError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_texts_async(
        self, texts: list[str], parameters: EmbeddingParameters | None = None
    ) -> EmbeddingResult:
        """Embed texts asynchronously with 8K context support.

        Args:
            texts: List of text strings to embed
            parameters: Optional embedding parameters override

        Returns:
            EmbeddingResult with dense/sparse embeddings and metadata

        Raises:
            EmbeddingError: If embedding processing fails
        """
        start_time = time.time()
        embed_params = parameters or self.parameters

        logger.info(f"Embedding {len(texts)} texts with 8K context window")

        if not texts:
            logger.warning("No texts provided for embedding")
            return EmbeddingResult(
                dense_embeddings=[],
                sparse_embeddings=[],
                processing_time=time.time() - start_time,
                batch_size=0,
                memory_usage_mb=0.0,
                model_info={"warning": "No texts provided"},
            )

        try:
            # Ensure model is loaded
            if not self._model_loaded:
                self._load_model()

            # Process embeddings in thread pool
            result = await asyncio.to_thread(
                self._embed_texts_sync, texts, embed_params
            )

            # Track performance
            processing_time = time.time() - start_time
            self._embedding_count += len(texts)
            self._total_processing_time += processing_time

            # Update result with timing
            result.processing_time = processing_time

            logger.info(
                f"Successfully embedded {len(texts)} texts in {processing_time:.2f}s "
                f"(avg: {processing_time / len(texts) * 1000:.1f}ms per text)"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Failed to embed {len(texts)} texts after {processing_time:.2f}s: "
                f"{str(e)}"
            )

            # Clean up on error
            self._cleanup_memory()

            if "out of memory" in str(e).lower() and embed_params.batch_size_gpu > 2:
                logger.info("Retrying with smaller batch size due to OOM")
                smaller_params = embed_params.model_copy()
                smaller_params.batch_size_gpu = max(2, embed_params.batch_size_gpu // 2)
                return await self.embed_texts_async(texts, smaller_params)

            raise EmbeddingError(f"Text embedding failed: {e}") from e

    def _embed_texts_sync(
        self, texts: list[str], parameters: EmbeddingParameters
    ) -> EmbeddingResult:
        """Synchronous text embedding with BGE-M3.

        Args:
            texts: List of text strings to embed
            parameters: Embedding parameters

        Returns:
            EmbeddingResult with embeddings and metadata

        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            # Track initial memory
            initial_memory = self._track_memory_usage()

            batch_size = self._get_optimal_batch_size()

            logger.debug(
                f"Embedding {len(texts)} texts with batch_size={batch_size}, "
                f"max_length={parameters.max_length}"
            )

            # Process in batches for memory efficiency
            all_dense_embeddings = []
            all_sparse_embeddings = []
            all_colbert_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Call BGE-M3 encode with 8K context
                embeddings = self._model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    max_length=parameters.max_length,
                    return_dense=parameters.return_dense,
                    return_sparse=parameters.return_sparse,
                    return_colbert_vecs=parameters.return_colbert,
                    request_qid=None,  # Not needed for document processing
                    request_pid=None,  # Not needed for document processing
                )

                # Extract embeddings by type
                if parameters.return_dense and "dense_vecs" in embeddings:
                    dense_batch = embeddings["dense_vecs"]
                    if parameters.normalize_embeddings:
                        # L2 normalize dense embeddings
                        dense_batch = dense_batch / np.linalg.norm(
                            dense_batch, axis=1, keepdims=True
                        )
                    all_dense_embeddings.extend(dense_batch.tolist())

                if parameters.return_sparse and "lexical_weights" in embeddings:
                    all_sparse_embeddings.extend(embeddings["lexical_weights"])

                if parameters.return_colbert and "colbert_vecs" in embeddings:
                    all_colbert_embeddings.extend(embeddings["colbert_vecs"])

                # Track memory usage
                self._track_memory_usage()

                # Optional memory cleanup between batches for large datasets
                if len(texts) > 50:
                    torch.cuda.empty_cache() if self.device == "cuda" else None

            # Track final memory
            final_memory = self._track_memory_usage()
            memory_usage_mb = final_memory - initial_memory

            # Build result
            result = EmbeddingResult(
                dense_embeddings=all_dense_embeddings
                if parameters.return_dense
                else None,
                sparse_embeddings=all_sparse_embeddings
                if parameters.return_sparse
                else None,
                colbert_embeddings=all_colbert_embeddings
                if parameters.return_colbert
                else None,
                processing_time=0.0,  # Will be set by caller
                batch_size=batch_size,
                memory_usage_mb=memory_usage_mb,
                model_info={
                    "model_name": self.settings.bge_m3_model_name,
                    "device": self.device,
                    "max_length": parameters.max_length,
                    "embedding_dim": 1024,
                    "fp16_enabled": parameters.use_fp16,
                },
            )

            # Clean up memory
            self._cleanup_memory()

            return result

        except Exception as e:
            logger.error(f"BGE-M3 encode failed: {str(e)}")
            self._cleanup_memory()
            raise EmbeddingError(f"BGE-M3 embedding failed: {e}") from e

    async def embed_single_text_async(self, text: str) -> list[float]:
        """Embed single text and return dense embedding vector.

        Args:
            text: Text string to embed

        Returns:
            Dense embedding vector (1024D)

        Raises:
            EmbeddingError: If embedding fails
        """
        result = await self.embed_texts_async([text])
        if result.dense_embeddings:
            return result.dense_embeddings[0]
        else:
            raise EmbeddingError("No dense embeddings returned")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the embedder.

        Returns:
            Dictionary with performance metrics
        """
        avg_time_per_text = (
            self._total_processing_time / self._embedding_count
            if self._embedding_count > 0
            else 0.0
        )

        return {
            "total_texts_embedded": self._embedding_count,
            "total_processing_time": self._total_processing_time,
            "avg_time_per_text_ms": avg_time_per_text * 1000,
            "peak_memory_usage_mb": self._peak_memory_mb,
            "device": self.device,
            "model_loaded": self._model_loaded,
            "optimal_batch_size": self._get_optimal_batch_size(),
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._embedding_count = 0
        self._total_processing_time = 0.0
        self._peak_memory_mb = 0.0
        logger.info("Performance statistics reset")

    def unload_model(self) -> None:
        """Unload model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False
            self._cleanup_memory()
            logger.info("BGE-M3 model unloaded and memory cleaned")


# Factory function for easy instantiation
def create_bgem3_embedder(
    settings: Any | None = None, parameters: EmbeddingParameters | None = None
) -> BGEM3Embedder:
    """Factory function to create BGEM3Embedder instance.

    Args:
        settings: Optional DocMind settings. Uses app_settings if None.
        parameters: Optional embedding parameters. Uses defaults if None.

    Returns:
        Configured BGEM3Embedder instance
    """
    return BGEM3Embedder(settings, parameters)
