"""BGE-M3 Complete Unified Embedder with Dense AND Sparse Support.

Uses FlagEmbedding library directly to provide BGE-M3's
unified dense, sparse, and ColBERT embeddings.

Key Features:
- BGE-M3 dense embeddings (1024D) for semantic similarity
- BGE-M3 sparse embeddings (learned token weights) for keyword matching
- Optional ColBERT embeddings for late interaction
- 8K context window support
- Built-in batch processing and GPU memory optimization
- Automatic device detection and FP16 acceleration
"""

import time
from typing import Any

import torch
from loguru import logger

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    logger.error(
        "FlagEmbedding not available. Install with: uv add FlagEmbedding>=1.3.5"
    )
    BGEM3FlagModel = None

from src.models.embeddings import (
    EmbeddingError,
    EmbeddingParameters,
    EmbeddingResult,
)


class BGEM3Embedder:
    """Complete BGE-M3 embedder with unified dense AND sparse embedding support.

    Provides both dense and sparse embeddings from BGE-M3, enabling true hybrid search.

    Features:
    - Dense embeddings (1024D) for semantic similarity
    - Sparse embeddings (learned token weights) for keyword matching
    - Optional ColBERT embeddings for late interaction
    - 8K context window support
    - FP16 acceleration for RTX 4090 optimization
    - Batch processing with automatic device detection
    """

    def __init__(
        self,
        settings: Any | None = None,
        parameters: EmbeddingParameters | None = None,
        *,  # Force keyword-only arguments after this point
        pooling_method: str = "cls",
        normalize_embeddings: bool = True,
        weights_for_different_modes: list[float] | None = None,
        devices: list[str] | None = None,
        return_numpy: bool = False,
    ):
        """Initialize BGE-M3 embedder with full library capabilities.

        Args:
            settings: Application settings
            parameters: Embedding parameters
            pooling_method: Pooling method ('cls', 'mean'). Default: 'cls'
            normalize_embeddings: Whether to normalize embeddings. Default: True
            weights_for_different_modes: Weights for [dense, sparse, colbert] fusion.
                Default: [0.4, 0.2, 0.4]
            devices: Specific devices to use. Default: auto-detect
            return_numpy: Whether to return numpy arrays. Default: False (returns lists)
        """
        self.settings = settings
        self.parameters = parameters or EmbeddingParameters()
        self.pooling_method = pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.weights_for_different_modes = weights_for_different_modes or [
            0.4,
            0.2,
            0.4,
        ]
        self.return_numpy = return_numpy

        # Check FlagEmbedding availability
        if BGEM3FlagModel is None:
            raise EmbeddingError(
                "FlagEmbedding not available. Install with: uv add FlagEmbedding>=1.3.5"
            )

        # Auto device detection (let FlagEmbedding handle optimization)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_list = devices or [device]

        # Initialize BGE-M3 FlagModel with full library capabilities
        try:
            self.model = BGEM3FlagModel(
                model_name_or_path=self.settings.embedding.model_name,
                use_fp16=self.parameters.use_fp16 and device == "cuda",
                pooling_method=self.pooling_method,
                devices=device_list,
            )
            self.device = device
            # Remove custom batch size - let library handle optimization
            self._embedding_count = 0
            self._total_processing_time = 0.0

            logger.info(
                "BGE-M3 FlagModel loaded: %s (device=%s, FP16=%s, pooling=%s, "
                "normalize=%s)",
                self.settings.embedding.model_name,
                device_list,
                self.parameters.use_fp16 and device == "cuda",
                self.pooling_method,
                self.normalize_embeddings,
            )
        except Exception as e:
            raise EmbeddingError(f"BGE-M3 model initialization failed: {e}") from e

    async def embed_texts_async(
        self, texts: list[str], parameters: EmbeddingParameters | None = None
    ) -> EmbeddingResult:
        """Embed texts asynchronously with unified BGE-M3 dense + sparse embeddings."""
        if not texts:
            return EmbeddingResult(
                dense_embeddings=[],
                sparse_embeddings=None,
                processing_time=0.0,
                batch_size=0,
                memory_usage_mb=0.0,
                model_info={"warning": "No texts provided"},
            )

        params = parameters or self.parameters
        start_time = time.time()

        try:
            # Use BGE-M3 unified encoding with library optimization
            embeddings = self.model.encode(
                texts,
                max_length=params.max_length,
                return_dense=params.return_dense,
                return_sparse=params.return_sparse,
                return_colbert_vecs=params.return_colbert,
                # Let library handle batch size optimization
            )

            processing_time = time.time() - start_time
            self._embedding_count += len(texts)
            self._total_processing_time += processing_time

            # Extract dense embeddings (convert numpy to lists)
            dense_embeddings = None
            if params.return_dense and "dense_vecs" in embeddings:
                dense_embeddings = embeddings["dense_vecs"].tolist()

            # Extract sparse embeddings (token weights)
            sparse_embeddings = None
            if params.return_sparse and "lexical_weights" in embeddings:
                sparse_embeddings = embeddings["lexical_weights"]

            # Extract ColBERT embeddings if requested
            colbert_embeddings = None
            if params.return_colbert and "colbert_vecs" in embeddings:
                colbert_embeddings = embeddings["colbert_vecs"]

            memory_usage = (
                torch.cuda.memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else 0.0
            )

            # Convert to numpy if requested
            if self.return_numpy and dense_embeddings:
                import numpy as np

                dense_embeddings = np.array(dense_embeddings).tolist()

            return EmbeddingResult(
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
                colbert_embeddings=colbert_embeddings,
                processing_time=processing_time,
                batch_size=len(texts),  # Use actual batch size
                memory_usage_mb=memory_usage,
                model_info={
                    "model_name": self.settings.embedding.model_name,
                    "device": str(self.device),
                    "embedding_dim": 1024,
                    "library": "FlagEmbedding.BGEM3FlagModel",
                    "pooling_method": self.pooling_method,
                    "normalize_embeddings": self.normalize_embeddings,
                    "dense_enabled": params.return_dense,
                    "sparse_enabled": params.return_sparse,
                    "colbert_enabled": params.return_colbert,
                    "weights_for_modes": self.weights_for_different_modes,
                },
            )
        except Exception as e:
            raise EmbeddingError(f"BGE-M3 unified embedding failed: {e}") from e

    async def embed_single_text_async(self, text: str) -> list[float]:
        """Embed single text and return dense embedding vector.

        Note: This method only returns dense embeddings for backward compatibility.
        Use embed_texts_async() to get both dense and sparse embeddings.
        """
        try:
            result = await self.embed_texts_async([text])
            if result.dense_embeddings and len(result.dense_embeddings) > 0:
                return result.dense_embeddings[0]
            raise EmbeddingError("No dense embeddings generated")
        except Exception as e:
            raise EmbeddingError(f"Single text embedding failed: {e}") from e

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        avg_time_per_text = (
            self._total_processing_time / self._embedding_count
            if self._embedding_count > 0
            else 0.0
        )
        return {
            "total_texts_embedded": self._embedding_count,
            "total_processing_time": self._total_processing_time,
            "avg_time_per_text_ms": avg_time_per_text * 1000,
            "device": self.device,
            "model_library": "FlagEmbedding.BGEM3FlagModel",
            "unified_embeddings_enabled": True,
            "pooling_method": self.pooling_method,
            "normalize_embeddings": self.normalize_embeddings,
            "weights_for_modes": self.weights_for_different_modes,
            "library_optimization": True,
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._embedding_count = 0
        self._total_processing_time = 0.0

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict[int, float]] | None:
        """Get sparse embeddings only for the provided texts.

        Args:
            texts: List of texts to get sparse embeddings for

        Returns:
            List of sparse embedding dictionaries (token_id -> weight)
        """
        try:
            embeddings = self.model.encode(
                texts,
                max_length=self.parameters.max_length,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )

            if "lexical_weights" in embeddings:
                return embeddings["lexical_weights"]
            return None
        except (RuntimeError, ValueError, ImportError) as e:
            logger.error(f"Failed to get sparse embeddings: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting sparse embeddings: {e}")
            return None

    def get_dense_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        """Get dense embeddings only for the provided texts.

        Args:
            texts: List of texts to get dense embeddings for

        Returns:
            List of 1024-dimensional dense embedding vectors
        """
        try:
            embeddings = self.model.encode(
                texts,
                max_length=self.parameters.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )

            if "dense_vecs" in embeddings:
                return embeddings["dense_vecs"].tolist()
            return None
        except (RuntimeError, ValueError, ImportError) as e:
            logger.error(f"Failed to get dense embeddings: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting dense embeddings: {e}")
            return None

    def compute_sparse_similarity(
        self, sparse1: dict[int, float], sparse2: dict[int, float]
    ) -> float:
        """Compute lexical matching score between two sparse embeddings.

        Args:
            sparse1: First sparse embedding
            sparse2: Second sparse embedding

        Returns:
            Sparse similarity score
        """
        try:
            return self.model.compute_lexical_matching_score(sparse1, sparse2)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Failed to compute sparse similarity: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error computing sparse similarity: {e}")
            return 0.0

    async def encode_queries(
        self, queries: list[str], parameters: EmbeddingParameters | None = None
    ) -> EmbeddingResult:
        """Encode queries using BGEM3FlagModel's optimized query encoding.

        Uses FlagEmbedding's encode_queries() for query-optimized encoding.
        This method applies query-specific optimizations for better retrieval
        performance.

        Args:
            queries: List of query texts to encode
            parameters: Optional embedding parameters

        Returns:
            EmbeddingResult with query-optimized embeddings
        """
        if not queries:
            return EmbeddingResult(
                dense_embeddings=[],
                sparse_embeddings=None,
                processing_time=0.0,
                batch_size=0,
                memory_usage_mb=0.0,
                model_info={"warning": "No queries provided"},
            )

        params = parameters or self.parameters
        start_time = time.time()

        try:
            # Use library's query-optimized encoding
            embeddings = self.model.encode_queries(
                queries,
                max_length=params.max_length,
                return_dense=params.return_dense,
                return_sparse=params.return_sparse,
                return_colbert_vecs=params.return_colbert,
            )

            processing_time = time.time() - start_time
            self._embedding_count += len(queries)
            self._total_processing_time += processing_time

            # Process embeddings same as embed_texts_async
            dense_embeddings = None
            if params.return_dense and "dense_vecs" in embeddings:
                dense_embeddings = embeddings["dense_vecs"].tolist()

            sparse_embeddings = None
            if params.return_sparse and "lexical_weights" in embeddings:
                sparse_embeddings = embeddings["lexical_weights"]

            colbert_embeddings = None
            if params.return_colbert and "colbert_vecs" in embeddings:
                colbert_embeddings = embeddings["colbert_vecs"]

            memory_usage = (
                torch.cuda.memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else 0.0
            )

            return EmbeddingResult(
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
                colbert_embeddings=colbert_embeddings,
                processing_time=processing_time,
                batch_size=len(queries),
                memory_usage_mb=memory_usage,
                model_info={
                    "model_name": self.settings.embedding.model_name,
                    "device": str(self.device),
                    "embedding_dim": 1024,
                    "library": "FlagEmbedding.BGEM3FlagModel.encode_queries",
                    "optimization": "query-optimized",
                    "pooling_method": self.pooling_method,
                },
            )
        except Exception as e:
            raise EmbeddingError(f"BGE-M3 query encoding failed: {e}") from e

    async def encode_corpus(
        self, corpus: list[str], parameters: EmbeddingParameters | None = None
    ) -> EmbeddingResult:
        """Encode corpus using BGEM3FlagModel's optimized corpus encoding.

        Uses FlagEmbedding's encode_corpus() for document-optimized encoding.
        This method applies corpus-specific optimizations for better indexing
        performance.

        Args:
            corpus: List of corpus texts to encode
            parameters: Optional embedding parameters

        Returns:
            EmbeddingResult with corpus-optimized embeddings
        """
        if not corpus:
            return EmbeddingResult(
                dense_embeddings=[],
                sparse_embeddings=None,
                processing_time=0.0,
                batch_size=0,
                memory_usage_mb=0.0,
                model_info={"warning": "No corpus provided"},
            )

        params = parameters or self.parameters
        start_time = time.time()

        try:
            # Use library's corpus-optimized encoding
            embeddings = self.model.encode_corpus(
                corpus,
                max_length=params.max_length,
                return_dense=params.return_dense,
                return_sparse=params.return_sparse,
                return_colbert_vecs=params.return_colbert,
            )

            processing_time = time.time() - start_time
            self._embedding_count += len(corpus)
            self._total_processing_time += processing_time

            # Process embeddings same as embed_texts_async
            dense_embeddings = None
            if params.return_dense and "dense_vecs" in embeddings:
                dense_embeddings = embeddings["dense_vecs"].tolist()

            sparse_embeddings = None
            if params.return_sparse and "lexical_weights" in embeddings:
                sparse_embeddings = embeddings["lexical_weights"]

            colbert_embeddings = None
            if params.return_colbert and "colbert_vecs" in embeddings:
                colbert_embeddings = embeddings["colbert_vecs"]

            memory_usage = (
                torch.cuda.memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else 0.0
            )

            return EmbeddingResult(
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
                colbert_embeddings=colbert_embeddings,
                processing_time=processing_time,
                batch_size=len(corpus),
                memory_usage_mb=memory_usage,
                model_info={
                    "model_name": self.settings.embedding.model_name,
                    "device": str(self.device),
                    "embedding_dim": 1024,
                    "library": "FlagEmbedding.BGEM3FlagModel.encode_corpus",
                    "optimization": "corpus-optimized",
                    "pooling_method": self.pooling_method,
                },
            )
        except Exception as e:
            raise EmbeddingError(f"BGE-M3 corpus encoding failed: {e}") from e

    def compute_similarity(
        self,
        texts1: list[str],
        texts2: list[str],
        mode: str = "hybrid",
        max_passage_length: int = 8192,
    ) -> dict[str, list[float]]:
        """Compute similarity using BGEM3FlagModel's built-in similarity computation.

        Uses the library's compute_score() method for comprehensive similarity analysis.

        Args:
            texts1: First set of texts (queries)
            texts2: Second set of texts (passages)
            mode: Similarity mode ('dense', 'sparse', 'colbert', 'hybrid')
            max_passage_length: Maximum passage length for latency control

        Returns:
            Dictionary with similarity scores for different modes
        """
        try:
            # Create sentence pairs for compute_score
            sentence_pairs = [[q, p] for q in texts1 for p in texts2]

            # Use library's comprehensive similarity computation
            scores = self.model.compute_score(
                sentence_pairs,
                max_passage_length=max_passage_length,
                weights_for_different_modes=self.weights_for_different_modes,
            )

            return scores
        except (RuntimeError, ValueError, ImportError) as e:
            logger.error(f"Failed to compute similarity: {e}")
            return {"error": [0.0] * (len(texts1) * len(texts2))}
        except Exception as e:
            logger.error(f"Unexpected error computing similarity: {e}")
            return {"error": [0.0] * (len(texts1) * len(texts2))}

    def compute_colbert_similarity(
        self, colbert_vecs1: list, colbert_vecs2: list
    ) -> list[float]:
        """Compute ColBERT late-interaction similarity scores.

        Uses library's colbert_score() method for multi-vector similarity.

        Args:
            colbert_vecs1: First set of ColBERT vectors
            colbert_vecs2: Second set of ColBERT vectors

        Returns:
            List of ColBERT similarity scores
        """
        try:
            scores = []
            for vec1 in colbert_vecs1:
                for vec2 in colbert_vecs2:
                    score = self.model.colbert_score(vec1, vec2)
                    scores.append(float(score))
            return scores
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Failed to compute ColBERT similarity: {e}")
            return [0.0] * (len(colbert_vecs1) * len(colbert_vecs2))
        except Exception as e:
            logger.error(f"Unexpected error computing ColBERT similarity: {e}")
            return [0.0] * (len(colbert_vecs1) * len(colbert_vecs2))

    def get_sparse_embedding_tokens(
        self, sparse_embeddings: list[dict[int, float]]
    ) -> list[dict[str, float]]:
        """Convert sparse embedding IDs to tokens for debugging.

        Uses library's convert_id_to_token() method for token inspection.

        Args:
            sparse_embeddings: List of sparse embeddings with token IDs

        Returns:
            List of sparse embeddings with human-readable tokens
        """
        try:
            return self.model.convert_id_to_token(sparse_embeddings)
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Failed to convert sparse tokens: {e}")
            return [{} for _ in sparse_embeddings]
        except Exception as e:
            logger.error(f"Unexpected error converting sparse tokens: {e}")
            return [{} for _ in sparse_embeddings]

    def unload_model(self) -> None:
        """Unload BGE-M3 model and reset statistics."""
        self.reset_stats()
        if hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            # Move model to CPU to free GPU memory
            self.model.model.to("cpu")
        logger.info("BGE-M3 model unloaded")
