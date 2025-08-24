"""BGE-M3 unified embedding implementation for FEAT-002.

This module implements the complete architectural replacement of BGE-large + SPLADE++
with BGE-M3 unified dense/sparse embeddings per ADR-002.

Key features:
- BGE-M3 unified dense (1024D) + sparse embeddings in single model
- 8K context window (vs 512 in BGE-large)
- FP16 acceleration for RTX 4090 optimization
- Multilingual support (100+ languages)
- Native LlamaIndex integration via BaseEmbedding
"""

import asyncio
from typing import Any

from loguru import logger

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    logger.error("FlagEmbedding not available. Install with: uv add FlagEmbedding")
    BGEM3FlagModel = None

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field

from src.config.settings import settings


class BGEM3Embedding(BaseEmbedding):
    """BGE-M3 unified dense/sparse embedding model for DocMind AI.

    Provides unified dense, sparse, and multi-vector (ColBERT) embeddings
    in a single model, replacing BGE-Large + SPLADE++ combination per ADR-002.

    Features:
    - Unified dense (1024D) + sparse embeddings
    - ColBERT multi-vector support
    - FP16 acceleration for RTX 4090
    - 8192 token context window
    - Multilingual support (100+ languages)

    Performance targets (RTX 4090 Laptop):
    - <50ms per chunk embedding generation
    - <3GB VRAM usage (unified model)
    - 8K context vs 512 in BGE-large
    """

    model_name: str = Field(default=settings.bge_m3_model_name)
    max_length: int = Field(default=settings.bge_m3_max_length)
    use_fp16: bool = Field(default=True)
    batch_size: int = Field(default=settings.bge_m3_batch_size_gpu)
    normalize_embeddings: bool = Field(default=True)
    device: str = Field(default="cuda")

    # Private attributes for Pydantic v1
    _model: Any = None

    class Config:
        """Pydantic configuration for BGE-M3 embedding model.

        Allows arbitrary types (like model instances), forbids extra fields,
        and enables assignment validation for model updates.
        """

        arbitrary_types_allowed = True
        extra = "forbid"
        validate_assignment = True

    def __init__(
        self,
        *,
        model_name: str = settings.bge_m3_model_name,
        max_length: int = settings.bge_m3_max_length,
        use_fp16: bool = True,
        batch_size: int = settings.bge_m3_batch_size_gpu,
        device: str = "cuda",
        **kwargs,
    ):
        """Initialize BGE-M3 unified embedding model.

        Args:
            model_name: BGE-M3 model identifier
            max_length: Maximum token length (8K context)
            use_fp16: Enable FP16 acceleration
            batch_size: Batch size for RTX 4090 optimization
            device: Target device (cuda/cpu)
            **kwargs: Additional BaseEmbedding arguments
        """
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            use_fp16=use_fp16,
            batch_size=batch_size,
            device=device,
            **kwargs,
        )

        if BGEM3FlagModel is None:
            raise ImportError(
                "FlagEmbedding not available. Install with: uv add FlagEmbedding"
            )

        try:
            # Use object.__setattr__ to bypass Pydantic validation
            # for private attributes
            model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
            object.__setattr__(self, "_model", model)
            logger.info("BGE-M3 model loaded: %s (FP16: %s)", model_name, use_fp16)
        except (ImportError, RuntimeError, ValueError) as e:
            logger.error("Failed to load BGE-M3 model: %s", e)
            raise

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get dense embedding for single query.

        Args:
            query: Query text to embed

        Returns:
            1024-dimensional dense embedding vector
        """
        embeddings = self._model.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return embeddings["dense_vecs"][0].tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get dense embedding for single text.

        Args:
            text: Text to embed

        Returns:
            1024-dimensional dense embedding vector
        """
        return self._get_query_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async query embedding (fallback to sync).

        Args:
            query: Query text to embed

        Returns:
            1024-dimensional dense embedding vector
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)

    def get_unified_embeddings(
        self,
        texts: list[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
    ) -> dict[str, Any]:
        """Get unified dense/sparse/colbert embeddings.

        This is the core method that provides BGE-M3 unified capabilities,
        replacing separate BGE-large + SPLADE++ model calls.

        Args:
            texts: List of texts to embed
            return_dense: Return dense embeddings (1024D)
            return_sparse: Return sparse embeddings
            return_colbert: Return ColBERT multi-vector embeddings

        Returns:
            Dictionary containing requested embedding types:
            - 'dense': numpy array of 1024D vectors
            - 'sparse': list of sparse weight dictionaries
            - 'colbert': list of ColBERT multi-vectors
        """
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert,
            )

            result = {}
            if return_dense and "dense_vecs" in embeddings:
                result["dense"] = embeddings["dense_vecs"]
            if return_sparse and "lexical_weights" in embeddings:
                result["sparse"] = embeddings["lexical_weights"]
            if return_colbert and "colbert_vecs" in embeddings:
                result["colbert"] = embeddings["colbert_vecs"]

            logger.debug("Generated unified embeddings for %d texts", len(texts))
            return result

        except (RuntimeError, ValueError) as e:
            logger.error("Failed to generate unified embeddings: %s", e)
            raise

    def get_sparse_embedding(self, text: str) -> dict[int, float]:
        """Get sparse embedding for single text.

        Args:
            text: Text to embed

        Returns:
            Dictionary mapping token indices to weights
        """
        embeddings = self.get_unified_embeddings(
            [text], return_dense=False, return_sparse=True, return_colbert=False
        )
        if "sparse" in embeddings and len(embeddings["sparse"]) > 0:
            return embeddings["sparse"][0]
        return {}

    @property
    def embed_dim(self) -> int:
        """BGE-M3 dense embedding dimension."""
        return settings.bge_m3_embedding_dim


def create_bgem3_embedding(
    model_name: str = settings.bge_m3_model_name,
    use_fp16: bool = True,
    device: str = "cuda",
    max_length: int = settings.bge_m3_max_length,
) -> BGEM3Embedding:
    """Create BGE-M3 embedding instance with optimal settings for RTX 4090.

    Factory function following library-first principle for easy instantiation
    with performance-optimized defaults.

    Args:
        model_name: BGE-M3 model identifier
        use_fp16: Enable FP16 acceleration
        device: Target device (cuda/cpu)
        max_length: Maximum token length (8K context)

    Returns:
        Configured BGEM3Embedding instance optimized for RTX 4090 Laptop
    """
    # RTX 4090 optimized batch size
    batch_size = (
        settings.bge_m3_batch_size_gpu
        if device == "cuda"
        else settings.bge_m3_batch_size_cpu
    )

    return BGEM3Embedding(
        model_name=model_name,
        use_fp16=use_fp16,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )


# Settings integration helper
def configure_bgem3_settings() -> None:
    """Configure LlamaIndex Settings for BGE-M3 unified embeddings.

    Updates global Settings singleton to use BGE-M3 as the default
    embedding model, replacing any existing BGE-large configuration.
    """
    from llama_index.core import Settings

    try:
        bgem3_model = create_bgem3_embedding()
        Settings.embed_model = bgem3_model
        logger.info("LlamaIndex Settings configured for BGE-M3 unified embeddings")
    except (ImportError, RuntimeError, ValueError) as e:
        logger.error("Failed to configure BGE-M3 settings: %s", e)
        raise
