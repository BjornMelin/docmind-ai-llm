"""Unified embeddings module for BGE-M3 and CLIP models.

This module consolidates BGE-M3 unified embedding and CLIP multimodal functionality
from the previous nested structure, implementing ADR-002 embedding strategy.

Key features:
- BGE-M3 unified dense (1024D) + sparse embeddings
- CLIP multimodal image-text embeddings with VRAM constraints
- Factory functions for easy instantiation
- FP16 acceleration for RTX 4090 optimization
"""

import asyncio
from typing import Any

import torch
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, ValidationInfo, field_validator
from llama_index.embeddings.clip import ClipEmbedding
from loguru import logger
from pydantic import BaseModel

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    logger.error("FlagEmbedding not available. Install with: uv add FlagEmbedding")
    BGEM3FlagModel = None


# BGE-M3 and retrieval configuration constants (moved from src.config)
class EmbeddingConfig(BaseModel):
    """BGE-M3 embedding configuration."""

    model_name: str = Field(default="BAAI/bge-m3")
    dimension: int = Field(default=1024, ge=256, le=4096)
    max_length: int = Field(default=8192, ge=512, le=16384)
    batch_size_gpu: int = Field(default=12, ge=1, le=128)
    batch_size_cpu: int = Field(default=4, ge=1, le=32)


# Global embedding settings instance
embedding_settings = EmbeddingConfig()

# CLIP configuration constants
VRAM_PER_IMAGE_GB = 0.14  # ~140MB per image for ViT-B/32
DEFAULT_CLIP_BATCH_SIZE = 10  # Optimized batch size for 1.4GB VRAM limit
MAX_VRAM_GB_LIMIT = 1.4  # Maximum VRAM allocation for CLIP operations


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

    model_name: str = Field(default=embedding_settings.model_name)
    max_length: int = Field(default=embedding_settings.max_length)
    use_fp16: bool = Field(default=True)
    batch_size: int = Field(default=embedding_settings.batch_size_gpu)
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
        model_name: str = embedding_settings.model_name,
        max_length: int = embedding_settings.max_length,
        use_fp16: bool = True,
        batch_size: int = embedding_settings.batch_size_gpu,
        device: str = "cuda",
        **kwargs: Any,
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
        return embedding_settings.dimension


class ClipConfig(BaseModel):
    """Configuration for CLIP multimodal embeddings with VRAM constraints."""

    model_name: str = Field(
        default="openai/clip-vit-base-patch32",
        description="CLIP model name (ViT-B/32)",
    )
    embed_batch_size: int = Field(
        default=DEFAULT_CLIP_BATCH_SIZE,
        description=(
            f"Batch size for embedding generation "
            f"(optimized for {MAX_VRAM_GB_LIMIT}GB VRAM)"
        ),
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device for model execution",
    )
    max_vram_gb: float = Field(
        default=MAX_VRAM_GB_LIMIT,
        description="Maximum VRAM usage in GB",
    )
    auto_adjust_batch: bool = Field(
        default=True,
        description="Automatically adjust batch size for VRAM constraints",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate CLIP model name."""
        supported_models = [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
        ]
        if v not in supported_models:
            raise ValueError(f"Unsupported model: {v}. Choose from {supported_models}")
        return v

    @field_validator("embed_batch_size")
    @classmethod
    def validate_batch_size(cls, v: int, info: ValidationInfo) -> int:
        """Validate batch size for VRAM constraints."""
        if (
            v > DEFAULT_CLIP_BATCH_SIZE
            and info.data.get("max_vram_gb", MAX_VRAM_GB_LIMIT) <= MAX_VRAM_GB_LIMIT
        ):
            logger.warning(
                f"Batch size {v} may exceed {MAX_VRAM_GB_LIMIT}GB VRAM limit, "
                f"adjusting to {DEFAULT_CLIP_BATCH_SIZE}"
            )
            return DEFAULT_CLIP_BATCH_SIZE
        return v

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return (
            self.model_name
            in ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16"]
            and self.embed_batch_size <= DEFAULT_CLIP_BATCH_SIZE
            and self.max_vram_gb <= MAX_VRAM_GB_LIMIT
        )

    def optimize_for_hardware(self) -> "ClipConfig":
        """Optimize configuration for hardware constraints."""
        if not self.auto_adjust_batch:
            return self

        # Estimate VRAM usage and adjust batch size
        if self.device == "cuda":
            max_batch = int(self.max_vram_gb / VRAM_PER_IMAGE_GB)
            if self.embed_batch_size > max_batch:
                logger.info(
                    f"Adjusting batch size from {self.embed_batch_size} to {max_batch} "
                    f"for {self.max_vram_gb}GB VRAM limit"
                )
                self.embed_batch_size = max_batch

        return self

    def estimated_vram_usage(self) -> float:
        """Estimate VRAM usage in GB."""
        return self.embed_batch_size * VRAM_PER_IMAGE_GB


def create_bgem3_embedding(
    model_name: str = embedding_settings.model_name,
    use_fp16: bool = True,
    device: str = "cuda",
    max_length: int = embedding_settings.max_length,
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
        embedding_settings.batch_size_gpu
        if device == "cuda"
        else embedding_settings.batch_size_cpu
    )

    return BGEM3Embedding(
        model_name=model_name,
        use_fp16=use_fp16,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )


def create_clip_embedding(config: dict | ClipConfig) -> ClipEmbedding:
    """Create CLIP embedding model with VRAM constraints.

    Args:
        config: CLIP configuration dict or ClipConfig object

    Returns:
        Configured ClipEmbedding instance
    """
    if isinstance(config, dict):
        config = ClipConfig(**config)

    # Optimize for hardware
    config = config.optimize_for_hardware()

    # Create CLIP embedding
    clip_embedding = ClipEmbedding(
        model_name=config.model_name,
        embed_batch_size=config.embed_batch_size,
        device=config.device,
    )

    logger.info(
        f"CLIP embedding created: {config.model_name} "
        f"(batch_size={config.embed_batch_size}, device={config.device})"
    )

    # Validate VRAM usage
    if config.device == "cuda":
        estimated_vram = config.estimated_vram_usage()
        if estimated_vram > config.max_vram_gb:
            logger.warning(
                f"Estimated VRAM usage ({estimated_vram:.2f}GB) "
                f"exceeds limit ({config.max_vram_gb}GB)"
            )

    return clip_embedding


def setup_clip_for_llamaindex(config: dict | None = None) -> ClipEmbedding:
    """Setup CLIP as the default embedding model for LlamaIndex.

    Args:
        config: Optional CLIP configuration

    Returns:
        Configured ClipEmbedding instance
    """
    if config is None:
        config = ClipConfig()
    else:
        config = ClipConfig(**config) if isinstance(config, dict) else config

    # Create CLIP embedding
    clip_embedding = create_clip_embedding(config)

    # Set as default embedding model
    Settings.embed_model = clip_embedding

    logger.info("CLIP set as default embedding model for LlamaIndex")
    return clip_embedding


def configure_bgem3_settings() -> None:
    """Configure LlamaIndex Settings for BGE-M3 unified embeddings.

    Updates global Settings singleton to use BGE-M3 as the default
    embedding model, replacing any existing BGE-large configuration.
    """
    try:
        bgem3_model = create_bgem3_embedding()
        Settings.embed_model = bgem3_model
        logger.info("LlamaIndex Settings configured for BGE-M3 unified embeddings")
    except (ImportError, RuntimeError, ValueError) as e:
        logger.error("Failed to configure BGE-M3 settings: %s", e)
        raise
from src.models.embeddings import UnifiedEmbedder



# === New unified embedder route (SPEC-003) ===
def get_unified_embedder() -> UnifiedEmbedder:
    """Factory for the new UnifiedEmbedder.

    Returns a router exposing text (BGE‑M3 dense+sparse) and image encoders
    (OpenCLIP/SigLIP-tiered). The existing BGEM3Embedding and CLIP helpers
    remain for compatibility with LlamaIndex wrappers and tests.
    """

    return UnifiedEmbedder()
