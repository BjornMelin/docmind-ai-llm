"""CLIP multimodal embedding configuration for REQ-0044.

This module provides CLIP ViT-B/32 embedding configuration with VRAM constraints,
optimized for RTX 4090 with 1.4GB VRAM limit for CLIP operations.

Library-first implementation using llama-index-embeddings-clip.
"""

import torch
from llama_index.core import Settings
from llama_index.embeddings.clip import ClipEmbedding
from loguru import logger
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ClipConfig(BaseModel):
    """Configuration for CLIP multimodal embeddings with VRAM constraints."""

    model_name: str = Field(
        default="openai/clip-vit-base-patch32",
        description="CLIP model name (ViT-B/32)",
    )
    embed_batch_size: int = Field(
        default=10,
        description="Batch size for embedding generation (optimized for 1.4GB VRAM)",
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device for model execution",
    )
    max_vram_gb: float = Field(
        default=1.4,
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
        if v > 10 and info.data.get("max_vram_gb", 1.4) <= 1.4:
            logger.warning(
                f"Batch size {v} may exceed 1.4GB VRAM limit, adjusting to 10"
            )
            return 10
        return v

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return (
            self.model_name
            in ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16"]
            and self.embed_batch_size <= 10
            and self.max_vram_gb <= 1.4
        )

    def optimize_for_hardware(self) -> "ClipConfig":
        """Optimize configuration for hardware constraints."""
        if not self.auto_adjust_batch:
            return self

        # Estimate VRAM usage and adjust batch size
        if self.device == "cuda":
            vram_per_image = 0.14  # ~140MB per image for ViT-B/32
            max_batch = int(self.max_vram_gb / vram_per_image)
            if self.embed_batch_size > max_batch:
                logger.info(
                    f"Adjusting batch size from {self.embed_batch_size} to {max_batch} "
                    f"for {self.max_vram_gb}GB VRAM limit"
                )
                self.embed_batch_size = max_batch

        return self

    def estimated_vram_usage(self) -> float:
        """Estimate VRAM usage in GB."""
        vram_per_image = 0.14  # ~140MB per image for ViT-B/32
        return self.embed_batch_size * vram_per_image


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
