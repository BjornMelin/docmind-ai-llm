"""Embedding model manager for DocMind AI.

This module provides a singleton for managing embedding models to prevent
redundant initializations, supporting text and multimodal embeddings.

Classes:
    ModelManager: Singleton for embedding model management.
"""

from typing import Any

try:
    from fastembed import LateInteractionMultimodalEmbedding
except ImportError:
    LateInteractionMultimodalEmbedding = None

try:
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
except ImportError:
    FastEmbedEmbedding = None


class ModelManager:
    """Singleton for managing embedding models to prevent redundant initializations."""

    _instance = None
    _text_model = None
    _multimodal_model = None

    def __new__(cls):
        """Create a new instance of the class if it doesn't exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_text_embedding_model(self, model_name: str) -> Any:
        """Get or initialize text embedding model."""
        if self._text_model is None:
            if FastEmbedEmbedding is not None:
                self._text_model = FastEmbedEmbedding(model_name)
            else:
                # Return None if FastEmbedEmbedding is not available
                return None
        return self._text_model

    def get_multimodal_embedding_model(self) -> Any:
        """Get or initialize multimodal embedding model."""
        if self._multimodal_model is None:
            if LateInteractionMultimodalEmbedding is not None:
                self._multimodal_model = LateInteractionMultimodalEmbedding(
                    model_name="Qdrant/jina-clip-v1",
                    max_length=512,
                )
            else:
                # Return None if LateInteractionMultimodalEmbedding is not available
                return None
        return self._multimodal_model
