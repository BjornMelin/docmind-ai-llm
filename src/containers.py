"""Service factories for DocMind AI.

This module exposes a minimal, explicit set of factory functions for
constructing core services. It intentionally eliminates any historical
dependency-injection containers, wiring helpers, or other backward
compatibility shims in favor of simple, typed constructors.

All factories read sane defaults from the unified application settings in
``src.config.settings``. Each factory supports targeted overrides for clarity
and tests.

Examples:
  Get a coordinator with defaults::

      from src.containers import get_multi_agent_coordinator
      coordinator = get_multi_agent_coordinator()

  Create an embedding model on CPU::

      from src.containers import get_embedding_model
      embed = get_embedding_model(device="cpu")
"""

from __future__ import annotations

from typing import Any

from src.agents.coordinator import MultiAgentCoordinator
from src.config import settings
from src.processing.document_processor import DocumentProcessor
from src.retrieval.bge_m3_index import build_bge_m3_index, build_bge_m3_retriever


def get_embedding_model(*, nodes: list[Any] | None = None) -> Any:
    """Create and return the tri-mode BGEM3 retriever (index + retriever).

    This replaces the legacy embedding-model factory with a library-first
    index+retriever composition using LlamaIndex BGEM3Index.
    """
    idx = build_bge_m3_index(nodes or [], model_name=settings.embedding.model_name)
    return build_bge_m3_retriever(idx, weights_for_different_modes=[0.4, 0.2, 0.4])


def get_document_processor(*, config: Any | None = None) -> DocumentProcessor:
    """Create and return a DocumentProcessor.

    The processor applies Unstructured-first parsing and project-specific
    chunking according to unified settings.

    Args:
      config: Optional settings object. When omitted, the global settings are
        used.

    Returns:
      DocumentProcessor: A configured document processor instance.
    """
    return DocumentProcessor(settings=config or settings)


def get_multi_agent_coordinator(
    *,
    model_path: str | None = None,
    max_context_length: int | None = None,
    enable_fallback: bool = True,
) -> MultiAgentCoordinator:
    """Create and return the multi-agent coordinator.

    Initializes the ADR-compliant MultiAgentCoordinator using unified
    configuration, with optional overrides for model path and context tokens.

    Args:
      model_path: Optional model identifier/path override.
      max_context_length: Optional maximum context length (tokens).
      enable_fallback: Whether to allow fallback behavior on agent errors.

    Returns:
      MultiAgentCoordinator: Configured coordinator instance.
    """
    model_cfg = settings.get_model_config()
    selected_model = model_path or model_cfg.get("model_name")
    selected_ctx = max_context_length or model_cfg.get("context_window")

    return MultiAgentCoordinator(
        model_path=str(selected_model),
        max_context_length=int(selected_ctx),
        enable_fallback=bool(enable_fallback),
    )


__all__ = [
    "get_document_processor",
    "get_embedding_model",
    "get_multi_agent_coordinator",
]
