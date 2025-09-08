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
from src.retrieval.query_engine import ServerHybridRetriever, _HybridParams


def get_embedding_model(*, nodes: list[Any] | None = None) -> Any:
    """Return the default hybrid retriever (ServerHybridRetriever).

    Note: The factory name is kept for backwards import compatibility in tests,
    but this returns the final architecture's hybrid retriever configured from
    unified settings (Qdrant Query API, named vectors, server-side fusion).
    """
    try:
        rconf = settings.retrieval
        params = _HybridParams(
            collection=settings.database.qdrant_collection,
            fused_top_k=int(getattr(rconf, "fused_top_k", 60)),
            prefetch_sparse=400,
            prefetch_dense=200,
            fusion_mode=str(getattr(rconf, "fusion_mode", "rrf")),
            dedup_key=str(getattr(rconf, "dedup_key", "page_id")),
        )
    except Exception:  # pragma: no cover - defensive defaults
        params = _HybridParams(
            collection=getattr(settings.database, "qdrant_collection", "docmind_docs")
        )
    return ServerHybridRetriever(params)


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
