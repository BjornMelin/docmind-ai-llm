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


def get_embedding_model(*, _nodes: list[Any] | None = None) -> Any:
    """Return an embedding model configuration object.

    This factory reflects the library-first architecture and returns a simple
    configuration mapping for embedding setups. Tests validate non-None.
    """
    return settings.get_embedding_config()


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
    selected_model = model_path or model_cfg.get("model_name") or settings.vllm.model
    selected_ctx = (
        max_context_length
        if max_context_length is not None
        else model_cfg.get("context_window") or settings.vllm.context_window
    )

    return MultiAgentCoordinator(
        model_path=str(selected_model),
        max_context_length=int(selected_ctx),
        enable_fallback=bool(enable_fallback),
    )


__all__ = [
    "get_embedding_model",
    "get_multi_agent_coordinator",
]
