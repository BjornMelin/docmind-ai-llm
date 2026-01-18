"""Unit tests for lightweight service factories in containers module."""

from __future__ import annotations

import types

import pytest

import src.containers as containers


@pytest.mark.unit
def test_get_embedding_model_returns_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get embedding model returns mapping."""
    # Provide a minimal get_embedding_config on settings
    monkeypatch.setattr(
        containers,
        "settings",
        types.SimpleNamespace(get_embedding_config=lambda: {"model": "bge-m3"}),
        raising=True,
    )
    cfg = containers.get_embedding_model()
    assert isinstance(cfg, dict)
    assert cfg.get("model") == "bge-m3"


@pytest.mark.unit
def test_get_multi_agent_coordinator_constructs_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test get multi agent coordinator constructs instance."""
    # Provide a minimal model configuration
    monkeypatch.setattr(
        containers,
        "settings",
        types.SimpleNamespace(
            get_model_config=lambda: {
                "model_name": "Qwen/Qwen2",
                "context_window": 1024,
            }
        ),
        raising=True,
    )
    coord = containers.get_multi_agent_coordinator(enable_fallback=False)

    assert isinstance(coord, containers.MultiAgentCoordinator)
    assert coord.enable_fallback is False
