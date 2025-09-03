"""Tests for service factories (no dependency injection).

Validates the factory functions that construct the embedding model,
document processor, and multi-agent coordinator using unified settings.
"""

from __future__ import annotations

import pytest

from src.agents.coordinator import MultiAgentCoordinator
from src.containers import (
    get_document_processor,
    get_embedding_model,
    get_multi_agent_coordinator,
)


@pytest.mark.unit
class TestServiceFactories:
    """Validate factory behavior and types."""

    def test_get_embedding_model_defaults(self):
        """Factory returns an embedding model with default settings."""
        model = get_embedding_model()
        assert model is not None

    def test_get_document_processor_default(self):
        """Factory returns a document processor using global settings."""
        proc = get_document_processor()
        assert proc is not None

    def test_get_multi_agent_coordinator_defaults(self):
        """Factory returns a properly configured coordinator instance."""
        coord = get_multi_agent_coordinator()
        assert isinstance(coord, MultiAgentCoordinator)


@pytest.mark.unit
class TestFactoryOverrides:
    """Basic properties of factory outputs and overrides."""

    def test_embedding_override_device(self):
        """Embedding factory respects the explicit device override."""
        model = get_embedding_model(device="cpu")
        assert model is not None

    def test_coordinator_override(self):
        """Coordinator factory applies model and context overrides."""
        coord = get_multi_agent_coordinator(
            model_path="custom/model", max_context_length=64000
        )
        assert coord.model_path == "custom/model"
        assert coord.max_context_length == 64000


@pytest.mark.unit
class TestCoordinatorInstances:
    """Coordinator instance semantics in factory-only design."""

    def test_multiple_instances_are_allowed(self):
        """Multiple calls create distinct coordinator instances."""
        coord1 = get_multi_agent_coordinator()
        coord2 = get_multi_agent_coordinator()
        assert isinstance(coord1, MultiAgentCoordinator)
        assert isinstance(coord2, MultiAgentCoordinator)
        assert coord1 is not coord2
