"""Integration tests for deterministic coordinator response extraction.

These tests validate real coordinator initialization and response metrics without
external model or retrieval dependencies.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.coordinator import MultiAgentCoordinator


@pytest.mark.integration
class TestAgentCoordinatorIntegration:
    """Coordinator integration with deterministic terminal state."""

    def test_initialization_and_basic_query(self):
        """Coordinator initializes and extracts a deterministic response."""
        coord = MultiAgentCoordinator()
        final_state = {
            "messages": [SimpleNamespace(content="Shim: processed successfully")],
            "validation_result": {"confidence": 0.9},
        }

        resp = coord._extract_response(final_state, "hello world", 0.0, 0.01)

        assert isinstance(resp.content, str)
        assert resp.content

    def test_metadata_and_metrics_present(self):
        """Response includes metadata and populated optimization metrics."""
        coord = MultiAgentCoordinator()
        final_state = {
            "messages": [SimpleNamespace(content="Shim result")],
            "validation_result": {"confidence": 0.75},
            "agent_timings": {"retrieval_agent": 0.01},
        }

        resp = coord._extract_response(final_state, "test routing", 0.0, 0.02)

        assert isinstance(resp.metadata, dict)
        assert "coordination_overhead_ms" in resp.optimization_metrics
