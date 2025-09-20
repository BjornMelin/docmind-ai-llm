"""Contract tests requiring llama_index to be installed."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core", reason="requires llama_index.core")

from src.retrieval.llama_index_adapter import build_llama_index_factory

pytestmark = pytest.mark.requires_llama


def test_factory_exposes_expected_interfaces() -> None:
    """The real factory exposes graph artifacts and telemetry hooks."""
    factory = build_llama_index_factory()
    assert factory.supports_graphrag is True
    assert factory.get_index_builder() is not None
    telemetry = factory.get_telemetry_hooks()
    assert hasattr(telemetry, "router_built")
    assert hasattr(telemetry, "graph_exported")
