"""Contract tests requiring llama_index to be installed."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core", reason="requires llama_index.core")

from src.retrieval.llama_index_adapter import (
    build_llama_index_factory,
    get_llama_index_adapter,
    set_llama_index_adapter,
)

pytestmark = pytest.mark.requires_llama


def test_real_router_adapter_surface() -> None:
    """Adapter exposes the symbols expected by the router factory."""
    set_llama_index_adapter(None)
    adapter = get_llama_index_adapter(force_reload=True)
    assert not adapter.__is_stub__
    for attr in (
        "RouterQueryEngine",
        "RetrieverQueryEngine",
        "QueryEngineTool",
        "ToolMetadata",
        "LLMSingleSelector",
        "get_pydantic_selector",
    ):
        assert hasattr(adapter, attr), f"missing attribute {attr}"


def test_graph_factory_exposes_expected_interfaces() -> None:
    """The real GraphRAG factory exposes graph artifacts and telemetry hooks."""
    factory = build_llama_index_factory()
    assert factory.supports_graphrag is True
    assert factory.get_index_builder() is not None
    telemetry = factory.get_telemetry_hooks()
    assert hasattr(telemetry, "router_built")
    assert callable(telemetry.router_built)
    assert hasattr(telemetry, "graph_exported")
    assert callable(telemetry.graph_exported)
