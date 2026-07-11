"""Contract tests requiring llama_index to be installed."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core", reason="requires llama_index.core")

from src.retrieval.llama_index_adapter import (
    get_graphrag_health,
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


def test_required_core_exposes_property_graph_index() -> None:
    """The required LlamaIndex core install exposes GraphRAG APIs."""
    supported, name, _hint = get_graphrag_health(force_refresh=True)
    assert supported is True
    assert name == "llama_index"
