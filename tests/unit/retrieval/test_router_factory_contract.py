"""Contract tests requiring the real ``llama_index.core`` dependency."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core", reason="requires llama_index.core")

from src.retrieval.llama_index_adapter import (
    get_llama_index_adapter,
    set_llama_index_adapter,
)

pytestmark = pytest.mark.requires_llama


def test_real_adapter_surface() -> None:
    """Adapter exposes the symbols expected by the router factory."""
    set_llama_index_adapter(None)
    adapter = get_llama_index_adapter(force_reload=True)
    assert not adapter.__is_stub__
    assert adapter.supports_graphrag is True
    for attr in (
        "RouterQueryEngine",
        "RetrieverQueryEngine",
        "QueryEngineTool",
        "ToolMetadata",
        "LLMSingleSelector",
        "get_pydantic_selector",
    ):
        assert hasattr(adapter, attr), f"missing attribute {attr}"
