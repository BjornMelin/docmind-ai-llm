"""Contract tests requiring llama_index to be installed."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core", reason="requires llama_index.core")

from src.retrieval.llama_index_adapter import get_graphrag_health

pytestmark = pytest.mark.requires_llama


def test_required_core_exposes_property_graph_index() -> None:
    """The required LlamaIndex core install exposes GraphRAG APIs."""
    supported, name, _hint = get_graphrag_health(force_refresh=True)
    assert supported is True
    assert name == "llama_index"
