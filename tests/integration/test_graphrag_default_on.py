"""GraphRAG default-on behavior and router registration tests."""

import pytest

from src.retrieval.query_engine import create_adaptive_router_engine


@pytest.mark.integration
@pytest.mark.integration
def test_router_registers_knowledge_graph_when_index_present(
    mock_vector_index, mock_property_graph
):
    """Router includes knowledge_graph tool when KG index is provided."""
    engine = create_adaptive_router_engine(
        vector_index=mock_vector_index,
        kg_index=mock_property_graph,
    )
    names = engine.get_available_strategies()
    assert "knowledge_graph" in names


@pytest.mark.integration
def test_router_excludes_knowledge_graph_when_index_absent(
    mock_vector_index, mock_llm_for_routing
):
    """Router excludes knowledge_graph when no KG index is provided (rollback)."""
    engine = create_adaptive_router_engine(
        vector_index=mock_vector_index,
        kg_index=None,
        llm=mock_llm_for_routing,
    )
    names = engine.get_available_strategies()
    assert "knowledge_graph" not in names
