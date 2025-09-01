"""Additional unit tests to raise coverage for AdaptiveRouterQueryEngine."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.retrieval.query_engine import AdaptiveRouterQueryEngine


@pytest.mark.unit
def test_detect_multimodal_query_branches(mock_vector_index, mock_llm_for_routing):
    engine = AdaptiveRouterQueryEngine(vector_index=mock_vector_index, llm=mock_llm_for_routing)

    true_cases = [
        "Show me diagrams of the pipeline",
        "Find images of transformers",
        "Picture of a GPU",
        "chart displaying results",
        "graph of accuracy",
        "figure 1",
        "screenshot of UI",
        "visual representation of data",
        "file:/tmp/image.png",
        "photo.jpg",
    ]
    false_cases = [
        "Explain transformers architecture",
        "What is semantic search?",
    ]

    for q in true_cases:
        assert engine._detect_multimodal_query(q) is True
    for q in false_cases:
        assert engine._detect_multimodal_query(q) is False


@pytest.mark.unit
def test_create_router_engine_raises_when_no_tools(mock_llm_for_routing):
    # Build engine, then delete tools and ensure the guard trips
    vec = MagicMock()
    vec.as_query_engine.return_value = MagicMock()
    engine = AdaptiveRouterQueryEngine(vector_index=vec, llm=mock_llm_for_routing)
    engine._query_engine_tools = []
    with pytest.raises(ValueError, match="No query engine tools available"):
        engine._create_router_engine()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aquery_fallback_on_router_failure(mock_vector_index, mock_llm_for_routing):
    engine = AdaptiveRouterQueryEngine(vector_index=mock_vector_index, llm=mock_llm_for_routing)
    engine.router_engine.aquery = AsyncMock(side_effect=RuntimeError("boom"))

    fallback = MagicMock()
    fallback.aquery = AsyncMock(return_value="ok")
    engine.vector_index.as_query_engine.return_value = fallback

    result = await engine.aquery("q")
    assert result == "ok"


@pytest.mark.unit
def test_get_available_strategies_names(mock_vector_index, mock_llm_for_routing):
    engine = AdaptiveRouterQueryEngine(vector_index=mock_vector_index, llm=mock_llm_for_routing)
    names = engine.get_available_strategies()
    # At minimum semantic and multi_query are present
    assert "semantic_search" in names
    assert "multi_query_search" in names

