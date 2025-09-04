"""Integration tests for AdaptiveRouterQueryEngine + ADR-037 reranking.

Validates end-to-end router usage with a simple text reranker stub and
multimodal-aware gating using MultimodalReranker.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import QueryBundle

from src.retrieval.query_engine import AdaptiveRouterQueryEngine
from src.retrieval.reranking import MultimodalReranker


class _StubTextReranker:
    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def postprocess_nodes(self, nodes, query_bundle: QueryBundle | None = None):
        sorted_nodes = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
        return sorted_nodes[: self.top_n]


@pytest.mark.integration
class TestQueryEngineRerankingIntegration:
    """Integration tests for complete query engine + reranking workflow."""

    @patch("src.retrieval.query_engine.RouterQueryEngine")
    def test_hybrid_search_with_reranking_pipeline(
        self,
        mock_router_class,
        mock_vector_index,
        mock_hybrid_retriever,
        mock_llm_for_routing,
        performance_test_nodes,
    ):
        """Validate router + stub reranker pipeline works and sorts nodes."""
        # Mock router to return initial results
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.source_nodes = performance_test_nodes[:10]
        mock_response.response = "Initial search response"
        mock_router.query.return_value = mock_response
        mock_router_class.return_value = mock_router

        reranker = _StubTextReranker(top_n=5)

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        response = router_engine.query("hybrid test")
        assert response == mock_response

        # Reranker behavior (explicit call with nodes)
        qb = QueryBundle(query_str="test")
        out = reranker.postprocess_nodes(performance_test_nodes[:10], qb)
        assert len(out) == 5
        assert all(out[i].score >= out[i + 1].score for i in range(4))

    def test_multimodal_query_detection_with_reranking(
        self,
        mock_vector_index,
        mock_llm_for_routing,
    ):
        """Router advertises multimodal tool when multimodal index is present."""
        # Minimal multimodal index stub
        mock_multimodal_index = MagicMock()
        mock_multimodal_engine = MagicMock()
        mock_multimodal_index.as_query_engine.return_value = mock_multimodal_engine

        reranker = MultimodalReranker()

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            multimodal_index=mock_multimodal_index,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        strategies = router_engine.get_available_strategies()
        assert "multimodal_search" in strategies
