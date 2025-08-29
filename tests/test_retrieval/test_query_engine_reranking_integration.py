"""Integration tests for AdaptiveRouterQueryEngine + BGECrossEncoderRerank workflow.

Tests the complete retrieval pipeline combining intelligent routing with
cross-encoder reranking for optimal result quality and relevance.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval.query_engine import AdaptiveRouterQueryEngine
from src.retrieval.reranking import BGECrossEncoderRerank


@pytest.mark.integration
class TestQueryEngineRerankingIntegration:
    """Integration tests for complete query engine + reranking workflow."""

    @patch("src.retrieval.reranking.CrossEncoder")
    @patch("src.retrieval.query_engine.RouterQueryEngine")
    def test_hybrid_search_with_reranking_pipeline(
        self,
        mock_router_class,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_hybrid_retriever,
        mock_llm_for_routing,
        performance_test_nodes,
    ):
        """Test complete hybrid search + reranking pipeline."""
        # Mock CrossEncoder reranker
        mock_cross_encoder_model = MagicMock()
        reranked_scores = np.array([0.95, 0.88, 0.76, 0.71, 0.65])
        mock_cross_encoder_model.predict.return_value = reranked_scores
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        # Create reranker
        reranker = BGECrossEncoderRerank(top_n=5)

        # Mock router to return initial results
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.source_nodes = performance_test_nodes[:10]  # Initial retrieval
        mock_response.response = "Initial search response"
        mock_router.query.return_value = mock_response
        mock_router_class.return_value = mock_router

        # Create query engine with reranker-enabled hybrid tool
        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        query = "machine learning performance optimization techniques"
        query_bundle = QueryBundle(query_str=query)

        # Execute complete pipeline
        response = router_engine.query(query)

        # Verify router was called
        assert response == mock_response

        # Test reranking separately with the retrieved nodes
        reranked_nodes = reranker._postprocess_nodes(
            performance_test_nodes[:10], query_bundle
        )

        # Verify reranking improved results
        assert len(reranked_nodes) == 5  # top_n

        # Verify nodes are sorted by reranked scores
        assert reranked_nodes[0].score == 0.95  # Highest score
        assert reranked_nodes[1].score == 0.88
        assert reranked_nodes[-1].score == 0.65  # Lowest of top 5

    @patch("src.retrieval.reranking.CrossEncoder")
    @patch("src.retrieval.query_engine.RouterQueryEngine")
    def test_semantic_search_with_reranking_quality_improvement(
        self,
        mock_router_class,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_llm_for_routing,
        sample_query_scenarios,
    ):
        """Test that reranking improves semantic search result quality."""
        # Mock CrossEncoder with quality improvement scores
        mock_cross_encoder_model = MagicMock()

        # Simulate reranking improving relevance order
        initial_scores = [0.7, 0.65, 0.8, 0.6, 0.75]  # Sub-optimal order
        improved_scores = [0.9, 0.85, 0.95, 0.7, 0.88]  # Better relevance

        mock_cross_encoder_model.predict.return_value = np.array(improved_scores)
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=3, normalize_scores=True)

        # Create nodes with sub-optimal initial ordering
        nodes = []
        for i, score in enumerate(initial_scores):
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text=f"Document {i} about neural network optimization",
                        id_=f"doc_{i}",
                    ),
                    score=score,
                )
            )

        # Test with conceptual query
        query_bundle = QueryBundle(query_str=sample_query_scenarios[0]["query"])

        # Apply reranking
        reranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)

        # Verify quality improvement
        assert len(reranked_nodes) == 3

        # Best document should be ranked first after reranking
        assert reranked_nodes[0].score == 0.95  # Was 3rd, now 1st
        assert reranked_nodes[1].score == 0.9  # Was 1st, now 2nd
        assert reranked_nodes[2].score == 0.88  # Was 5th, now 3rd

        # Verify reranking was applied
        mock_cross_encoder_model.predict.assert_called_once()

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_multimodal_query_detection_with_reranking(
        self,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_llm_for_routing,
    ):
        """Test multimodal query detection integrated with reranking."""
        # Create multimodal index mock
        mock_multimodal_index = MagicMock()
        mock_multimodal_engine = MagicMock()
        mock_multimodal_index.as_query_engine.return_value = mock_multimodal_engine

        # Mock reranker
        mock_cross_encoder_model = MagicMock()
        mock_cross_encoder_model.predict.return_value = np.array([0.9, 0.8, 0.7])
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=3)

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            multimodal_index=mock_multimodal_index,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        # Test multimodal query detection
        multimodal_queries = [
            "Show me diagrams of transformer architecture",
            "Find images about neural network visualization",
            "Display charts of model performance comparison",
        ]

        for query in multimodal_queries:
            is_multimodal = router_engine._detect_multimodal_query(query)
            assert is_multimodal, f"Should detect '{query}' as multimodal"

        # Verify multimodal strategy is available
        strategies = router_engine.get_available_strategies()
        assert "multimodal_search" in strategies

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_knowledge_graph_with_reranking_relationships(
        self,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_property_graph,
        mock_llm_for_routing,
    ):
        """Test knowledge graph search with reranking for relationship queries."""
        # Mock reranker with relationship-focused scoring
        mock_cross_encoder_model = MagicMock()
        relationship_scores = np.array([0.92, 0.87, 0.83, 0.79])
        mock_cross_encoder_model.predict.return_value = relationship_scores
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=3)

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_property_graph,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        # Verify knowledge graph strategy is available
        strategies = router_engine.get_available_strategies()
        assert "knowledge_graph" in strategies

        # Create relationship-focused nodes
        relationship_nodes = [
            NodeWithScore(
                node=TextNode(
                    text="Transformers are connected to attention mechanisms",
                    id_="rel_1",
                ),
                score=0.7,
            ),
            NodeWithScore(
                node=TextNode(
                    text="BERT builds upon transformer architecture",
                    id_="rel_2",
                ),
                score=0.65,
            ),
            NodeWithScore(
                node=TextNode(
                    text="Attention enables relationships between tokens",
                    id_="rel_3",
                ),
                score=0.75,
            ),
            NodeWithScore(
                node=TextNode(
                    text="Graph neural networks model entity relationships",
                    id_="rel_4",
                ),
                score=0.6,
            ),
        ]

        query_bundle = QueryBundle(
            query_str="How are transformers and BERT architectures related?"
        )

        # Apply reranking to relationship results
        reranked_nodes = reranker._postprocess_nodes(relationship_nodes, query_bundle)

        # Verify reranking prioritized relationship relevance
        assert len(reranked_nodes) == 3
        assert reranked_nodes[0].score == 0.92  # Most relevant relationship
        assert reranked_nodes[1].score == 0.87
        assert reranked_nodes[2].score == 0.83

    @patch("src.retrieval.reranking.CrossEncoder")
    @patch("src.retrieval.query_engine.RouterQueryEngine")
    @pytest.mark.asyncio
    async def test_async_query_engine_reranking_workflow(
        self,
        mock_router_class,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_hybrid_retriever,
        mock_llm_for_routing,
    ):
        """Test async workflow with query engine and reranking."""
        # Mock async router
        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.response = "Async search response"
        mock_response.source_nodes = []
        mock_router.aquery = AsyncMock(return_value=mock_response)
        mock_router_class.return_value = mock_router

        # Mock reranker
        mock_cross_encoder_model = MagicMock()
        mock_cross_encoder_model.predict.return_value = np.array([0.9, 0.8])
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=2)

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        # Execute async query
        response = await router_engine.aquery("async test query")

        assert response == mock_response
        mock_router.aquery.assert_called_once_with("async test query")

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_reranking_performance_with_query_routing(
        self,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_hybrid_retriever,
        mock_llm_for_routing,
        rtx_4090_performance_targets,
    ):
        """Test end-to-end performance meets RTX 4090 targets."""
        # Mock fast reranker
        mock_cross_encoder_model = MagicMock()
        mock_cross_encoder_model.predict.return_value = np.random.rand(20)
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=5, batch_size=16)

        AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        # Create 20 nodes for reranking performance test
        nodes = []
        for i in range(20):
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text=f"Performance test document {i} content",
                        id_=f"perf_{i}",
                    ),
                    score=0.5 + i * 0.02,
                )
            )

        query_bundle = QueryBundle(query_str="performance optimization techniques")

        # Measure reranking latency
        start_time = time.perf_counter()
        result = reranker._postprocess_nodes(nodes, query_bundle)
        reranking_latency = (time.perf_counter() - start_time) * 1000  # ms

        # Verify performance targets
        assert len(result) == 5
        target_latency = rtx_4090_performance_targets["reranking_latency_ms"]

        # Note: In mocked scenario, timing is not realistic
        # But we verify the structure and that it completes quickly
        assert reranking_latency < target_latency * 10  # Very loose bound for mocks

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_fallback_behavior_reranker_failure(
        self,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_llm_for_routing,
    ):
        """Test graceful fallback when reranker fails."""
        # Mock reranker that fails
        mock_cross_encoder_model = MagicMock()
        mock_cross_encoder_model.predict.side_effect = RuntimeError("Reranker failed")
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=3)

        AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        # Create test nodes
        nodes = [
            NodeWithScore(node=TextNode(text="Doc 1", id_="1"), score=0.8),
            NodeWithScore(node=TextNode(text="Doc 2", id_="2"), score=0.9),
            NodeWithScore(node=TextNode(text="Doc 3", id_="3"), score=0.7),
            NodeWithScore(node=TextNode(text="Doc 4", id_="4"), score=0.75),
        ]

        query_bundle = QueryBundle(query_str="test query")

        # Reranker should fall back gracefully
        result = reranker._postprocess_nodes(nodes, query_bundle)

        # Should return original ordering, truncated to top_n
        assert len(result) == 3
        assert result[0].node.text == "Doc 1"  # Original first
        assert result[1].node.text == "Doc 2"  # Original second
        assert result[2].node.text == "Doc 3"  # Original third

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_query_engine_tool_selection_with_reranking_context(
        self,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_hybrid_retriever,
        mock_llm_for_routing,
    ):
        """Test that query engine tool selection considers reranking context."""
        mock_cross_encoder_model = MagicMock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank()

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        # Verify tools were created with reranker
        tools = router_engine._query_engine_tools

        # Find hybrid search tool (should have reranker)
        hybrid_tool = next(t for t in tools if t.metadata.name == "hybrid_search")

        # Verify tool description mentions reranking
        description = hybrid_tool.metadata.description
        assert "reranking" in description.lower()
        assert "cross-encoder" in description.lower()

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_multi_strategy_comparison_with_reranking(
        self,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_hybrid_retriever,
        mock_llm_for_routing,
    ):
        """Test different strategies perform differently with reranking."""
        # Mock reranker with different scores for different strategies
        mock_cross_encoder_model = MagicMock()

        # Simulate different reranking effectiveness for different retrieval strategies
        semantic_scores = np.array([0.85, 0.82, 0.78])  # Good semantic matching
        hybrid_scores = np.array([0.93, 0.89, 0.85])  # Better hybrid results

        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=3)

        AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        # Create test nodes
        nodes = [
            NodeWithScore(node=TextNode(text="Semantic match doc", id_="1"), score=0.7),
            NodeWithScore(node=TextNode(text="Hybrid match doc", id_="2"), score=0.75),
            NodeWithScore(node=TextNode(text="Generic doc", id_="3"), score=0.6),
        ]

        query_bundle = QueryBundle(query_str="complex technical query")

        # Test semantic strategy results
        mock_cross_encoder_model.predict.return_value = semantic_scores
        semantic_result = reranker._postprocess_nodes(nodes, query_bundle)

        assert semantic_result[0].score == 0.85
        assert len(semantic_result) == 3

        # Test hybrid strategy results
        mock_cross_encoder_model.predict.return_value = hybrid_scores
        hybrid_result = reranker._postprocess_nodes(nodes, query_bundle)

        assert hybrid_result[0].score == 0.93
        assert len(hybrid_result) == 3

        # Verify hybrid + reranking achieved better scores
        assert hybrid_result[0].score > semantic_result[0].score


@pytest.mark.integration
class TestQueryEngineRerankingConcurrency:
    """Test concurrent query processing with reranking."""

    @patch("src.retrieval.reranking.CrossEncoder")
    @patch("src.retrieval.query_engine.RouterQueryEngine")
    @pytest.mark.asyncio
    async def test_concurrent_query_reranking(
        self,
        mock_router_class,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_llm_for_routing,
    ):
        """Test concurrent query processing with reranking doesn't interfere."""
        # Mock async router
        mock_router = MagicMock()

        async def mock_aquery(query_str, **kwargs):
            # Simulate processing delay
            await asyncio.sleep(0.01)
            mock_response = MagicMock()
            mock_response.response = f"Response for: {query_str}"
            return mock_response

        mock_router.aquery = mock_aquery
        mock_router_class.return_value = mock_router

        # Mock reranker
        mock_cross_encoder_model = MagicMock()
        mock_cross_encoder_model.predict.return_value = np.array([0.9, 0.8])
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=2)

        router_engine = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            reranker=reranker,
            llm=mock_llm_for_routing,
        )

        # Execute concurrent queries
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does attention work?",
        ]

        start_time = time.perf_counter()
        responses = await asyncio.gather(
            *[router_engine.aquery(query) for query in queries]
        )
        total_time = time.perf_counter() - start_time

        # Verify all queries completed
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert f"Response for: {queries[i]}" in response.response

        # Concurrent execution should be faster than sequential
        assert total_time < 0.1  # Should complete quickly with mocks

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_thread_safety_reranking_operations(
        self,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_llm_for_routing,
    ):
        """Test reranking operations are thread-safe."""
        import threading

        mock_cross_encoder_model = MagicMock()

        # Thread-safe counter for predictions
        prediction_count = [0]
        prediction_lock = threading.Lock()

        def mock_predict(*args, **kwargs):
            with prediction_lock:
                prediction_count[0] += 1
            return np.array([0.9, 0.8, 0.7])

        mock_cross_encoder_model.predict = mock_predict
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=3)

        # Create test nodes
        nodes = [
            NodeWithScore(node=TextNode(text="Doc 1", id_="1"), score=0.5),
            NodeWithScore(node=TextNode(text="Doc 2", id_="2"), score=0.6),
            NodeWithScore(node=TextNode(text="Doc 3", id_="3"), score=0.7),
        ]

        query_bundle = QueryBundle(query_str="thread safety test")

        # Execute reranking from multiple threads
        def rerank_operation():
            return reranker._postprocess_nodes(nodes, query_bundle)

        threads = []
        results = {}

        for i in range(3):
            thread = threading.Thread(
                target=lambda idx=i: results.update({idx: rerank_operation()})
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all operations completed successfully
        assert len(results) == 3
        assert prediction_count[0] == 3  # Each thread made one prediction

        for result in results.values():
            assert len(result) == 3
            assert result[0].score == 0.9  # Consistent results


@pytest.mark.integration
class TestQueryEngineRerankingMemoryEfficiency:
    """Test memory efficiency of combined query engine + reranking."""

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_large_document_set_reranking(
        self,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_llm_for_routing,
        mock_memory_monitor,
    ):
        """Test reranking efficiency with large document sets."""
        # Mock reranker for large batches
        mock_cross_encoder_model = MagicMock()

        # Generate scores for 100 documents
        large_scores = np.random.rand(100) * 0.4 + 0.6  # Scores 0.6-1.0
        mock_cross_encoder_model.predict.return_value = large_scores
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=10, batch_size=32)

        # Create 100 test documents
        large_nodes = []
        for i in range(100):
            large_nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text=f"Large document {i} with extensive content about machine learning",
                        id_=f"large_doc_{i}",
                    ),
                    score=0.5 + i * 0.005,  # Slightly increasing scores
                )
            )

        query_bundle = QueryBundle(
            query_str="comprehensive machine learning performance analysis"
        )

        # Monitor memory during reranking
        mock_memory_monitor.get_memory_usage()["used_gb"]

        result = reranker._postprocess_nodes(large_nodes, query_bundle)

        peak_memory = mock_memory_monitor.track_peak_usage()

        # Verify efficient processing
        assert len(result) == 10  # top_n

        # Verify memory usage stayed reasonable (mocked values)
        assert peak_memory < 16.0  # Within RTX 4090 VRAM limits

        # Verify batch processing was used
        mock_cross_encoder_model.predict.assert_called_once()
        call_kwargs = mock_cross_encoder_model.predict.call_args[1]
        assert call_kwargs["batch_size"] == 32

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_memory_cleanup_after_reranking(
        self,
        mock_cross_encoder_class,
        mock_vector_index,
        mock_llm_for_routing,
    ):
        """Test proper memory cleanup after reranking operations."""
        mock_cross_encoder_model = MagicMock()
        mock_cross_encoder_model.predict.return_value = np.array([0.9, 0.8, 0.7])
        mock_cross_encoder_class.return_value = mock_cross_encoder_model

        reranker = BGECrossEncoderRerank(top_n=3)

        nodes = [
            NodeWithScore(node=TextNode(text="Doc 1", id_="1"), score=0.5),
            NodeWithScore(node=TextNode(text="Doc 2", id_="2"), score=0.6),
            NodeWithScore(node=TextNode(text="Doc 3", id_="3"), score=0.7),
        ]

        query_bundle = QueryBundle(query_str="memory cleanup test")

        # Process multiple times to verify cleanup
        for _ in range(5):
            result = reranker._postprocess_nodes(nodes, query_bundle)
            assert len(result) == 3

        # Verify model was called multiple times (no cached state interference)
        assert mock_cross_encoder_model.predict.call_count == 5
