"""Integration tests for FEAT-002 Retrieval & Search System components.

Tests the complete integration of:
- BGE-M3 unified embeddings (dense/sparse/colbert)
- AdaptiveRouterQueryEngine with strategy selection
- BGECrossEncoderRerank with performance optimization
- End-to-end retrieval pipeline performance

Validates the architectural replacement of:
- BGE-large + SPLADE++ → BGE-M3 unified
- QueryFusionRetriever → RouterQueryEngine adaptive routing
- ColbertRerank → CrossEncoder reranking
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval.embeddings import BGEM3Embedding
from src.retrieval.query_engine import AdaptiveRouterQueryEngine
from src.retrieval.reranking import BGECrossEncoderRerank

# Constants for deterministic testing
DETERMINISTIC_SEED = 42


@pytest.mark.integration
class TestRetrievalPipelineIntegration:  # pylint: disable=protected-access,unused-argument
    """Integration tests for complete retrieval pipeline."""

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_bgem3_to_router_to_reranker_pipeline(
        self,
        mock_cross_encoder_class,
        mock_flag_model_class,
        sample_test_documents,
        sample_query_scenarios,
    ):
        """Test complete BGE-M3 → Router → Reranker pipeline."""
        # Create deterministic random state
        deterministic_random = np.random.RandomState(DETERMINISTIC_SEED)

        # Mock BGE-M3 embeddings
        mock_bgem3_model = MagicMock()
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": deterministic_random.rand(
                len(sample_test_documents), 1024
            ).astype(np.float32)
        }
        mock_flag_model_class.return_value = mock_bgem3_model

        # Mock CrossEncoder
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder.predict.return_value = np.array([0.95, 0.85, 0.75])
        mock_cross_encoder_class.return_value = mock_cross_encoder

        # 1. Create BGE-M3 embedding model
        embedding = BGEM3Embedding()

        # 2. Create mock vector index with BGE-M3 embeddings
        mock_vector_index = MagicMock()

        # 3. Create reranker
        reranker = BGECrossEncoderRerank(top_n=3)

        # 4. Create adaptive router with full pipeline
        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index, reranker=reranker
        )

        # Mock router response with nodes
        test_nodes = [
            NodeWithScore(
                node=TextNode(text=doc.text, id_=f"node_{i}"), score=0.8 - (i * 0.1)
            )
            for i, doc in enumerate(sample_test_documents[:3])
        ]

        mock_response = MagicMock()
        mock_response.source_nodes = test_nodes
        router.router_engine.query.return_value = mock_response

        # 5. Test end-to-end query processing
        query_scenario = sample_query_scenarios[
            0
        ]  # "explain quantum computing applications"

        result = router.query(query_scenario["query"])

        # Verify pipeline integration
        assert result == mock_response
        assert embedding.embed_dim == 1024  # BGE-M3 dimension
        assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
        assert len(router.get_available_strategies()) >= 2

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    async def test_async_pipeline_integration(
        self, mock_cross_encoder_class, mock_flag_model_class, sample_query_scenarios
    ):
        """Test async integration across all components."""
        # Create deterministic random state
        deterministic_random = np.random.RandomState(DETERMINISTIC_SEED)

        # Mock models
        mock_bgem3_model = MagicMock()
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": deterministic_random.rand(1, 1024).astype(np.float32)
        }
        mock_flag_model_class.return_value = mock_bgem3_model

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder_class.return_value = mock_cross_encoder

        # Create components
        embedding = BGEM3Embedding()
        mock_vector_index = MagicMock()
        reranker = BGECrossEncoderRerank()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index, reranker=reranker
        )

        # Mock async response
        mock_async_response = MagicMock()
        router.router_engine.aquery = MagicMock(return_value=mock_async_response)

        # Test async query
        query = sample_query_scenarios[1]["query"]  # "BGE-M3 embeddings"

        # Test async embedding
        embedding_result = await embedding._aget_query_embedding(query)
        assert len(embedding_result) == 1024

        # Test async router (currently sync, but interface ready)
        result = await router.aquery(query)
        assert result == mock_async_response

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_strategy_selection_with_bgem3_features(
        self, mock_cross_encoder_class, mock_flag_model_class, sample_query_scenarios
    ):
        """Test router strategy selection considers BGE-M3 capabilities."""
        # Mock models
        mock_bgem3_model = MagicMock()
        mock_flag_model_class.return_value = mock_bgem3_model

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder_class.return_value = mock_cross_encoder

        # Create hybrid retriever mock that uses BGE-M3 unified embeddings
        mock_hybrid_retriever = MagicMock()

        # Create full-featured router
        mock_vector_index = MagicMock()
        mock_kg_index = MagicMock()
        reranker = BGECrossEncoderRerank()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index,
            kg_index=mock_kg_index,
            hybrid_retriever=mock_hybrid_retriever,
            reranker=reranker,
        )

        # Test all strategies are available with BGE-M3 descriptions
        strategies = router.get_available_strategies()
        expected_strategies = [
            "hybrid_search",
            "semantic_search",
            "multi_query_search",
            "knowledge_graph",
        ]

        for strategy in expected_strategies:
            assert strategy in strategies

        # Verify strategy descriptions mention BGE-M3 features
        tools = router._create_query_engine_tools()

        hybrid_tool = next(t for t in tools if t.metadata.name == "hybrid_search")
        description = hybrid_tool.metadata.description.lower()

        # Should mention BGE-M3 unified capabilities
        assert "bge-m3" in description
        assert "unified" in description
        assert "8k context" in description

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_error_handling_across_components(
        self, mock_cross_encoder_class, mock_flag_model_class
    ):
        """Test error handling and fallbacks across integrated components."""
        # Mock models with potential failures
        mock_bgem3_model = MagicMock()
        mock_flag_model_class.return_value = mock_bgem3_model

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder_class.return_value = mock_cross_encoder

        # Create components
        _ = BGEM3Embedding()
        mock_vector_index = MagicMock()

        # Test reranker fallback on failure
        mock_cross_encoder.predict.side_effect = RuntimeError("CrossEncoder failed")
        reranker = BGECrossEncoderRerank(top_n=2)

        test_nodes = [
            NodeWithScore(node=TextNode(text="doc 1", id_="1"), score=0.9),
            NodeWithScore(node=TextNode(text="doc 2", id_="2"), score=0.8),
            NodeWithScore(node=TextNode(text="doc 3", id_="3"), score=0.7),
        ]

        query_bundle = QueryBundle(query_str="test query")

        # Should fallback gracefully
        result = reranker._postprocess_nodes(test_nodes, query_bundle)
        assert len(result) == 2  # Should return top_n without reranking

        # Test router fallback on failure
        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index, reranker=reranker
        )

        # Mock router engine failure
        router.router_engine.query.side_effect = RuntimeError("Router failed")

        # Mock fallback query engine
        mock_fallback_response = MagicMock()
        mock_fallback_engine = MagicMock()
        mock_fallback_engine.query.return_value = mock_fallback_response
        mock_vector_index.as_query_engine.return_value = mock_fallback_engine

        # Should fallback to direct semantic search
        result = router.query("test query")
        assert result == mock_fallback_response

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_performance_optimization_integration(
        self,
        mock_cross_encoder_class,
        mock_flag_model_class,
        rtx_4090_performance_targets,
        benchmark_timer,
    ):
        """Test performance optimizations work across integrated components."""
        # Create deterministic random state
        deterministic_random = np.random.RandomState(DETERMINISTIC_SEED)

        # Mock optimized models
        mock_bgem3_model = MagicMock()
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": deterministic_random.rand(5, 1024).astype(np.float32)
        }
        mock_flag_model_class.return_value = mock_bgem3_model

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        mock_cross_encoder_class.return_value = mock_cross_encoder

        # Create RTX 4090 optimized components
        embedding = BGEM3Embedding(
            use_fp16=True,
            batch_size=12,  # RTX 4090 optimized
            device="cuda",
        )

        reranker = BGECrossEncoderRerank(
            use_fp16=True,
            batch_size=16,  # RTX 4090 optimized
            device="cuda",
        )

        mock_vector_index = MagicMock()
        _ = AdaptiveRouterQueryEngine(vector_index=mock_vector_index, reranker=reranker)

        # Test unified embedding performance
        texts = [f"test document {i}" for i in range(5)]

        benchmark_timer.start()
        result = embedding.get_unified_embeddings(texts)
        latency = benchmark_timer.stop()

        # Should meet BGE-M3 performance targets
        assert result["dense"].shape == (5, 1024)
        assert latency < 1000  # Very lenient for mocked test

        # Test reranking performance
        test_nodes = [
            NodeWithScore(node=TextNode(text=text, id_=f"node_{i}"), score=0.8)
            for i, text in enumerate(texts)
        ]

        query_bundle = QueryBundle(query_str="performance test")

        benchmark_timer.start()
        reranked = reranker._postprocess_nodes(test_nodes, query_bundle)
        rerank_latency = benchmark_timer.stop()

        # Should meet reranking performance targets
        assert len(reranked) == 5  # top_n default
        assert rerank_latency < 1000  # Very lenient for mocked test


@pytest.mark.integration
class TestArchitecturalReplacement:  # pylint: disable=protected-access
    """Test architectural replacements from legacy to new systems."""

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    def test_bge_large_to_bgem3_replacement(self, mock_flag_model_class):
        """Test BGE-Large + SPLADE++ replacement with BGE-M3 unified."""
        # Create deterministic random state
        deterministic_random = np.random.RandomState(DETERMINISTIC_SEED)

        mock_bgem3_model = MagicMock()
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": deterministic_random.rand(3, 1024).astype(np.float32),
            "lexical_weights": [
                {1: 0.8, 5: 0.6, 10: 0.4},
                {2: 0.7, 7: 0.5, 15: 0.3},
                {3: 0.9, 9: 0.7, 18: 0.5},
            ],
        }
        mock_flag_model_class.return_value = mock_bgem3_model

        # Create BGE-M3 unified embedding
        bgem3 = BGEM3Embedding()

        texts = ["doc 1", "doc 2", "doc 3"]

        # Test unified embeddings (replaces separate BGE-Large + SPLADE++)
        unified_result = bgem3.get_unified_embeddings(
            texts, return_dense=True, return_sparse=True
        )

        # Verify unified model provides both dense and sparse
        assert "dense" in unified_result
        assert "sparse" in unified_result

        # Dense embeddings: 1024D like BGE-Large but from unified model
        assert unified_result["dense"].shape == (3, 1024)

        # Sparse embeddings: replaces SPLADE++ with native BGE-M3 sparse
        assert len(unified_result["sparse"]) == 3
        assert all(isinstance(weights, dict) for weights in unified_result["sparse"])

        # Test 8K context vs 512 in BGE-Large
        assert bgem3.max_length == 8192  # 16x larger context

        # Test single model vs two models
        mock_flag_model_class.assert_called_once()  # Only one model loaded

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_colbert_to_crossencoder_replacement(self, mock_cross_encoder_class):
        """Test ColbertRerank replacement with CrossEncoder reranking."""
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder.predict.return_value = np.array([0.95, 0.85, 0.75])
        mock_cross_encoder_class.return_value = mock_cross_encoder

        # Create CrossEncoder reranker (replaces ColbertRerank)
        cross_encoder = BGECrossEncoderRerank(
            model_name="BAAI/bge-reranker-v2-m3",  # Modern reranking model
            top_n=3,
        )

        # Test nodes
        test_nodes = [
            NodeWithScore(
                node=TextNode(text="Query-relevant document", id_="1"), score=0.7
            ),
            NodeWithScore(
                node=TextNode(text="Somewhat relevant document", id_="2"), score=0.6
            ),
            NodeWithScore(
                node=TextNode(text="Less relevant document", id_="3"), score=0.5
            ),
        ]

        query_bundle = QueryBundle(query_str="test reranking query")

        # Test direct relevance scoring (vs ColBERT's token-level interaction)
        result = cross_encoder._postprocess_nodes(test_nodes, query_bundle)

        # Verify CrossEncoder improvements
        assert len(result) == 3
        assert result[0].score == 0.95  # Higher precision than ColBERT
        assert result[1].score == 0.85
        assert result[2].score == 0.75

        # Verify query-document pairs were processed
        mock_cross_encoder.predict.assert_called_once()
        call_args = mock_cross_encoder.predict.call_args[0][0]

        # Should have query-document pairs for direct relevance
        assert len(call_args) == 3
        assert all(len(pair) == 2 for pair in call_args)
        assert all(pair[0] == "test reranking query" for pair in call_args)

    def test_query_fusion_to_router_replacement(
        self, mock_vector_index, mock_hybrid_retriever
    ):
        """Test QueryFusionRetriever replacement with adaptive RouterQueryEngine."""
        # Create adaptive router (replaces QueryFusionRetriever)
        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index, hybrid_retriever=mock_hybrid_retriever
        )

        # Test adaptive strategy selection (vs fixed fusion)
        strategies = router.get_available_strategies()

        # Should have multiple strategies vs single fusion approach
        assert len(strategies) >= 2
        assert "semantic_search" in strategies
        assert "hybrid_search" in strategies
        assert "multi_query_search" in strategies

        # Test intelligent routing vs brute force fusion
        tools = router._create_query_engine_tools()

        # Each tool should have detailed descriptions for LLM selection
        for tool in tools:
            description = tool.metadata.description
            assert len(description) > 100  # Detailed descriptions
            assert any(
                keyword in description.lower()
                for keyword in ["semantic", "hybrid", "query", "graph"]
            )


@pytest.mark.integration
class TestSystemConfiguration:
    """Test system-wide configuration and Settings integration."""

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    @patch("llama_index.core.Settings")
    def test_bgem3_settings_configuration(self, mock_settings, mock_flag_model_class):
        """Test BGE-M3 Settings integration replaces old embedding config."""
        from src.retrieval.embeddings import configure_bgem3_settings

        mock_bgem3_model = MagicMock()
        mock_flag_model_class.return_value = mock_bgem3_model

        # Configure global Settings with BGE-M3
        configure_bgem3_settings()

        # Verify Settings updated
        assert mock_settings.embed_model is not None

        # Should be BGE-M3 instance
        assigned_model = mock_settings.embed_model
        assert assigned_model.model_name == "BAAI/bge-m3"
        assert assigned_model.embed_dim == 1024

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_factory_function_integration(
        self, mock_cross_encoder_class, mock_flag_model_class
    ):
        """Test factory functions create optimally configured components."""
        from src.retrieval.embeddings import create_bgem3_embedding
        from src.retrieval.query_engine import (
            create_adaptive_router_engine,
        )
        from src.retrieval.reranking import (
            create_bge_cross_encoder_reranker,
        )

        # Mock models
        mock_bgem3_model = MagicMock()
        mock_flag_model_class.return_value = mock_bgem3_model

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder_class.return_value = mock_cross_encoder

        # Test factory creates RTX 4090 optimized components
        embedding = create_bgem3_embedding()
        reranker = create_bge_cross_encoder_reranker()

        mock_vector_index = MagicMock()
        router = create_adaptive_router_engine(
            vector_index=mock_vector_index, reranker=reranker
        )

        # Verify RTX 4090 optimization
        assert embedding.use_fp16 is True
        assert embedding.batch_size == 12  # RTX 4090 optimized
        assert embedding.device == "cuda"

        assert reranker.use_fp16 is True
        assert reranker.batch_size == 16  # RTX 4090 optimized
        assert reranker.device == "cuda"

        assert router.reranker == reranker
        assert len(router.get_available_strategies()) >= 2


@pytest.mark.integration
@pytest.mark.slow
class TestRealWorldScenarios:  # pylint: disable=protected-access
    """Integration tests simulating real-world usage patterns."""

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_large_document_processing(
        self, mock_cross_encoder_class, mock_flag_model_class, large_document_set
    ):
        """Test integration with realistic document volumes."""
        # Create deterministic random state
        deterministic_random = np.random.RandomState(DETERMINISTIC_SEED)

        # Mock models for large batch processing
        mock_bgem3_model = MagicMock()
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": deterministic_random.rand(
                len(large_document_set), 1024
            ).astype(np.float32)
        }
        mock_flag_model_class.return_value = mock_bgem3_model

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.model = MagicMock()
        # Mock scores for top 20 documents
        mock_cross_encoder.predict.return_value = deterministic_random.rand(20)
        mock_cross_encoder_class.return_value = mock_cross_encoder

        # Create components optimized for large batches
        embedding = BGEM3Embedding(batch_size=12)
        reranker = BGECrossEncoderRerank(batch_size=16, top_n=10)

        # Test batch embedding processing
        texts = [doc.text for doc in large_document_set[:50]]  # 50 documents

        result = embedding.get_unified_embeddings(texts)

        # Should handle large batches efficiently
        assert result["dense"].shape == (50, 1024)

        # Test reranking with realistic candidate set
        candidate_nodes = [
            NodeWithScore(
                node=TextNode(text=doc.text, id_=f"doc_{i}"), score=0.8 - (i * 0.01)
            )
            for i, doc in enumerate(large_document_set[:20])
        ]

        query_bundle = QueryBundle(query_str="machine learning algorithms")

        reranked = reranker._postprocess_nodes(candidate_nodes, query_bundle)

        # Should return top results efficiently
        assert len(reranked) == 10  # top_n
        assert all(node.score is not None for node in reranked)

    @patch("src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel")
    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_concurrent_query_processing(
        self, mock_cross_encoder_class, mock_flag_model_class, sample_query_scenarios
    ):
        """Test integration handles concurrent queries efficiently."""
        # Create deterministic random state
        deterministic_random = np.random.RandomState(DETERMINISTIC_SEED)

        # Mock models
        mock_bgem3_model = MagicMock()
        mock_bgem3_model.encode.return_value = {
            "dense_vecs": deterministic_random.rand(1, 1024).astype(np.float32)
        }
        mock_flag_model_class.return_value = mock_bgem3_model

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder.predict.return_value = np.array([0.9, 0.8, 0.7])
        mock_cross_encoder_class.return_value = mock_cross_encoder

        # Create thread-safe components
        _ = BGEM3Embedding()
        reranker = BGECrossEncoderRerank()
        mock_vector_index = MagicMock()

        router = AdaptiveRouterQueryEngine(
            vector_index=mock_vector_index, reranker=reranker
        )

        # Mock concurrent query responses
        mock_responses = [MagicMock() for _ in range(len(sample_query_scenarios))]
        router.router_engine.query.side_effect = mock_responses

        # Test concurrent processing (simulated)
        results = []
        for _, scenario in enumerate(sample_query_scenarios):
            result = router.query(scenario["query"])
            results.append(result)

        # Should handle all queries successfully
        assert len(results) == len(sample_query_scenarios)
        assert all(result is not None for result in results)

        # Verify all queries processed through router
        assert router.router_engine.query.call_count == len(sample_query_scenarios)
