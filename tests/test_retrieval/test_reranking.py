"""Comprehensive test suite for BGECrossEncoderRerank (REQ-0052).

Tests CrossEncoder reranking implementation with BGE-reranker-v2-m3,
FP16 acceleration, score normalization, and performance validation.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval.reranking import (
    BGECrossEncoderRerank,
    benchmark_reranking_latency,
    create_bge_cross_encoder_reranker,
)


@pytest.mark.unit
class TestBGECrossEncoderRerankUnit:
    """Unit tests for BGECrossEncoderRerank initialization and configuration."""

    @patch("src.retrieval.reranking.CrossEncoder")
    @patch("torch.cuda.is_available", return_value=True)
    def test_init_default_configuration(
        self, mock_cuda_available, mock_cross_encoder_class
    ):
        """Test BGECrossEncoderRerank initialization with default settings."""
        mock_model = MagicMock()
        mock_model.model.half = MagicMock()
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank()

        assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
        assert reranker.top_n == 5
        assert reranker.device == "cuda"
        assert reranker.use_fp16 is True
        assert reranker.normalize_scores is True
        assert reranker.batch_size == 16
        assert reranker.max_length == 512

        # Verify CrossEncoder was initialized with correct parameters
        mock_cross_encoder_class.assert_called_once_with(
            "BAAI/bge-reranker-v2-m3",
            device="cuda",
            trust_remote_code=True,
            max_length=512,
        )

        # Verify FP16 was enabled
        mock_model.model.half.assert_called_once()

    @patch("src.retrieval.reranking.CrossEncoder")
    @patch("torch.cuda.is_available", return_value=True)
    def test_init_custom_configuration(
        self, mock_cuda_available, mock_cross_encoder_class
    ):
        """Test initialization with custom parameters."""
        mock_model = MagicMock()
        mock_model.model.half = MagicMock()
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank(
            model_name="custom/reranker",
            top_n=10,
            device="cpu",
            use_fp16=False,
            normalize_scores=False,
            batch_size=8,
            max_length=256,
        )

        assert reranker.model_name == "custom/reranker"
        assert reranker.top_n == 10
        assert reranker.device == "cpu"
        assert reranker.use_fp16 is False
        assert reranker.normalize_scores is False
        assert reranker.batch_size == 8
        assert reranker.max_length == 256

        mock_cross_encoder_class.assert_called_once_with(
            "custom/reranker",
            device="cpu",
            trust_remote_code=True,
            max_length=256,
        )

        # Verify FP16 was NOT enabled
        mock_model.model.half.assert_not_called()

    @patch("src.retrieval.reranking.CrossEncoder")
    @patch("torch.cuda.is_available", return_value=False)
    def test_init_fp16_disabled_when_cuda_unavailable(
        self, mock_cuda_available, mock_cross_encoder_class
    ):
        """Test FP16 is disabled when CUDA is not available."""
        mock_model = MagicMock()
        mock_model.model.half = MagicMock()
        mock_cross_encoder_class.return_value = mock_model

        BGECrossEncoderRerank(use_fp16=True)

        # FP16 should not be enabled when CUDA is unavailable
        mock_model.model.half.assert_not_called()

    def test_init_fails_without_sentence_transformers(self):
        """Test initialization fails gracefully when sentence-transformers unavailable."""
        with patch("src.retrieval.reranking.CrossEncoder", None), pytest.raises(
            ImportError, match="sentence-transformers not available"
        ):
            BGECrossEncoderRerank()

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_init_handles_model_loading_errors(self, mock_cross_encoder_class):
        """Test initialization handles model loading errors gracefully."""
        mock_cross_encoder_class.side_effect = OSError("Model not found")

        with pytest.raises(OSError, match="Model not found"):
            BGECrossEncoderRerank()

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_get_model_info(self, mock_cross_encoder_class):
        """Test model information retrieval."""
        mock_cross_encoder_class.return_value = MagicMock()

        reranker = BGECrossEncoderRerank(
            model_name="BAAI/bge-reranker-v2-m3",
            top_n=5,
            device="cuda",
            use_fp16=True,
            batch_size=16,
            max_length=512,
        )

        info = reranker.get_model_info()

        expected_info = {
            "model_name": "BAAI/bge-reranker-v2-m3",
            "device": "cuda",
            "use_fp16": True,
            "max_length": 512,
            "batch_size": 16,
            "normalize_scores": True,
            "top_n": 5,
        }

        assert info == expected_info


@pytest.mark.unit
class TestBGECrossEncoderRerankReranking:
    """Unit tests for reranking logic and scoring."""

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_postprocess_nodes_empty_input(self, mock_cross_encoder_class):
        """Test reranking with empty nodes list."""
        mock_cross_encoder_class.return_value = MagicMock()
        reranker = BGECrossEncoderRerank()

        result = reranker._postprocess_nodes([], None)
        assert result == []

        result = reranker._postprocess_nodes(
            [NodeWithScore(node=TextNode(text="test"), score=1.0)], None
        )
        assert len(result) == 1

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_postprocess_nodes_single_node(self, mock_cross_encoder_class):
        """Test reranking with single node (no reranking needed)."""
        mock_cross_encoder_class.return_value = MagicMock()
        reranker = BGECrossEncoderRerank()

        node = NodeWithScore(node=TextNode(text="single node", id_="test_1"), score=0.8)
        query_bundle = QueryBundle(query_str="test query")

        result = reranker._postprocess_nodes([node], query_bundle)

        assert len(result) == 1
        assert result[0] == node

    @patch("src.retrieval.reranking.CrossEncoder")
    @patch("torch.sigmoid")
    def test_postprocess_nodes_successful_reranking(
        self, mock_sigmoid, mock_cross_encoder_class
    ):
        """Test successful reranking with multiple nodes."""
        # Mock CrossEncoder model
        mock_model = MagicMock()
        mock_scores = np.array([0.9, 0.7, 0.8, 0.6, 0.75])
        mock_model.predict.return_value = mock_scores
        mock_cross_encoder_class.return_value = mock_model

        # Mock sigmoid normalization
        mock_sigmoid.return_value = MagicMock()
        mock_sigmoid.return_value.numpy.return_value = mock_scores

        reranker = BGECrossEncoderRerank(top_n=3, normalize_scores=True)

        # Create test nodes
        nodes = []
        for i in range(5):
            node = NodeWithScore(
                node=TextNode(text=f"Document {i} content", id_=f"doc_{i}"),
                score=0.5 + i * 0.1,  # Initial scores: 0.5, 0.6, 0.7, 0.8, 0.9
            )
            nodes.append(node)

        query_bundle = QueryBundle(query_str="test query")

        result = reranker._postprocess_nodes(nodes, query_bundle)

        # Verify reranking behavior
        assert len(result) == 3  # top_n

        # Verify scores were updated and nodes sorted by new scores
        assert result[0].score == 0.9  # Highest score first
        assert result[1].score == 0.8  # Second highest
        assert result[2].score == 0.75  # Third highest

        # Verify model was called with correct pairs
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args
        pairs = call_args[0][0]  # First positional argument
        assert len(pairs) == 5

        # Check that pairs contain query and document text
        for i, pair in enumerate(pairs):
            assert pair[0] == "test query"
            assert f"Document {i} content" in pair[1]

        # Verify sigmoid was applied for normalization
        mock_sigmoid.assert_called_once()

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_postprocess_nodes_without_normalization(self, mock_cross_encoder_class):
        """Test reranking without score normalization."""
        # Mock CrossEncoder model
        mock_model = MagicMock()
        mock_scores = np.array([0.9, 0.7, 0.8])
        mock_model.predict.return_value = mock_scores
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank(normalize_scores=False, top_n=2)

        # Create test nodes
        nodes = [
            NodeWithScore(node=TextNode(text="Doc 1", id_="1"), score=0.5),
            NodeWithScore(node=TextNode(text="Doc 2", id_="2"), score=0.6),
            NodeWithScore(node=TextNode(text="Doc 3", id_="3"), score=0.7),
        ]

        query_bundle = QueryBundle(query_str="test query")

        with patch("torch.sigmoid") as mock_sigmoid:
            result = reranker._postprocess_nodes(nodes, query_bundle)

            # Verify sigmoid was NOT called
            mock_sigmoid.assert_not_called()

        # Verify scores are raw model scores
        assert result[0].score == 0.9  # Raw score, not normalized
        assert len(result) == 2  # top_n

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_postprocess_nodes_model_failure_fallback(self, mock_cross_encoder_class):
        """Test fallback behavior when CrossEncoder fails."""
        # Mock CrossEncoder model to fail
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA out of memory")
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank(top_n=2)

        # Create test nodes
        nodes = [
            NodeWithScore(node=TextNode(text="Doc 1", id_="1"), score=0.9),
            NodeWithScore(node=TextNode(text="Doc 2", id_="2"), score=0.7),
            NodeWithScore(node=TextNode(text="Doc 3", id_="3"), score=0.8),
        ]

        query_bundle = QueryBundle(query_str="test query")

        result = reranker._postprocess_nodes(nodes, query_bundle)

        # Should fall back to original ordering and return top_n
        assert len(result) == 2
        assert result == nodes[:2]  # First 2 nodes in original order

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_postprocess_nodes_batch_processing(self, mock_cross_encoder_class):
        """Test batch processing configuration."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8, 0.7])
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank(batch_size=32)

        nodes = [
            NodeWithScore(node=TextNode(text="Doc 1", id_="1"), score=0.5),
            NodeWithScore(node=TextNode(text="Doc 2", id_="2"), score=0.6),
        ]

        query_bundle = QueryBundle(query_str="test query")

        reranker._postprocess_nodes(nodes, query_bundle)

        # Verify batch_size was passed to predict
        mock_model.predict.assert_called_once()
        kwargs = mock_model.predict.call_args[1]
        assert kwargs["batch_size"] == 32
        assert kwargs["show_progress_bar"] is False


@pytest.mark.unit
class TestBGECrossEncoderRerankFactoryFunctions:
    """Unit tests for factory functions and utilities."""

    @patch("src.retrieval.reranking.BGECrossEncoderRerank")
    def test_create_bge_cross_encoder_reranker_default(self, mock_reranker_class):
        """Test factory function with default parameters."""
        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        reranker = create_bge_cross_encoder_reranker()

        mock_reranker_class.assert_called_once_with(
            model_name="BAAI/bge-reranker-v2-m3",
            top_n=5,
            device="cuda",
            use_fp16=True,
            normalize_scores=True,
            batch_size=16,  # RTX 4090 optimized
            max_length=512,
        )
        assert reranker == mock_reranker

    @patch("src.retrieval.reranking.BGECrossEncoderRerank")
    def test_create_bge_cross_encoder_reranker_custom(self, mock_reranker_class):
        """Test factory function with custom parameters."""
        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        create_bge_cross_encoder_reranker(
            model_name="custom/model",
            top_n=10,
            use_fp16=False,
            device="cpu",
            batch_size=4,
        )

        mock_reranker_class.assert_called_once_with(
            model_name="custom/model",
            top_n=10,
            device="cpu",
            use_fp16=False,
            normalize_scores=True,
            batch_size=4,  # CPU batch size kept small
            max_length=512,
        )

    @patch("src.retrieval.reranking.BGECrossEncoderRerank")
    def test_create_bge_cross_encoder_reranker_cuda_batch_optimization(
        self, mock_reranker_class
    ):
        """Test factory function optimizes batch size for CUDA."""
        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        # Test small batch size gets increased for CUDA
        create_bge_cross_encoder_reranker(device="cuda", batch_size=8)

        # Should increase to minimum CUDA batch size (16)
        args, kwargs = mock_reranker_class.call_args
        assert kwargs["batch_size"] == 16

    @patch("src.retrieval.reranking.BGECrossEncoderRerank")
    def test_create_bge_cross_encoder_reranker_cpu_batch_optimization(
        self, mock_reranker_class
    ):
        """Test factory function optimizes batch size for CPU."""
        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        # Test large batch size gets reduced for CPU
        create_bge_cross_encoder_reranker(device="cpu", batch_size=32)

        # Should decrease to maximum CPU batch size (4)
        args, kwargs = mock_reranker_class.call_args
        assert kwargs["batch_size"] == 4


@pytest.mark.integration
class TestBGECrossEncoderRerankIntegration:
    """Integration tests for reranking with realistic data."""

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_realistic_reranking_scenario(
        self, mock_cross_encoder_class, performance_test_nodes
    ):
        """Test reranking with realistic document nodes."""
        # Mock model with realistic scores
        mock_model = MagicMock()
        # Generate decreasing scores for realistic reranking
        mock_scores = np.linspace(0.95, 0.3, len(performance_test_nodes))
        np.random.shuffle(mock_scores)  # Randomize to simulate real reranking
        mock_model.predict.return_value = mock_scores
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank(top_n=5)

        query_bundle = QueryBundle(query_str="machine learning algorithms performance")

        result = reranker._postprocess_nodes(performance_test_nodes, query_bundle)

        # Verify results
        assert len(result) == 5

        # Verify results are sorted by score (descending)
        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

        # Verify model was called with query-document pairs
        mock_model.predict.assert_called_once()

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_reranking_with_various_query_types(
        self, mock_cross_encoder_class, sample_query_scenarios
    ):
        """Test reranking with different types of queries."""
        mock_model = MagicMock()
        mock_scores = np.array([0.9, 0.8, 0.7])  # Fixed scores for consistency
        mock_model.predict.return_value = mock_scores
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank(top_n=3)

        # Create test nodes
        nodes = [
            NodeWithScore(
                node=TextNode(text="Machine learning overview", id_="1"), score=0.5
            ),
            NodeWithScore(
                node=TextNode(text="Deep learning techniques", id_="2"), score=0.6
            ),
            NodeWithScore(
                node=TextNode(text="Neural network architectures", id_="3"), score=0.7
            ),
        ]

        # Test with different query scenarios
        for scenario in sample_query_scenarios[:3]:  # Test first 3 scenarios
            query_bundle = QueryBundle(query_str=scenario["query"])

            result = reranker._postprocess_nodes(nodes, query_bundle)

            # Verify consistent reranking behavior
            assert len(result) == 3
            assert result[0].score == 0.9  # Highest reranked score
            assert result[1].score == 0.8
            assert result[2].score == 0.7

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_reranking_performance_characteristics(self, mock_cross_encoder_class):
        """Test reranking performance and memory characteristics."""
        mock_model = MagicMock()
        # Simulate fast processing
        mock_model.predict.return_value = np.random.rand(20)
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank(top_n=5, batch_size=16)

        # Create 20 test nodes (typical reranking scenario)
        nodes = []
        for i in range(20):
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text=f"Document {i} with relevant content", id_=f"doc_{i}"
                    ),
                    score=0.5 + i * 0.02,  # Increasing scores
                )
            )

        query_bundle = QueryBundle(
            query_str="machine learning performance optimization"
        )

        start_time = time.perf_counter()
        result = reranker._postprocess_nodes(nodes, query_bundle)
        processing_time = time.perf_counter() - start_time

        # Verify results
        assert len(result) == 5  # top_n

        # Performance should be reasonable (not enforcing strict timing in unit tests)
        assert processing_time < 10.0  # Very loose constraint for mocked scenario

        # Verify batch processing was configured correctly
        mock_model.predict.assert_called_once()
        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["batch_size"] == 16


@pytest.mark.integration
class TestBenchmarkReranking:
    """Integration tests for benchmarking utilities."""

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_benchmark_reranking_latency(self, mock_cross_encoder_class):
        """Test reranking latency benchmarking function."""
        # Mock model for fast execution
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(5)
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank()

        query = "test query for benchmarking"
        documents = [
            "Document about machine learning algorithms",
            "Deep learning neural networks overview",
            "Computer vision applications study",
            "Natural language processing techniques",
            "Reinforcement learning strategies guide",
        ]

        results = benchmark_reranking_latency(
            reranker=reranker,
            query=query,
            documents=documents,
            num_runs=3,
        )

        # Verify benchmark results structure
        required_keys = {
            "mean_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
            "num_documents",
            "target_latency_ms",
        }
        assert set(results.keys()) == required_keys

        # Verify values are reasonable
        assert results["num_documents"] == 5
        assert results["target_latency_ms"] == 100.0  # RTX 4090 target
        assert results["mean_latency_ms"] >= 0
        assert results["min_latency_ms"] >= 0
        assert results["max_latency_ms"] >= results["min_latency_ms"]
        assert results["mean_latency_ms"] >= results["min_latency_ms"]
        assert results["mean_latency_ms"] <= results["max_latency_ms"]

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_benchmark_reranking_warmup(self, mock_cross_encoder_class):
        """Test that benchmarking includes warmup runs."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(10)
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank()

        documents = ["Doc " + str(i) for i in range(10)]

        # Mock the postprocess_nodes method to track calls
        with patch.object(reranker, "postprocess_nodes") as mock_postprocess:
            mock_postprocess.return_value = []  # Return empty for simplicity

            benchmark_reranking_latency(
                reranker=reranker,
                query="test",
                documents=documents,
                num_runs=2,
            )

            # Should be called 3 times: 1 warmup + 2 benchmark runs
            assert mock_postprocess.call_count == 3

            # First call should be warmup with subset of nodes
            warmup_call = mock_postprocess.call_args_list[0]
            warmup_nodes = warmup_call[0][0]  # First argument (nodes)
            assert len(warmup_nodes) == 3  # WARMUP_SIZE

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_benchmark_with_performance_targets(
        self, mock_cross_encoder_class, rtx_4090_performance_targets
    ):
        """Test benchmarking against RTX 4090 performance targets."""
        # Mock very fast model to meet targets
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(20)
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank(batch_size=16)

        documents = ["Document " + str(i) for i in range(20)]

        results = benchmark_reranking_latency(
            reranker=reranker,
            query="performance test query",
            documents=documents,
            num_runs=5,
        )

        # Compare against RTX 4090 targets
        target_latency = rtx_4090_performance_targets["reranking_latency_ms"]
        assert results["target_latency_ms"] == target_latency

        # Note: In mocked tests, we can't enforce actual performance,
        # but we verify the benchmark structure is correct
        assert isinstance(results["mean_latency_ms"], float)
        assert results["num_documents"] == 20


@pytest.mark.unit
class TestBGECrossEncoderRerankErrorHandling:
    """Unit tests for error handling and edge cases."""

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_torch_cuda_out_of_memory_fallback(self, mock_cross_encoder_class):
        """Test fallback when CUDA runs out of memory."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = torch.cuda.OutOfMemoryError(
            "CUDA out of memory"
        )
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank()

        nodes = [
            NodeWithScore(node=TextNode(text="Doc 1", id_="1"), score=0.8),
            NodeWithScore(node=TextNode(text="Doc 2", id_="2"), score=0.9),
        ]
        query_bundle = QueryBundle(query_str="test query")

        result = reranker._postprocess_nodes(nodes, query_bundle)

        # Should fall back to original ordering
        assert len(result) == 2
        assert result[0].node.text == "Doc 1"  # Original order preserved
        assert result[1].node.text == "Doc 2"

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_value_error_handling(self, mock_cross_encoder_class):
        """Test handling of ValueError during prediction."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = ValueError("Invalid input shape")
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank(top_n=1)

        nodes = [
            NodeWithScore(node=TextNode(text="Doc 1", id_="1"), score=0.5),
            NodeWithScore(node=TextNode(text="Doc 2", id_="2"), score=0.7),
        ]
        query_bundle = QueryBundle(query_str="test query")

        result = reranker._postprocess_nodes(nodes, query_bundle)

        # Should fall back to original ordering and respect top_n
        assert len(result) == 1
        assert result[0].node.text == "Doc 1"  # First node from original list

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_empty_node_content_handling(self, mock_cross_encoder_class):
        """Test handling of nodes with empty content."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.6])
        mock_cross_encoder_class.return_value = mock_model

        reranker = BGECrossEncoderRerank()

        nodes = [
            NodeWithScore(node=TextNode(text="", id_="1"), score=0.5),  # Empty content
            NodeWithScore(node=TextNode(text="Valid content", id_="2"), score=0.7),
        ]
        query_bundle = QueryBundle(query_str="test query")

        result = reranker._postprocess_nodes(nodes, query_bundle)

        # Should handle empty content gracefully
        assert len(result) == 2
        mock_model.predict.assert_called_once()

        # Verify pairs were created correctly
        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0][1] == ""  # Empty content preserved
        assert pairs[1][1] == "Valid content"
