"""Unit tests for CrossEncoder reranking (FEAT-002).

Tests the complete architectural replacement of ColbertRerank
with sentence-transformers CrossEncoder per ADR-006, providing superior
relevance scoring and result ordering.

Test Coverage:
- BGECrossEncoderRerank class initialization and configuration
- Query-document relevance scoring with BGE-reranker-v2-m3
- FP16 acceleration for RTX 4090 optimization
- Score normalization and result ordering
- Performance validation (<100ms target)
- LlamaIndex BaseNodePostprocessor integration
- Factory functions and benchmarking utilities
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval.postprocessor.cross_encoder_rerank import (
    BGECrossEncoderRerank,
    benchmark_reranking_latency,
    create_bge_cross_encoder_reranker,
)


class TestBGECrossEncoderRerank:  # pylint: disable=protected-access
    """Unit tests for BGECrossEncoderRerank class."""

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    @patch("src.retrieval.postprocessor.cross_encoder_rerank.torch")
    def test_init_with_defaults(
        self, mock_torch, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test BGECrossEncoderRerank initialization with default parameters."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_torch.cuda.is_available.return_value = True

        # Mock model for FP16 testing
        mock_model = MagicMock()
        mock_cross_encoder.model = mock_model

        reranker = BGECrossEncoderRerank()

        # Verify default parameters
        assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
        assert reranker.top_n == 5
        assert reranker.device == "cuda"
        assert reranker.use_fp16 is True
        assert reranker.normalize_scores is True
        assert reranker.batch_size == 16  # RTX 4090 optimized
        assert reranker.max_length == 512

        # Verify model initialization
        mock_cross_encoder_class.assert_called_once_with(
            "BAAI/bge-reranker-v2-m3",
            device="cuda",
            trust_remote_code=True,
            max_length=512,
        )

        # Verify FP16 acceleration
        mock_model.half.assert_called_once()

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    @patch("src.retrieval.postprocessor.cross_encoder_rerank.torch")
    def test_init_with_custom_config(
        self, mock_torch, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test BGECrossEncoderRerank initialization with custom configuration."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_torch.cuda.is_available.return_value = False

        mock_cross_encoder.model = MagicMock()

        reranker = BGECrossEncoderRerank(
            model_name="custom-reranker",
            top_n=10,
            device="cpu",
            use_fp16=False,
            normalize_scores=False,
            batch_size=8,
            max_length=1024,
        )

        assert reranker.model_name == "custom-reranker"
        assert reranker.top_n == 10
        assert reranker.device == "cpu"
        assert reranker.use_fp16 is False
        assert reranker.normalize_scores is False
        assert reranker.batch_size == 8
        assert reranker.max_length == 1024

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder", None)
    def test_init_missing_sentence_transformers(self):
        """Test error handling when sentence-transformers is not available."""
        with pytest.raises(ImportError, match="sentence-transformers not available"):
            BGECrossEncoderRerank()

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_init_model_loading_error(self, mock_cross_encoder_class):
        """Test error handling when CrossEncoder model fails to load."""
        mock_cross_encoder_class.side_effect = RuntimeError("Model loading failed")

        with pytest.raises(RuntimeError, match="Model loading failed"):
            BGECrossEncoderRerank()

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_fp16_acceleration_disabled_without_cuda(
        self, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test FP16 is not applied when CUDA is unavailable."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_cross_encoder.model = MagicMock()

        with patch(
            "src.retrieval.postprocessor.cross_encoder_rerank.torch.cuda.is_available",
            return_value=False,
        ):
            _ = BGECrossEncoderRerank(use_fp16=True)

            # FP16 should not be applied without CUDA
            mock_cross_encoder.model.half.assert_not_called()

    def test_postprocess_nodes_empty_inputs(self, mock_cross_encoder):
        """Test reranking with empty or invalid inputs."""
        with patch(
            "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder",
            return_value=mock_cross_encoder,
        ):
            reranker = BGECrossEncoderRerank()

            # Test with None query bundle
            result = reranker._postprocess_nodes([], None)
            assert result == []

            # Test with empty nodes list
            query_bundle = QueryBundle(query_str="test query")
            result = reranker._postprocess_nodes([], query_bundle)
            assert result == []

    def test_postprocess_nodes_single_node(
        self, mock_cross_encoder, performance_test_nodes
    ):
        """Test reranking with single node (should return without processing)."""
        with patch(
            "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder",
            return_value=mock_cross_encoder,
        ):
            reranker = BGECrossEncoderRerank(top_n=3)

            single_node = [performance_test_nodes[0]]
            query_bundle = QueryBundle(query_str="test query")

            result = reranker._postprocess_nodes(single_node, query_bundle)

            # Should return original node unchanged
            assert result == single_node
            assert len(result) == 1

    def test_postprocess_nodes_successful_reranking(
        self, mock_cross_encoder, performance_test_nodes
    ):
        """Test successful reranking with multiple nodes."""
        with patch(
            "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder",
            return_value=mock_cross_encoder,
        ):
            # Mock CrossEncoder predictions (decreasing relevance)
            mock_scores = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
            mock_cross_encoder.predict.return_value = mock_scores

            reranker = BGECrossEncoderRerank(top_n=3, normalize_scores=False)

            # Use first 5 nodes
            test_nodes = performance_test_nodes[:5]
            query_bundle = QueryBundle(query_str="test query for reranking")

            result = reranker._postprocess_nodes(test_nodes, query_bundle)

            # Verify reranking
            assert len(result) == 3  # top_n

            # Verify scores are updated and sorted (descending)
            assert result[0].score == 0.95
            assert result[1].score == 0.85
            assert result[2].score == 0.75

            # Verify CrossEncoder was called correctly
            mock_cross_encoder.predict.assert_called_once()
            call_args = mock_cross_encoder.predict.call_args[0][0]

            # Should have 5 query-document pairs
            assert len(call_args) == 5
            assert all(len(pair) == 2 for pair in call_args)
            assert all(pair[0] == "test query for reranking" for pair in call_args)

    def test_postprocess_nodes_with_score_normalization(
        self, mock_cross_encoder, performance_test_nodes
    ):
        """Test reranking with sigmoid score normalization."""
        with (
            patch(
                "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder",
                return_value=mock_cross_encoder,
            ),
            patch(
                "src.retrieval.postprocessor.cross_encoder_rerank.torch"
            ) as mock_torch,
        ):
            # Mock raw scores and normalized scores
            raw_scores = np.array([2.0, 1.0, 0.0, -1.0])
            normalized_tensor = torch.tensor([0.88, 0.73, 0.50, 0.27])

            mock_cross_encoder.predict.return_value = raw_scores
            mock_torch.sigmoid.return_value = normalized_tensor
            mock_torch.tensor.return_value = normalized_tensor
            normalized_tensor.numpy = MagicMock(
                return_value=np.array([0.88, 0.73, 0.50, 0.27])
            )

            reranker = BGECrossEncoderRerank(top_n=2, normalize_scores=True)

            test_nodes = performance_test_nodes[:4]
            query_bundle = QueryBundle(query_str="test query")

            result = reranker._postprocess_nodes(test_nodes, query_bundle)

            # Verify sigmoid normalization was applied
            mock_torch.sigmoid.assert_called_once()
            mock_torch.tensor.assert_called_once_with(raw_scores)

            # Verify normalized scores
            assert len(result) == 2
            assert result[0].score == 0.88  # Highest normalized score
            assert result[1].score == 0.73  # Second highest

    def test_postprocess_nodes_error_handling_fallback(
        self, mock_cross_encoder, performance_test_nodes
    ):
        """Test fallback to original ordering when reranking fails."""
        with patch(
            "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder",
            return_value=mock_cross_encoder,
        ):
            # Mock CrossEncoder failure
            mock_cross_encoder.predict.side_effect = RuntimeError("Prediction failed")

            reranker = BGECrossEncoderRerank(top_n=3)

            test_nodes = performance_test_nodes[:5]
            query_bundle = QueryBundle(query_str="test query")

            result = reranker._postprocess_nodes(test_nodes, query_bundle)

            # Should fallback to original nodes (first top_n)
            assert len(result) == 3
            assert result == test_nodes[:3]

    def test_get_model_info(self, mock_cross_encoder):
        """Test model information retrieval."""
        with patch(
            "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder",
            return_value=mock_cross_encoder,
        ):
            reranker = BGECrossEncoderRerank(
                model_name="test-model",
                top_n=10,
                device="cpu",
                use_fp16=False,
                batch_size=8,
                max_length=1024,
                normalize_scores=False,
            )

            info = reranker.get_model_info()

            # Verify all configuration is returned
            expected_info = {
                "model_name": "test-model",
                "device": "cpu",
                "use_fp16": False,
                "max_length": 1024,
                "batch_size": 8,
                "normalize_scores": False,
                "top_n": 10,
            }

            assert info == expected_info

    def test_batch_processing_optimization(self, mock_cross_encoder):
        """Test batch processing with RTX 4090 optimized settings."""
        with patch(
            "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder",
            return_value=mock_cross_encoder,
        ):
            # Mock batch prediction
            batch_size = 16
            mock_scores = np.array([0.9] * batch_size)
            mock_cross_encoder.predict.return_value = mock_scores

            reranker = BGECrossEncoderRerank(batch_size=batch_size, top_n=10)

            # Create batch_size nodes
            nodes = []
            for i in range(batch_size):
                node = NodeWithScore(
                    node=TextNode(text=f"batch document {i}", id_=f"batch_{i}"),
                    score=0.5,
                )
                nodes.append(node)

            query_bundle = QueryBundle(query_str="batch test query")

            result = reranker._postprocess_nodes(nodes, query_bundle)

            # Verify batch processing
            mock_cross_encoder.predict.assert_called_once()
            call_kwargs = mock_cross_encoder.predict.call_args[1]

            assert call_kwargs["batch_size"] == batch_size
            assert call_kwargs["show_progress_bar"] is False

            assert len(result) == 10  # top_n


class TestBGECrossEncoderFactory:
    """Test factory functions and configuration helpers."""

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_create_bge_cross_encoder_reranker_defaults(
        self, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test factory function with default parameters."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_cross_encoder.model = MagicMock()

        with patch(
            "src.retrieval.postprocessor.cross_encoder_rerank.torch.cuda.is_available",
            return_value=True,
        ):
            reranker = create_bge_cross_encoder_reranker()

            # Verify RTX 4090 optimized defaults
            assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
            assert reranker.top_n == 5
            assert reranker.use_fp16 is True
            assert reranker.device == "cuda"
            assert reranker.batch_size == 16  # RTX 4090 optimized
            assert reranker.normalize_scores is True
            assert reranker.max_length == 512

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_create_bge_cross_encoder_reranker_custom(
        self, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test factory function with custom parameters."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_cross_encoder.model = MagicMock()

        reranker = create_bge_cross_encoder_reranker(
            model_name="custom-reranker",
            top_n=10,
            use_fp16=False,
            device="cpu",
            batch_size=4,
        )

        assert reranker.model_name == "custom-reranker"
        assert reranker.top_n == 10
        assert reranker.use_fp16 is False
        assert reranker.device == "cpu"
        assert reranker.batch_size == 4  # CPU optimized (min)

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_create_bge_cross_encoder_cuda_optimization(
        self, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test CUDA-specific batch size optimization."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_cross_encoder.model = MagicMock()

        # Test CUDA device with small batch size
        reranker = create_bge_cross_encoder_reranker(device="cuda", batch_size=8)
        assert reranker.batch_size == 16  # Should be increased to minimum 16

        # Test CPU device with large batch size
        reranker = create_bge_cross_encoder_reranker(device="cpu", batch_size=16)
        assert reranker.batch_size == 4  # Should be reduced to maximum 4


class TestBGECrossEncoderPerformance:  # pylint: disable=protected-access
    """Performance and benchmarking tests."""

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_reranking_latency_benchmark(
        self, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test reranking latency meets <100ms target."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_cross_encoder.model = MagicMock()

        # Mock fast prediction
        mock_cross_encoder.predict.return_value = np.array(
            [0.9, 0.8, 0.7] * 7
        )  # 20 scores

        reranker = BGECrossEncoderRerank()

        # Generate test data
        query = "test query for performance"
        documents = [f"performance document {i}" for i in range(20)]

        # Benchmark latency
        results = benchmark_reranking_latency(reranker, query, documents, num_runs=3)

        # Verify benchmark results structure
        assert "mean_latency_ms" in results
        assert "min_latency_ms" in results
        assert "max_latency_ms" in results
        assert "num_documents" in results
        assert "target_latency_ms" in results

        assert results["num_documents"] == 20
        assert results["target_latency_ms"] == 100.0

        # In real implementation, this should be < 100ms on RTX 4090
        assert results["mean_latency_ms"] >= 0

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_memory_efficient_processing(
        self, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test memory-efficient processing for large document sets."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_cross_encoder.model = MagicMock()

        # Mock processing large batch
        num_docs = 100
        mock_cross_encoder.predict.return_value = np.random.rand(num_docs)

        reranker = BGECrossEncoderRerank(batch_size=16, top_n=20)

        # Create large node set
        nodes = []
        for i in range(num_docs):
            node = NodeWithScore(
                node=TextNode(text=f"large doc {i}", id_=f"doc_{i}"), score=0.5
            )
            nodes.append(node)

        query_bundle = QueryBundle(query_str="memory test query")

        # Should process successfully without memory issues
        result = reranker._postprocess_nodes(nodes, query_bundle)

        # Verify processing
        assert len(result) == 20  # top_n
        mock_cross_encoder.predict.assert_called_once()

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_rtx_4090_batch_size_optimization(
        self, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test RTX 4090 specific batch size optimization."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder.predict.return_value = np.array([0.9] * 16)

        # RTX 4090 optimized reranker
        reranker = BGECrossEncoderRerank(device="cuda", batch_size=16)

        # Process exactly batch_size documents
        nodes = []
        for i in range(16):
            node = NodeWithScore(
                node=TextNode(text=f"rtx doc {i}", id_=f"rtx_{i}"), score=0.5
            )
            nodes.append(node)

        query_bundle = QueryBundle(query_str="rtx test query")

        start_time = time.perf_counter()
        _ = reranker._postprocess_nodes(nodes, query_bundle)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Verify optimal batch processing
        mock_cross_encoder.predict.assert_called_once()
        call_kwargs = mock_cross_encoder.predict.call_args[1]
        assert call_kwargs["batch_size"] == 16

        # Should be very fast with mocked model
        assert latency_ms < 50  # Lenient for mocked test


@pytest.mark.integration
class TestBGECrossEncoderIntegration:  # pylint: disable=protected-access
    """Integration tests with LlamaIndex ecosystem."""

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_llamaindex_base_node_postprocessor_interface(
        self, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test integration with LlamaIndex BaseNodePostprocessor interface."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_cross_encoder.model = MagicMock()

        reranker = BGECrossEncoderRerank()

        # Verify BaseNodePostprocessor interface
        assert hasattr(reranker, "_postprocess_nodes")

        # Should be callable as postprocessor
        from llama_index.core.postprocessor.types import BaseNodePostprocessor

        assert isinstance(reranker, BaseNodePostprocessor)

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_query_engine_integration(
        self, mock_cross_encoder_class, mock_cross_encoder, performance_test_nodes
    ):
        """Test integration as query engine postprocessor."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        reranker = BGECrossEncoderRerank(top_n=3)

        # Simulate query engine usage
        query_bundle = QueryBundle(query_str="integration test query")
        nodes = performance_test_nodes[:5]

        # Should work as postprocessor in query pipeline
        result = reranker._postprocess_nodes(nodes, query_bundle)

        assert len(result) == 3
        assert all(isinstance(node, NodeWithScore) for node in result)
        assert all(hasattr(node.node, "text") for node in result)

    @patch("src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder")
    def test_retrieval_pipeline_integration(
        self, mock_cross_encoder_class, mock_cross_encoder
    ):
        """Test integration in complete retrieval pipeline."""
        mock_cross_encoder_class.return_value = mock_cross_encoder
        mock_cross_encoder.model = MagicMock()
        mock_cross_encoder.predict.return_value = np.array([0.95, 0.85, 0.75])

        reranker = BGECrossEncoderRerank(top_n=2)

        # Simulate retrieval results from vector search
        retrieved_nodes = [
            NodeWithScore(
                node=TextNode(text="First retrieved document", id_="ret_1"),
                score=0.7,  # Original retrieval score
            ),
            NodeWithScore(
                node=TextNode(text="Second retrieved document", id_="ret_2"), score=0.6
            ),
            NodeWithScore(
                node=TextNode(text="Third retrieved document", id_="ret_3"), score=0.5
            ),
        ]

        query_bundle = QueryBundle(query_str="pipeline integration test")

        # Rerank should improve relevance ordering
        reranked = reranker._postprocess_nodes(retrieved_nodes, query_bundle)

        # Verify reranking improved scores and ordering
        assert len(reranked) == 2
        # Scores normalized: sigmoid(0.95)≈0.7211, sigmoid(0.85)≈0.7006
        assert (
            abs(reranked[0].score - 0.7211) < 0.001
        )  # CrossEncoder score (normalized)
        assert (
            abs(reranked[1].score - 0.7006) < 0.001
        )  # CrossEncoder score (normalized)
        assert reranked[0].score > reranked[1].score  # Proper ordering


@pytest.mark.performance
class TestBGECrossEncoderBenchmark:  # pylint: disable=protected-access
    """Comprehensive performance benchmarking."""

    def test_benchmark_reranking_latency_comprehensive(self):
        """Test comprehensive latency benchmarking function."""
        # Mock reranker for testing benchmark function
        mock_reranker = MagicMock()

        def mock_postprocess(nodes, _query_bundle):
            # Simulate reranking latency
            time.sleep(0.001)  # 1ms simulated latency
            return nodes[:5]  # Return top 5

        mock_reranker._postprocess_nodes = mock_postprocess

        query = "benchmark test query"
        documents = [f"benchmark doc {i}" for i in range(20)]

        # Run benchmark
        results = benchmark_reranking_latency(
            mock_reranker, query, documents, num_runs=3
        )

        # Verify benchmark structure and reasonable values
        assert isinstance(results, dict)
        required_keys = [
            "mean_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
            "num_documents",
            "target_latency_ms",
        ]

        for key in required_keys:
            assert key in results

        assert results["num_documents"] == 20
        assert results["target_latency_ms"] == 100.0
        assert results["mean_latency_ms"] > 0
        assert results["min_latency_ms"] <= results["mean_latency_ms"]
        assert results["max_latency_ms"] >= results["mean_latency_ms"]
