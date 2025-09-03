"""Unit tests for the BGECrossEncoderRerank module.

Key areas:
- Initialization with various configurations
- Model loading and error handling
- Node reranking with query-document pairs
- Score normalization and FP16 handling
- Fallback behavior and benchmarking utilities
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


@pytest.fixture
def mock_sentence_transformers():
    """Mock sentence_transformers CrossEncoder for testing."""
    with patch("src.retrieval.reranking.CrossEncoder") as mock_class:
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.95, 0.87, 0.72, 0.65, 0.52])
        mock_model.model.half.return_value = None
        mock_class.return_value = mock_model
        yield mock_class, mock_model


@pytest.fixture
def sample_nodes_with_scores():
    """Sample NodeWithScore objects for testing."""
    nodes = []
    texts = [
        "Machine learning transforms document processing with advanced algorithms",
        "Vector databases enable efficient semantic similarity search capabilities",
        "BGE-M3 provides unified dense and sparse embeddings for retrieval",
        "Cross-encoder reranking improves relevance scoring significantly",
        "FP16 optimization reduces VRAM usage on RTX 4090 GPUs",
    ]

    for i, text in enumerate(texts):
        node = TextNode(text=text, id_=f"node_{i}")
        node_with_score = NodeWithScore(node=node, score=0.8 - i * 0.1)
        nodes.append(node_with_score)

    return nodes


@pytest.fixture
def sample_query_bundle():
    """Sample QueryBundle for testing."""
    return QueryBundle(query_str="machine learning document processing")


@pytest.mark.unit
class TestBGECrossEncoderRerankInitialization:
    """Test BGE CrossEncoder reranker initialization."""

    @patch("src.retrieval.reranking.torch.cuda.is_available", return_value=True)
    def test_init_with_defaults(self, mock_cuda, mock_sentence_transformers):
        """Test initialization with default parameters."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers

        reranker = BGECrossEncoderRerank()

        # Verify default configuration
        assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
        assert reranker.top_n == 5
        assert reranker.device == "cuda"
        assert reranker.use_fp16 is True
        assert reranker.normalize_scores is True
        assert reranker.batch_size == 16
        assert reranker.max_length == 512

        # Verify model initialization
        mock_class.assert_called_once_with(
            "BAAI/bge-reranker-v2-m3",
            device="cuda",
            trust_remote_code=True,
            max_length=512,
        )

        # Verify FP16 was enabled
        mock_model.model.half.assert_called_once()

    @patch("src.retrieval.reranking.torch.cuda.is_available", return_value=False)
    def test_init_cpu_mode(self, mock_cuda, mock_sentence_transformers):
        """Test initialization with CPU mode."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers

        reranker = BGECrossEncoderRerank(device="cpu", use_fp16=True)

        assert reranker.device == "cpu"

        # FP16 should not be enabled on CPU
        mock_model.model.half.assert_not_called()

    def test_init_custom_parameters(self, mock_sentence_transformers):
        """Test initialization with custom parameters."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers

        reranker = BGECrossEncoderRerank(
            model_name="custom/reranker-model",
            top_n=10,
            device="cpu",
            use_fp16=False,
            normalize_scores=False,
            batch_size=8,
            max_length=256,
        )

        assert reranker.model_name == "custom/reranker-model"
        assert reranker.top_n == 10
        assert reranker.device == "cpu"
        assert reranker.use_fp16 is False
        assert reranker.normalize_scores is False
        assert reranker.batch_size == 8
        assert reranker.max_length == 256

        mock_class.assert_called_once_with(
            "custom/reranker-model",
            device="cpu",
            trust_remote_code=True,
            max_length=256,
        )

    def test_init_missing_sentence_transformers(self):
        """Test initialization fails when sentence-transformers unavailable."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        with (
            patch("src.retrieval.reranking.CrossEncoder", None),
            pytest.raises(ImportError, match="sentence-transformers not available"),
        ):
            BGECrossEncoderRerank()

    def test_init_model_loading_error(self, mock_sentence_transformers):
        """Test initialization handles model loading errors."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        mock_class.side_effect = OSError("Model loading failed")

        with pytest.raises(OSError, match="Model loading failed"):
            BGECrossEncoderRerank()

    @patch("src.retrieval.reranking.torch.cuda.is_available", return_value=True)
    def test_init_fp16_error_handling(self, mock_cuda, mock_sentence_transformers):
        """Test FP16 initialization error handling."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        mock_model.model.half.side_effect = RuntimeError("FP16 not supported")

        with pytest.raises(RuntimeError):
            BGECrossEncoderRerank(use_fp16=True)


@pytest.mark.unit
class TestBGECrossEncoderRerankPostprocessing:
    """Test the core reranking postprocessing functionality."""

    def test_postprocess_nodes_success(
        self, mock_sentence_transformers, sample_nodes_with_scores, sample_query_bundle
    ):
        """Test successful node reranking."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        mock_model.predict.return_value = np.array([0.95, 0.87, 0.72, 0.65, 0.52])

        reranker = BGECrossEncoderRerank(top_n=3, normalize_scores=False)

        result = reranker.postprocess_nodes(
            sample_nodes_with_scores, sample_query_bundle
        )

        # Should return top 3 nodes
        assert len(result) == 3

        # Should be sorted by reranker scores (descending)
        assert result[0].score == 0.95
        assert result[1].score == 0.87
        assert result[2].score == 0.72

        # Verify CrossEncoder was called with query-document pairs
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args[0][0]  # First positional argument
        assert len(call_args) == 5  # 5 query-document pairs

        # Verify each pair contains query and document text
        for pair in call_args:
            assert len(pair) == 2
            assert pair[0] == "machine learning document processing"  # Query
            assert isinstance(pair[1], str)  # Document text

    def test_postprocess_nodes_with_normalization(
        self, mock_sentence_transformers, sample_nodes_with_scores, sample_query_bundle
    ):
        """Test node reranking with score normalization."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        raw_scores = np.array([2.0, 1.0, 0.0, -1.0, -2.0])  # Raw logits
        mock_model.predict.return_value = raw_scores

        with patch("torch.sigmoid") as mock_sigmoid:
            mock_sigmoid.return_value.numpy.return_value = np.array(
                [0.88, 0.73, 0.50, 0.27, 0.12]
            )

            reranker = BGECrossEncoderRerank(normalize_scores=True, top_n=3)
            result = reranker.postprocess_nodes(
                sample_nodes_with_scores, sample_query_bundle
            )

        # Verify sigmoid normalization was applied
        mock_sigmoid.assert_called_once()

        # Verify normalized scores
        assert len(result) == 3
        assert result[0].score == 0.88
        assert result[1].score == 0.73
        assert result[2].score == 0.50

    def test_postprocess_nodes_empty_query_bundle(
        self, mock_sentence_transformers, sample_nodes_with_scores
    ):
        """Test postprocessing with empty query bundle."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        reranker = BGECrossEncoderRerank()

        result = reranker.postprocess_nodes(sample_nodes_with_scores, None)

        # Should return original nodes unchanged
        assert result == sample_nodes_with_scores
        mock_model.predict.assert_not_called()

    def test_postprocess_nodes_empty_nodes_list(
        self, mock_sentence_transformers, sample_query_bundle
    ):
        """Test postprocessing with empty nodes list."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        reranker = BGECrossEncoderRerank()

        result = reranker.postprocess_nodes([], sample_query_bundle)

        assert result == []
        mock_model.predict.assert_not_called()

    def test_postprocess_nodes_single_node(
        self, mock_sentence_transformers, sample_query_bundle
    ):
        """Test postprocessing with single node (below threshold)."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        reranker = BGECrossEncoderRerank(top_n=5)

        # single_node = [sample_query_bundle]  # Unused; keep example for clarity
        single_node_with_score = [
            NodeWithScore(
                node=TextNode(text="Single test document", id_="single"), score=0.8
            )
        ]

        result = reranker.postprocess_nodes(single_node_with_score, sample_query_bundle)

        # Should return single node without reranking
        assert len(result) == 1
        assert result[0].score == 0.8
        mock_model.predict.assert_not_called()

    def test_postprocess_nodes_error_handling(
        self, mock_sentence_transformers, sample_nodes_with_scores, sample_query_bundle
    ):
        """Test error handling during reranking."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        mock_model.predict.side_effect = RuntimeError("CrossEncoder prediction failed")

        reranker = BGECrossEncoderRerank(top_n=3)
        result = reranker.postprocess_nodes(
            sample_nodes_with_scores, sample_query_bundle
        )

        # Should fall back to original ordering, truncated to top_n
        assert len(result) == 3
        assert result == sample_nodes_with_scores[:3]

    def test_postprocess_nodes_oom_error_handling(
        self, mock_sentence_transformers, sample_nodes_with_scores, sample_query_bundle
    ):
        """Test CUDA OOM error handling."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        mock_model.predict.side_effect = torch.cuda.OutOfMemoryError(
            "CUDA out of memory"
        )

        reranker = BGECrossEncoderRerank(top_n=2)
        result = reranker.postprocess_nodes(
            sample_nodes_with_scores, sample_query_bundle
        )

        # Should fall back to original nodes truncated
        assert len(result) == 2
        assert result == sample_nodes_with_scores[:2]

    def test_postprocess_nodes_batch_size_handling(
        self, mock_sentence_transformers, sample_nodes_with_scores, sample_query_bundle
    ):
        """Test batch size parameter is used correctly."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        mock_model.predict.return_value = np.array([0.95, 0.87, 0.72, 0.65, 0.52])

        reranker = BGECrossEncoderRerank(batch_size=2)
        reranker.postprocess_nodes(sample_nodes_with_scores, sample_query_bundle)

        # Verify batch_size was passed to predict
        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["batch_size"] == 2
        assert call_kwargs["show_progress_bar"] is False

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_avoids_double_normalization_when_scores_in_unit_interval(self, mock_cross):
        """When model returns probabilities in [0,1], skip extra sigmoid."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])
        mock_cross.return_value = mock_model

        reranker = BGECrossEncoderRerank(top_n=3, normalize_scores=True)

        nodes = [
            NodeWithScore(node=TextNode(text="a", id_="1"), score=0.1),
            NodeWithScore(node=TextNode(text="b", id_="2"), score=0.1),
            NodeWithScore(node=TextNode(text="c", id_="3"), score=0.1),
        ]
        qb = QueryBundle(query_str="q")

        result = reranker.postprocess_nodes(nodes, qb)
        scores = [n.score for n in result]
        assert scores == [0.9, 0.8, 0.7]

    @patch("src.retrieval.reranking.CrossEncoder")
    def test_applies_sigmoid_when_logits_returned(self, mock_cross):
        """When model returns logits, apply sigmoid normalization."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        logits = np.array([2.0, 0.0, -1.0])
        expected = torch.sigmoid(torch.tensor(logits)).numpy().tolist()

        mock_model = Mock()
        mock_model.predict.return_value = logits
        mock_cross.return_value = mock_model

        reranker = BGECrossEncoderRerank(top_n=3, normalize_scores=True)

        nodes = [
            NodeWithScore(node=TextNode(text="a", id_="1"), score=0.1),
            NodeWithScore(node=TextNode(text="b", id_="2"), score=0.1),
            NodeWithScore(node=TextNode(text="c", id_="3"), score=0.1),
        ]
        qb = QueryBundle(query_str="q")

        result = reranker.postprocess_nodes(nodes, qb)
        scores = [pytest.approx(float(n.score), rel=1e-6) for n in result]
        for got, exp in zip(scores, expected, strict=False):
            assert got == pytest.approx(exp, rel=1e-6)


@pytest.mark.unit
class TestBGECrossEncoderRerankUtilities:
    """Test utility methods and information functions."""

    def test_get_model_info(self, mock_sentence_transformers):
        """Test model information retrieval."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers

        reranker = BGECrossEncoderRerank(
            model_name="test/model",
            top_n=10,
            device="cpu",
            use_fp16=False,
            max_length=256,
            batch_size=8,
            normalize_scores=False,
        )

        info = reranker.get_model_info()

        expected_info = {
            "model_name": "test/model",
            "device": "cpu",
            "use_fp16": False,
            "max_length": 256,
            "batch_size": 8,
            "normalize_scores": False,
            "top_n": 10,
        }

        assert info == expected_info


@pytest.mark.unit
class TestBGECrossEncoderRerankFactoryFunctions:
    """Test factory functions for creating reranker instances."""

    @patch("src.retrieval.reranking.torch.cuda.is_available", return_value=True)
    def test_create_bge_cross_encoder_reranker_defaults(
        self, mock_cuda, mock_sentence_transformers
    ):
        """Test factory function with default parameters."""
        from src.retrieval.reranking import create_bge_cross_encoder_reranker

        mock_class, mock_model = mock_sentence_transformers

        reranker = create_bge_cross_encoder_reranker()

        assert reranker.model_name == "BAAI/bge-reranker-v2-m3"
        assert reranker.top_n == 5
        assert reranker.use_fp16 is True
        assert reranker.device == "cuda"
        assert reranker.batch_size >= 16  # Should be at least minimum for CUDA

    @patch("src.retrieval.reranking.torch.cuda.is_available", return_value=True)
    def test_create_bge_cross_encoder_reranker_custom(
        self, mock_cuda, mock_sentence_transformers
    ):
        """Test factory function with custom parameters."""
        from src.retrieval.reranking import create_bge_cross_encoder_reranker

        mock_class, mock_model = mock_sentence_transformers

        reranker = create_bge_cross_encoder_reranker(
            model_name="custom/model",
            top_n=8,
            use_fp16=False,
            device="cpu",
            batch_size=4,
        )

        assert reranker.model_name == "custom/model"
        assert reranker.top_n == 8
        assert reranker.use_fp16 is False
        assert reranker.device == "cpu"
        assert reranker.batch_size == 4  # Should respect CPU limit

    def test_create_bge_cross_encoder_reranker_cpu_batch_limit(
        self, mock_sentence_transformers
    ):
        """Test factory function enforces CPU batch size limits."""
        from src.retrieval.reranking import create_bge_cross_encoder_reranker

        mock_class, mock_model = mock_sentence_transformers

        reranker = create_bge_cross_encoder_reranker(
            device="cpu",
            batch_size=10,  # Above CPU limit
        )

        # Should be limited to CPU maximum
        assert reranker.batch_size == 4

    @patch("src.retrieval.reranking.torch.cuda.is_available", return_value=True)
    def test_create_bge_cross_encoder_reranker_cuda_batch_minimum(
        self, mock_cuda, mock_sentence_transformers
    ):
        """Test factory function enforces CUDA batch size minimum."""
        from src.retrieval.reranking import create_bge_cross_encoder_reranker

        mock_class, mock_model = mock_sentence_transformers

        reranker = create_bge_cross_encoder_reranker(
            device="cuda",
            batch_size=8,  # Below CUDA minimum
        )

        # Should be increased to minimum for CUDA
        assert reranker.batch_size == 16


@pytest.mark.unit
class TestBGECrossEncoderRerankBenchmarking:
    """Test performance benchmarking functionality."""

    def test_benchmark_reranking_latency_basic(self, mock_sentence_transformers):
        """Test basic latency benchmarking."""
        from src.retrieval.reranking import (
            BGECrossEncoderRerank,
            benchmark_reranking_latency,
        )

        mock_class, mock_model = mock_sentence_transformers
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])

        reranker = BGECrossEncoderRerank()

        query = "test query"
        documents = ["doc1", "doc2", "doc3"]

        with patch(
            "time.perf_counter", side_effect=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        ):  # Simulate timing
            stats = benchmark_reranking_latency(reranker, query, documents, num_runs=2)

        assert "mean_latency_ms" in stats
        assert "min_latency_ms" in stats
        assert "max_latency_ms" in stats
        assert "num_documents" in stats
        assert "target_latency_ms" in stats

        assert stats["num_documents"] == 3
        assert stats["target_latency_ms"] == 100.0  # RTX 4090 target
        assert stats["mean_latency_ms"] > 0

    def test_benchmark_reranking_latency_warmup(self, mock_sentence_transformers):
        """Test benchmarking includes warmup phase."""
        from src.retrieval.reranking import (
            BGECrossEncoderRerank,
            benchmark_reranking_latency,
        )

        mock_class, mock_model = mock_sentence_transformers

        # Mock to track how many times predict is called
        predict_calls = []

        def mock_predict(*args, **kwargs):
            predict_calls.append((args, kwargs))
            return np.array([0.9, 0.8, 0.7])

        mock_model.predict = mock_predict

        reranker = BGECrossEncoderRerank()

        query = "test query"
        documents = ["doc1", "doc2", "doc3"]

        benchmark_reranking_latency(reranker, query, documents, num_runs=2)

        # Should call predict for warmup (3 docs) + 2 benchmark runs
        # Each run processes all 3 documents
        assert len(predict_calls) == 3  # warmup + 2 runs

    def test_benchmark_reranking_latency_performance_calculation(
        self, mock_sentence_transformers
    ):
        """Test latency calculation accuracy."""
        from src.retrieval.reranking import (
            BGECrossEncoderRerank,
            benchmark_reranking_latency,
        )

        mock_class, mock_model = mock_sentence_transformers
        mock_model.predict.return_value = np.array([0.9])

        reranker = BGECrossEncoderRerank()

        # Mock precise timing with 2 runs (50ms, 100ms)
        timing_sequence = [
            0.00,
            0.05,  # Run 1
            0.05,
            0.15,  # Run 2
        ]

        with patch("time.perf_counter", side_effect=timing_sequence):
            stats = benchmark_reranking_latency(reranker, "query", ["doc"], num_runs=2)

        # Validate monotonicity and basic properties
        assert (
            stats["min_latency_ms"]
            <= stats["mean_latency_ms"]
            <= stats["max_latency_ms"]
        )
        assert stats["min_latency_ms"] > 0
        assert stats["max_latency_ms"] >= stats["min_latency_ms"]


@pytest.mark.unit
class TestBGECrossEncoderRerankEdgeCases:
    """Test edge cases and error conditions."""

    def test_postprocess_nodes_very_long_text(
        self, mock_sentence_transformers, sample_query_bundle
    ):
        """Test handling of very long document text."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        mock_model.predict.return_value = np.array([0.9])

        reranker = BGECrossEncoderRerank(max_length=512, normalize_scores=False)

        # Create node with very long text
        long_text = "word " * 1000  # 1000 words, likely > 512 tokens
        long_node = NodeWithScore(node=TextNode(text=long_text, id_="long"), score=0.8)

        result = reranker.postprocess_nodes([long_node], sample_query_bundle)

        # Should handle long text without error
        assert len(result) == 1
        # For single node, reranker returns original node without scoring
        assert result[0].score == 0.8

    def test_postprocess_nodes_special_characters(
        self, mock_sentence_transformers, sample_query_bundle
    ):
        """Test handling of special characters in text."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        mock_model.predict.return_value = np.array([0.85])

        reranker = BGECrossEncoderRerank(normalize_scores=False)

        # Text with special characters and Unicode
        special_text = "Héllo wörld! 🌍 Special chars: <>&\"' \\n\\t"
        special_node = NodeWithScore(
            node=TextNode(text=special_text, id_="special"), score=0.7
        )

        result = reranker.postprocess_nodes([special_node], sample_query_bundle)

        assert len(result) == 1
        # For single node, reranker returns original node without scoring
        assert result[0].score == 0.7

    def test_postprocess_nodes_empty_text_content(
        self, mock_sentence_transformers, sample_query_bundle
    ):
        """Test handling of nodes with empty text content."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers
        mock_model.predict.return_value = np.array([0.5])

        reranker = BGECrossEncoderRerank(normalize_scores=False)

        empty_node = NodeWithScore(node=TextNode(text="", id_="empty"), score=0.6)

        result = reranker.postprocess_nodes([empty_node], sample_query_bundle)

        assert len(result) == 1
        # For single node, reranker returns original node without scoring
        assert result[0].score == 0.6

    def test_postprocess_nodes_scoring_edge_values(
        self, mock_sentence_transformers, sample_nodes_with_scores, sample_query_bundle
    ):
        """Test handling of extreme scoring values."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers

        # Test extreme values including inf, -inf, nan
        extreme_scores = np.array([float("inf"), float("-inf"), float("nan"), 0.0, 1.0])
        mock_model.predict.return_value = extreme_scores

        reranker = BGECrossEncoderRerank(normalize_scores=False)

        result = reranker.postprocess_nodes(
            sample_nodes_with_scores, sample_query_bundle
        )

        # Should handle extreme values gracefully
        assert len(result) >= 1

        # Check that non-NaN scores are properly handled
        finite_scores = [node.score for node in result if not np.isnan(node.score)]
        assert len(finite_scores) >= 2  # Should have at least finite scores

    def test_get_model_info_all_combinations(self, mock_sentence_transformers):
        """Test get_model_info with all parameter combinations."""
        from src.retrieval.reranking import BGECrossEncoderRerank

        mock_class, mock_model = mock_sentence_transformers

        # Test various configuration combinations
        configs = [
            {
                "model_name": "BAAI/bge-reranker-base",
                "device": "cpu",
                "use_fp16": False,
                "normalize_scores": False,
                "top_n": 3,
                "batch_size": 4,
                "max_length": 256,
            },
            {
                "model_name": "BAAI/bge-reranker-large",
                "device": "cuda",
                "use_fp16": True,
                "normalize_scores": True,
                "top_n": 10,
                "batch_size": 32,
                "max_length": 1024,
            },
        ]

        for config in configs:
            reranker = BGECrossEncoderRerank(**config)
            info = reranker.get_model_info()

            # Verify all configuration values are returned
            for key, expected_value in config.items():
                assert info[key] == expected_value
