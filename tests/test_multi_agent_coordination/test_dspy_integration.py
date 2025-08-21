"""Comprehensive unit tests for DSPy Integration (ADR-018 compliant).

Tests cover:
- Real DSPy integration vs mock implementation
- Query optimization and refinement
- Query variant generation
- LLM wrapper functionality
- Pipeline optimization with bootstrapping
- Error handling and fallback mechanisms
- Performance measurement and quality scoring
"""

import time
from unittest.mock import Mock, patch

from src.dspy_integration import (
    DSPyLlamaIndexRetriever,
    QueryOptimizationResult,
    get_dspy_retriever,
    is_dspy_available,
)


class TestDSPyAvailability:
    """Test suite for DSPy availability detection."""

    def test_dspy_availability_true(self):
        """Test DSPy availability detection when available."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            assert is_dspy_available() is True

    def test_dspy_availability_false(self):
        """Test DSPy availability detection when not available."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            assert is_dspy_available() is False


class TestQueryOptimizationResult:
    """Test suite for QueryOptimizationResult data model."""

    def test_optimization_result_creation(self):
        """Test creating optimization result with all fields."""
        result = QueryOptimizationResult(
            original="What is AI?",
            refined="What is artificial intelligence and how does it work?",
            variants=[
                "Explain artificial intelligence concepts",
                "Define AI and its applications",
            ],
            optimization_time=0.125,
            quality_score=0.85,
        )

        assert result.original == "What is AI?"
        assert "artificial intelligence" in result.refined
        assert len(result.variants) == 2
        assert result.optimization_time == 0.125
        assert result.quality_score == 0.85

    def test_optimization_result_defaults(self):
        """Test optimization result with default values."""
        result = QueryOptimizationResult(
            original="Test query",
            refined="Refined test query",
        )

        assert result.original == "Test query"
        assert result.refined == "Refined test query"
        assert result.variants == []
        assert result.optimization_time == 0.0
        assert result.quality_score == 0.0


class TestDSPyLlamaIndexRetriever:
    """Test suite for DSPyLlamaIndexRetriever."""

    def test_initialization_with_dspy_available(self, mock_llm: Mock):
        """Test retriever initialization when DSPy is available."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy") as mock_dspy,
        ):
            # Mock DSPy components
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()

            retriever = DSPyLlamaIndexRetriever(llm=mock_llm, max_variants=3)

            assert retriever.llm == mock_llm
            assert retriever.max_variants == 3
            assert retriever.optimization_enabled is True

            # Verify DSPy was configured
            mock_dspy.configure.assert_called_once()
            assert (
                mock_dspy.ChainOfThought.call_count == 2
            )  # query_refiner + variant_generator

    def test_initialization_with_dspy_unavailable(self, mock_llm: Mock):
        """Test retriever initialization when DSPy is not available."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)

            assert retriever.llm == mock_llm
            assert retriever.optimization_enabled is False

    def test_initialization_without_llm(self):
        """Test retriever initialization without LLM."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            retriever = DSPyLlamaIndexRetriever()

            assert retriever.llm is None
            assert retriever.optimization_enabled is False

    def test_llm_wrapper_creation(self, mock_llm: Mock):
        """Test LLM wrapper for DSPy compatibility."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            retriever = DSPyLlamaIndexRetriever()

            # Mock LLM response
            mock_response = Mock()
            mock_response.text = "Test response"
            mock_llm.complete.return_value = mock_response

            wrapper = retriever._wrap_llm_for_dspy(mock_llm)
            result = wrapper("Test prompt")

            assert result == "Test response"
            mock_llm.complete.assert_called_once_with("Test prompt")

    def test_llm_wrapper_error_handling(self, mock_llm: Mock):
        """Test LLM wrapper error handling."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            retriever = DSPyLlamaIndexRetriever()

            # Mock LLM to raise exception
            mock_llm.complete.side_effect = Exception("LLM error")

            wrapper = retriever._wrap_llm_for_dspy(mock_llm)
            result = wrapper("Test prompt")

            assert result == ""  # Should return empty string on error

    def test_optimize_query_with_dspy_available(self, mock_llm: Mock):
        """Test query optimization when DSPy is available."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy") as mock_dspy,
        ):
            # Mock DSPy components
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()

            # Mock query refiner
            mock_refined_result = Mock()
            mock_refined_result.refined_query = "Optimized query about machine learning"

            # Mock variant generator
            mock_variant_result = Mock()
            mock_variant_result.variant1 = "What is machine learning?"
            mock_variant_result.variant2 = "Explain ML concepts"

            # Create retriever with mocked components
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock(return_value=mock_refined_result)
            retriever.variant_generator = Mock(return_value=mock_variant_result)

            result = retriever.optimize_query("ML", llm=mock_llm, enable_variants=True)

            # Verify optimization results
            assert result["original"] == "ML"
            assert result["refined"] == "Optimized query about machine learning"
            assert result["variants"] == [
                "What is machine learning?",
                "Explain ML concepts",
            ]
            assert result["optimized"] is True
            assert result["optimization_time"] > 0
            assert result["quality_score"] > 0

    def test_optimize_query_with_dspy_unavailable(self):
        """Test query optimization fallback when DSPy is unavailable."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            result = DSPyLlamaIndexRetriever.optimize_query("AI")

            # Verify fallback optimization
            assert result["original"] == "AI"
            assert (
                result["refined"] == "Find information about AI"
            )  # Short query enhancement
            assert "What is AI?" in result["variants"]
            assert "Explain AI" in result["variants"]
            assert result["optimized"] is False
            assert result["quality_score"] == 0.3  # Lower score for fallback

    def test_optimize_query_short_query_enhancement(self):
        """Test optimization of short queries."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            result = DSPyLlamaIndexRetriever.optimize_query("ML")

            # Verify short query enhancement
            assert result["original"] == "ML"
            assert result["refined"] == "Find information about ML"
            assert len(result["variants"]) == 2

    def test_optimize_query_comparison_enhancement(self):
        """Test optimization of comparison queries."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            result = DSPyLlamaIndexRetriever.optimize_query("compare AI vs ML")

            # Verify comparison query enhancement
            assert result["original"] == "compare AI vs ML"
            assert "difference between" in result["variants"][0]
            assert "similarities and differences" in result["variants"][1]

    def test_optimize_query_with_refinement_failure(self, mock_llm: Mock):
        """Test query optimization when refinement fails."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy") as mock_dspy,
        ):
            # Mock DSPy components
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()

            # Create retriever with failing refiner
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock(side_effect=Exception("Refinement failed"))
            retriever.variant_generator = Mock()

            result = retriever.optimize_query("test query", llm=mock_llm)

            # Should fall back to original query
            assert result["original"] == "test query"
            assert result["refined"] == "test query"  # Fallback to original

    def test_optimize_query_with_variant_failure(self, mock_llm: Mock):
        """Test query optimization when variant generation fails."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy") as mock_dspy,
        ):
            # Mock DSPy components
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()

            # Mock successful refiner
            mock_refined_result = Mock()
            mock_refined_result.refined_query = "Refined query"

            # Create retriever with failing variant generator
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock(return_value=mock_refined_result)
            retriever.variant_generator = Mock(
                side_effect=Exception("Variant generation failed")
            )

            result = retriever.optimize_query(
                "test query", llm=mock_llm, enable_variants=True
            )

            # Should succeed with empty variants
            assert result["original"] == "test query"
            assert result["refined"] == "Refined query"
            assert result["variants"] == []  # Empty due to failure

    def test_optimize_query_variants_disabled(self, mock_llm: Mock):
        """Test query optimization with variants disabled."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy") as mock_dspy,
        ):
            # Mock DSPy components
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()

            # Mock successful refiner
            mock_refined_result = Mock()
            mock_refined_result.refined_query = "Refined query"

            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock(return_value=mock_refined_result)
            retriever.variant_generator = Mock()  # Should not be called

            result = retriever.optimize_query(
                "test query", llm=mock_llm, enable_variants=False
            )

            # Verify variants were not generated
            assert result["variants"] == []
            retriever.variant_generator.assert_not_called()

    def test_optimize_query_max_variants_limit(self, mock_llm: Mock):
        """Test query optimization respects max variants limit."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy") as mock_dspy,
        ):
            # Mock DSPy components
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()

            # Mock components
            mock_refined_result = Mock()
            mock_refined_result.refined_query = "Refined query"

            mock_variant_result = Mock()
            mock_variant_result.variant1 = "Variant 1"
            mock_variant_result.variant2 = "Variant 2"

            # Create retriever with max_variants=1
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm, max_variants=1)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock(return_value=mock_refined_result)
            retriever.variant_generator = Mock(return_value=mock_variant_result)

            result = retriever.optimize_query(
                "test query", llm=mock_llm, enable_variants=True
            )

            # Should limit to max_variants
            assert len(result["variants"]) == 1
            assert result["variants"][0] == "Variant 1"

    def test_optimize_query_class_method_instance_creation(self, mock_llm: Mock):
        """Test class method creates instance when needed."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            # Clear any existing instance
            if hasattr(DSPyLlamaIndexRetriever, "_instance"):
                DSPyLlamaIndexRetriever._instance = None

            result = DSPyLlamaIndexRetriever.optimize_query("test", llm=mock_llm)

            # Should create instance and use fallback
            assert result["optimized"] is False
            assert hasattr(DSPyLlamaIndexRetriever, "_instance")

    def test_estimate_quality_improvement(self):
        """Test quality improvement estimation."""
        # Test with longer refined query
        score = DSPyLlamaIndexRetriever._estimate_quality_improvement(
            "AI", "What is artificial intelligence and how does it work?", ["Variant 1"]
        )
        assert score > 0.5  # Should be above base

        # Test with specific keywords
        score = DSPyLlamaIndexRetriever._estimate_quality_improvement(
            "AI", "Explain specific details about AI", []
        )
        assert score > 0.5  # "specific" keyword bonus

        # Test with multiple variants
        score = DSPyLlamaIndexRetriever._estimate_quality_improvement(
            "AI", "Enhanced AI query", ["Variant 1", "Variant 2"]
        )
        assert score > 0.7  # Variants bonus

    def test_optimize_retrieval_pipeline_with_dspy_available(self, mock_llm: Mock):
        """Test retrieval pipeline optimization when DSPy is available."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy") as mock_dspy,
            patch("src.dspy_integration.BootstrapFewShot") as mock_bootstrap,
        ):
            # Mock DSPy components
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()
            mock_dspy.Example = Mock(side_effect=lambda **kwargs: Mock(**kwargs))

            # Mock bootstrap optimizer
            mock_optimizer = Mock()
            mock_optimized_refiner = Mock()
            mock_optimizer.compile.return_value = mock_optimized_refiner
            mock_bootstrap.return_value = mock_optimizer

            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock()

            mock_retriever = Mock()
            queries = ["What is AI?", "Explain ML", "Define neural networks"]

            result = retriever.optimize_retrieval_pipeline(
                mock_retriever, queries, num_examples=3
            )

            # Verify optimization was performed
            assert result == mock_retriever
            mock_bootstrap.assert_called_once_with(max_bootstrapped_demos=3)
            mock_optimizer.compile.assert_called_once()
            assert retriever.query_refiner == mock_optimized_refiner

    def test_optimize_retrieval_pipeline_with_dspy_unavailable(self, mock_llm: Mock):
        """Test retrieval pipeline optimization when DSPy is unavailable."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)

            mock_retriever = Mock()
            queries = ["What is AI?"]

            result = retriever.optimize_retrieval_pipeline(mock_retriever, queries)

            # Should return original retriever
            assert result == mock_retriever

    def test_optimize_retrieval_pipeline_error_handling(self, mock_llm: Mock):
        """Test retrieval pipeline optimization error handling."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy") as mock_dspy,
            patch("src.dspy_integration.BootstrapFewShot"),
        ):
            # Mock DSPy components
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()
            mock_dspy.Example = Mock(side_effect=Exception("Example creation failed"))

            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True

            mock_retriever = Mock()
            queries = ["What is AI?"]

            result = retriever.optimize_retrieval_pipeline(mock_retriever, queries)

            # Should return original retriever on error
            assert result == mock_retriever


class TestGlobalInstance:
    """Test suite for global DSPy retriever instance management."""

    def test_get_dspy_retriever_creates_instance(self, mock_llm: Mock):
        """Test global retriever instance creation."""
        with patch("src.dspy_integration._dspy_retriever_instance", None):
            retriever = get_dspy_retriever(llm=mock_llm)

            assert isinstance(retriever, DSPyLlamaIndexRetriever)
            assert retriever.llm == mock_llm

    def test_get_dspy_retriever_reuses_instance(self, mock_llm: Mock):
        """Test global retriever instance reuse."""
        # Create first instance
        first_retriever = get_dspy_retriever(llm=mock_llm)

        # Get second instance
        second_retriever = get_dspy_retriever()

        # Should be the same instance
        assert first_retriever is second_retriever

    def test_get_dspy_retriever_without_llm(self):
        """Test global retriever instance creation without LLM."""
        with patch("src.dspy_integration._dspy_retriever_instance", None):
            retriever = get_dspy_retriever()

            assert isinstance(retriever, DSPyLlamaIndexRetriever)
            assert retriever.llm is None


class TestPerformanceAndTiming:
    """Test suite for performance measurement and timing."""

    def test_optimization_timing_measurement(self):
        """Test optimization timing is measured correctly."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            start_time = time.perf_counter()
            result = DSPyLlamaIndexRetriever.optimize_query("test query")
            end_time = time.perf_counter()

            reported_time = result["optimization_time"]
            actual_time = end_time - start_time

            # Verify timing is reasonable
            assert reported_time > 0
            assert reported_time <= actual_time + 0.01  # Allow small overhead

    def test_optimization_performance_with_variants(self):
        """Test optimization performance with variant generation."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            result = DSPyLlamaIndexRetriever.optimize_query(
                "compare machine learning vs deep learning", enable_variants=True
            )

            # Should complete quickly even with variants
            assert result["optimization_time"] < 1.0  # Should be very fast for fallback
            assert len(result["variants"]) > 0


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""

    def test_optimize_empty_query(self):
        """Test optimization with empty query."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            result = DSPyLlamaIndexRetriever.optimize_query("")

            assert result["original"] == ""
            assert result["refined"] == ""  # Should handle gracefully
            assert result["optimized"] is False

    def test_optimize_very_long_query(self):
        """Test optimization with very long query."""
        long_query = "What is " + "machine learning " * 100  # Very long query

        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            result = DSPyLlamaIndexRetriever.optimize_query(long_query)

            assert result["original"] == long_query
            assert len(result["refined"]) > 0
            assert result["optimized"] is False

    def test_optimization_with_none_llm(self):
        """Test optimization with None LLM."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            result = DSPyLlamaIndexRetriever.optimize_query("test", llm=None)

            # Should fall back to basic optimization
            assert result["optimized"] is False

    def test_llm_wrapper_with_invalid_response(self, mock_llm: Mock):
        """Test LLM wrapper with invalid response format."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            retriever = DSPyLlamaIndexRetriever()

            # Mock LLM to return non-string response
            mock_llm.complete.return_value = None

            wrapper = retriever._wrap_llm_for_dspy(mock_llm)
            result = wrapper("Test prompt")

            assert result == "None"  # Should convert to string

    def test_quality_estimation_edge_cases(self):
        """Test quality estimation with edge cases."""
        # Test with same original and refined
        score = DSPyLlamaIndexRetriever._estimate_quality_improvement(
            "test", "test", []
        )
        assert score == 0.5  # Base score

        # Test with very long refined query
        score = DSPyLlamaIndexRetriever._estimate_quality_improvement(
            "AI",
            "A" * 1000,
            ["variant"] * 10,  # Many variants
        )
        assert score <= 1.0  # Should not exceed maximum

    def test_optimize_query_exception_handling(self):
        """Test complete exception handling in optimize_query."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("time.perf_counter", side_effect=Exception("Timer error")),
        ):
            result = DSPyLlamaIndexRetriever.optimize_query("test")

            # Should fall back gracefully
            assert result["original"] == "test"
            assert result["optimized"] is False
