"""Integration tests for DSPy configuration and setup validation.

This module tests the real DSPy integration configuration, initialization,
and boundary validation without testing DSPy library internals.

Focus areas:
- Configuration loading and validation
- DSPy initialization and wrapper setup
- Integration boundary testing with mocked LLM calls
- Error handling for invalid configurations
- Basic query optimization integration
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.dspy_integration import (
    DEFAULT_MAX_VARIANTS,
    DSPyLlamaIndexRetriever,
    QueryOptimizationResult,
    get_dspy_retriever,
    is_dspy_available,
)


@pytest.mark.integration
class TestDSPyAvailabilityIntegration:
    """Integration tests for DSPy availability and import handling."""

    def test_dspy_import_detection_real(self):
        """Test real DSPy import detection without mocking."""
        # Test the actual import detection mechanism
        available = is_dspy_available()
        assert isinstance(available, bool)

        # Should be consistent with module-level constant
        from src.dspy_integration import DSPY_AVAILABLE

        assert available == DSPY_AVAILABLE

    @patch("src.dspy_integration.DSPY_AVAILABLE", False)
    def test_dspy_unavailable_fallback_integration(self):
        """Test integration behavior when DSPy is unavailable."""
        # Test retriever creation without DSPy
        retriever = DSPyLlamaIndexRetriever()
        assert retriever.optimization_enabled is False

        # Test optimization fallback
        result = DSPyLlamaIndexRetriever.optimize_query("test query")
        assert result["optimized"] is False
        assert result["quality_score"] < 1.0


@pytest.mark.integration
class TestDSPyConfigurationValidation:
    """Integration tests for DSPy configuration and initialization."""

    def test_retriever_initialization_with_valid_config(self):
        """Test DSPy retriever initialization with valid configuration."""
        mock_llm = Mock()
        mock_llm.complete = Mock(return_value=Mock(text="test response"))

        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            with patch("src.dspy_integration.dspy") as mock_dspy:
                # Setup DSPy mocks for configuration
                mock_dspy.configure = Mock()
                mock_dspy.ChainOfThought = Mock()

                # Test initialization with different configurations
                configs = [
                    {"llm": mock_llm, "max_variants": 2},
                    {"llm": mock_llm, "max_variants": 5},
                    {"llm": None, "max_variants": DEFAULT_MAX_VARIANTS},
                ]

                for config in configs:
                    retriever = DSPyLlamaIndexRetriever(**config)

                    # Validate configuration was applied
                    assert retriever.llm == config["llm"]
                    assert retriever.max_variants == config["max_variants"]

                    # Validate optimization state based on config
                    expected_enabled = config["llm"] is not None
                    if not expected_enabled:
                        assert retriever.optimization_enabled is False

    def test_llm_wrapper_configuration_integration(self):
        """Test LLM wrapper configuration and integration."""
        mock_llm = Mock()

        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            retriever = DSPyLlamaIndexRetriever()

            # Test wrapper creation with different LLM types
            wrapper = retriever._wrap_llm_for_dspy(mock_llm)

            # Validate wrapper interface
            assert callable(wrapper)
            assert hasattr(wrapper, "llm")
            assert wrapper.llm == mock_llm

    def test_configuration_validation_with_invalid_params(self):
        """Test configuration validation with invalid parameters."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            # Test that constructor accepts various parameter types
            # (the integration test focuses on behavior, not strict validation)

            # Test with negative max_variants (should handle gracefully)
            retriever = DSPyLlamaIndexRetriever(max_variants=-1)
            assert retriever.max_variants == -1  # Constructor should accept it

            # Test with very large max_variants
            retriever = DSPyLlamaIndexRetriever(max_variants=1000)
            assert retriever.max_variants == 1000


@pytest.mark.integration
class TestDSPyIntegrationBoundaries:
    """Integration tests for DSPy boundary testing with external dependencies."""

    def test_query_optimization_integration_with_mocked_llm(self):
        """Test query optimization integration with mocked LLM responses."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "optimized query about machine learning algorithms"
        mock_llm.complete = Mock(return_value=mock_response)

        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            with patch("src.dspy_integration.dspy") as mock_dspy:
                # Setup DSPy mocks
                mock_dspy.configure = Mock()
                mock_dspy.ChainOfThought = Mock()

                # Create mock optimization modules
                mock_refiner = Mock()
                mock_refined_result = Mock()
                mock_refined_result.refined_query = "optimized machine learning query"
                mock_refiner.return_value = mock_refined_result

                mock_variant_generator = Mock()
                mock_variant_result = Mock()
                mock_variant_result.variant1 = "What is machine learning?"
                mock_variant_result.variant2 = "Explain ML concepts"
                mock_variant_generator.return_value = mock_variant_result

                retriever = DSPyLlamaIndexRetriever(llm=mock_llm, max_variants=2)
                retriever.optimization_enabled = True
                retriever.query_refiner = mock_refiner
                retriever.variant_generator = mock_variant_generator

                # Test optimization integration
                result = retriever.optimize_query(
                    "ML algorithms", llm=mock_llm, enable_variants=True
                )

                # Validate integration results
                assert result["original"] == "ML algorithms"
                assert result["refined"] == "optimized machine learning query"
                assert len(result["variants"]) == 2
                assert result["optimization_time"] > 0
                assert result["quality_score"] > 0
                assert result["optimized"] is True

                # Verify integration calls
                mock_refiner.assert_called_once()
                mock_variant_generator.assert_called_once()

    def test_optimization_pipeline_integration(self):
        """Test retrieval pipeline optimization integration."""
        mock_llm = Mock()
        mock_retriever = Mock()

        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            with patch("src.dspy_integration.dspy") as mock_dspy:
                with patch("src.dspy_integration.BootstrapFewShot") as mock_bootstrap:
                    # Setup DSPy mocks
                    mock_dspy.configure = Mock()
                    mock_dspy.ChainOfThought = Mock()
                    mock_dspy.Example = Mock()

                    # Setup bootstrap optimizer
                    mock_optimizer = Mock()
                    mock_optimized_refiner = Mock()
                    mock_optimizer.compile = Mock(return_value=mock_optimized_refiner)
                    mock_bootstrap.return_value = mock_optimizer

                    retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
                    retriever.optimization_enabled = True
                    retriever.query_refiner = Mock()

                    # Test pipeline optimization integration
                    test_queries = [
                        "What is AI?",
                        "Explain ML",
                        "Define neural networks",
                    ]
                    result = retriever.optimize_retrieval_pipeline(
                        mock_retriever, test_queries, num_examples=3
                    )

                    # Validate integration
                    assert result == mock_retriever
                    mock_bootstrap.assert_called_once_with(max_bootstrapped_demos=3)
                    mock_optimizer.compile.assert_called_once()

    def test_global_instance_integration(self):
        """Test global DSPy retriever instance management integration."""
        # Clear any existing instance for clean test
        import src.dspy_integration

        src.dspy_integration.DSPY_RETRIEVER_INSTANCE = None

        mock_llm = Mock()

        # Test instance creation
        first_retriever = get_dspy_retriever(llm=mock_llm)
        assert isinstance(first_retriever, DSPyLlamaIndexRetriever)
        assert first_retriever.llm == mock_llm

        # Test instance reuse
        second_retriever = get_dspy_retriever()
        assert first_retriever is second_retriever

        # Verify global state persistence
        assert src.dspy_integration.DSPY_RETRIEVER_INSTANCE is not None


@pytest.mark.integration
class TestDSPyErrorHandlingIntegration:
    """Integration tests for DSPy error handling and recovery."""

    def test_dspy_initialization_failure_recovery(self):
        """Test recovery from DSPy initialization failures."""
        mock_llm = Mock()

        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            with patch("src.dspy_integration.dspy") as mock_dspy:
                # Simulate DSPy configuration failure
                mock_dspy.configure = Mock(
                    side_effect=RuntimeError("DSPy config failed")
                )
                mock_dspy.ChainOfThought = Mock()

                retriever = DSPyLlamaIndexRetriever(llm=mock_llm)

                # Should gracefully handle failure
                assert retriever.optimization_enabled is False
                assert retriever.llm == mock_llm

    def test_llm_wrapper_error_handling_integration(self):
        """Test LLM wrapper error handling integration."""
        mock_llm = Mock()

        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            retriever = DSPyLlamaIndexRetriever()

            # Test different LLM failure scenarios
            error_scenarios = [
                RuntimeError("LLM runtime error"),
                AttributeError("Missing method"),
                ValueError("Invalid input"),
            ]

            for error in error_scenarios:
                mock_llm.complete = Mock(side_effect=error)
                wrapper = retriever._wrap_llm_for_dspy(mock_llm)

                # Should handle errors gracefully
                result = wrapper("test prompt")
                assert result == ""  # Empty string fallback

    def test_optimization_with_partial_failures(self):
        """Test optimization integration with partial component failures."""
        mock_llm = Mock()

        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            with patch("src.dspy_integration.dspy") as mock_dspy:
                mock_dspy.configure = Mock()
                mock_dspy.ChainOfThought = Mock()

                retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
                retriever.optimization_enabled = True

                # Test refinement failure, variant success
                retriever.query_refiner = Mock(side_effect=Exception("Refiner failed"))

                mock_variant_result = Mock()
                mock_variant_result.variant1 = "variant 1"
                mock_variant_result.variant2 = "variant 2"
                retriever.variant_generator = Mock(return_value=mock_variant_result)

                result = retriever.optimize_query(
                    "test query", llm=mock_llm, enable_variants=True
                )

                # Should fallback for refinement but succeed with variants
                assert result["original"] == "test query"
                assert result["refined"] == "test query"  # Fallback to original
                assert len(result["variants"]) == 2
                assert result["optimized"] is True


@pytest.mark.integration
class TestDSPyPerformanceIntegration:
    """Integration tests for DSPy performance characteristics."""

    def test_optimization_timing_integration(self):
        """Test optimization timing measurement in integration context."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            start_time = time.perf_counter()

            result = DSPyLlamaIndexRetriever.optimize_query(
                "performance test query", enable_variants=True
            )

            end_time = time.perf_counter()
            actual_time = end_time - start_time
            reported_time = result["optimization_time"]

            # Validate timing measurement accuracy
            assert reported_time > 0
            assert (
                reported_time <= actual_time + 0.01
            )  # Allow small measurement overhead

            # Fallback optimization should be fast
            assert reported_time < 0.1  # Should complete in <100ms

    def test_quality_estimation_integration(self):
        """Test quality estimation integration with different scenarios."""
        scenarios = [
            {
                "original": "AI",
                "refined": "What is artificial intelligence and how does it work?",
                "variants": ["Explain AI", "Define artificial intelligence"],
                "expected_min": 0.7,
            },
            {
                "original": "short",
                "refined": "short",  # No improvement
                "variants": [],
                "expected_min": 0.4,
                "expected_max": 0.6,
            },
            {
                "original": "compare X Y",
                "refined": "Compare the detailed differences between X and Y",
                "variants": [
                    "X vs Y analysis",
                    "Differences between X and Y",
                    "X Y comparison",
                ],
                "expected_min": 0.8,
            },
        ]

        for scenario in scenarios:
            score = DSPyLlamaIndexRetriever._estimate_quality_improvement(
                scenario["original"], scenario["refined"], scenario["variants"]
            )

            assert score >= scenario["expected_min"]
            if "expected_max" in scenario:
                assert score <= scenario["expected_max"]
            assert 0 <= score <= 1.0  # Valid score range


@pytest.mark.integration
class TestDSPyConfigurationEdgeCases:
    """Integration tests for DSPy configuration edge cases and boundary conditions."""

    def test_configuration_with_edge_case_parameters(self):
        """Test DSPy configuration with edge case parameters."""
        edge_cases = [
            {"max_variants": 0},  # No variants
            {"max_variants": 1},  # Single variant
            {"max_variants": 10},  # Many variants
        ]

        mock_llm = Mock()

        for case in edge_cases:
            with patch("src.dspy_integration.DSPY_AVAILABLE", True):
                with patch("src.dspy_integration.dspy") as mock_dspy:
                    mock_dspy.configure = Mock()
                    mock_dspy.ChainOfThought = Mock()

                    retriever = DSPyLlamaIndexRetriever(llm=mock_llm, **case)

                    assert retriever.max_variants == case["max_variants"]

                    # Test optimization with edge case configuration
                    if case["max_variants"] == 0:
                        # Should skip variant generation entirely
                        with patch.object(retriever, "variant_generator"):
                            retriever.optimization_enabled = True
                            retriever.query_refiner = Mock(
                                return_value=Mock(refined_query="refined")
                            )

                            result = retriever.optimize_query(
                                "test", llm=mock_llm, enable_variants=True
                            )

                            # Should not call variant generator when max_variants is 0
                            assert result["variants"] == []

    def test_optimization_result_model_validation(self):
        """Test QueryOptimizationResult model validation integration."""
        # Test valid result creation
        valid_result = QueryOptimizationResult(
            original="test query",
            refined="enhanced test query for better retrieval",
            variants=["variant 1", "variant 2"],
            optimization_time=0.123,
            quality_score=0.85,
        )

        assert valid_result.original == "test query"
        assert valid_result.refined == "enhanced test query for better retrieval"
        assert len(valid_result.variants) == 2
        assert valid_result.optimization_time == 0.123
        assert valid_result.quality_score == 0.85

        # Test with minimal required fields
        minimal_result = QueryOptimizationResult(
            original="minimal query",
            refined="minimal refined query",
        )

        assert minimal_result.variants == []
        assert minimal_result.optimization_time == 0.0
        assert minimal_result.quality_score == 0.0

    def test_fallback_optimization_consistency(self):
        """Test fallback optimization consistency across multiple calls."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            query = "consistent test query"

            # Run multiple optimizations
            results = []
            for _ in range(3):
                result = DSPyLlamaIndexRetriever.optimize_query(query)
                results.append(result)

            # Results should be consistent (deterministic fallback)
            first_result = results[0]
            for result in results[1:]:
                assert result["original"] == first_result["original"]
                assert result["refined"] == first_result["refined"]
                assert result["variants"] == first_result["variants"]
                assert result["quality_score"] == first_result["quality_score"]
                assert result["optimized"] == first_result["optimized"]
