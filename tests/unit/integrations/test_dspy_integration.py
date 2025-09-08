"""Comprehensive unit tests for DSPy Integration (ADR-018 compliant).

Moved from tests/test_multi_agent_coordination/test_dspy_integration.py to
tests/unit/ to align with unit-tier scope and simplify the suite structure.
"""

# File content preserved from original location
from unittest.mock import Mock, patch

import pytest

from src.dspy_integration import (
    DSPyLlamaIndexRetriever,
    QueryOptimizationResult,
    get_dspy_retriever,
    is_dspy_available,
)


class TestDSPyAvailability:
    """Detects whether DSPy is available or not."""

    def test_dspy_availability_true(self):
        """Returns True when the availability flag is set."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            assert is_dspy_available() is True

    def test_dspy_availability_false(self):
        """Returns False when the availability flag is unset."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            assert is_dspy_available() is False


class TestQueryOptimizationResult:
    """Validation for the optimization result data model."""

    def test_optimization_result_creation(self):
        """Creates a full result and verifies fields are retained."""
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
        """Applies defaults when variants/time/score are not provided."""
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
    """DSPy retriever initialization, wrapping and optimization behavior."""

    def test_initialization_with_dspy_available(self, mock_llm: Mock):
        """Enables optimization and wires ChainOfThought when available."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy", create=True) as mock_dspy,
        ):
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()

            retriever = DSPyLlamaIndexRetriever(llm=mock_llm, max_variants=3)
            assert retriever.llm == mock_llm
            assert retriever.max_variants == 3
            assert retriever.optimization_enabled is True
            mock_dspy.configure.assert_called_once()

    def test_initialization_with_dspy_unavailable(self, mock_llm: Mock):
        """Disables optimization when DSPy is unavailable."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            assert retriever.llm == mock_llm
            assert retriever.optimization_enabled is False

    def test_initialization_without_llm(self):
        """Allows construction without an LLM and disables optimization."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            retriever = DSPyLlamaIndexRetriever()
            assert retriever.llm is None
            assert retriever.optimization_enabled is False

    def test_llm_wrapper_creation(self, mock_llm: Mock):
        """Wraps an LLM to provide a simple callable for DSPy."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            retriever = DSPyLlamaIndexRetriever()

            mock_response = Mock()
            mock_response.text = "Test response"
            mock_llm.complete.return_value = mock_response

            wrapper = retriever._wrap_llm_for_dspy(mock_llm)
            result = wrapper("Test prompt")
            assert result == str(mock_response)

    def test_llm_wrapper_error_handling(self, mock_llm: Mock):
        """Returns empty string if the wrapped LLM raises an error."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", True):
            retriever = DSPyLlamaIndexRetriever()
            # Wrapper catches RuntimeError/ValueError/AttributeError
            mock_llm.complete.side_effect = RuntimeError("LLM error")
            wrapper = retriever._wrap_llm_for_dspy(mock_llm)
            assert wrapper("Test prompt") == ""

    def test_optimize_query_with_dspy_available(self, mock_llm: Mock):
        """Produces a refined query and variants when DSPy is available."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy", create=True) as mock_dspy,
        ):
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()

            mock_refined_result = Mock()
            mock_refined_result.refined_query = "Optimized query about machine learning"
            mock_variant_result = Mock()
            mock_variant_result.variant1 = "What is machine learning?"
            mock_variant_result.variant2 = "Explain ML concepts"

            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock(return_value=mock_refined_result)
            retriever.variant_generator = Mock(return_value=mock_variant_result)

            type(retriever)._instance = retriever
            result = type(retriever).optimize_query(
                "ML", llm=mock_llm, enable_variants=True
            )
            assert result["original"] == "ML"
            assert result["refined"] == "Optimized query about machine learning"
            assert len(result["variants"]) == 2

    def test_optimize_query_with_dspy_unavailable(self):
        """Falls back to heuristic optimization when DSPy is unavailable."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            result = DSPyLlamaIndexRetriever.optimize_query("AI")
            assert result["original"] == "AI"
            assert result["optimized"] is False

    def test_optimize_query_short_query_enhancement(self):
        """Expands short queries to more descriptive text in fallback mode."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            result = DSPyLlamaIndexRetriever.optimize_query("ML")
            # Fallback path says "Find information about {query}"
            assert result["refined"] == "Find information about ML"

    def test_optimize_query_comparison_enhancement(self):
        """Suggests comparison phrasing when input implies compare tasks."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            result = DSPyLlamaIndexRetriever.optimize_query("compare AI vs ML")
            assert "difference between" in result["variants"][0]

    def test_optimize_query_with_refinement_failure(self, mock_llm: Mock):
        """Falls back to original query if the refiner fails at runtime."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy", create=True) as mock_dspy,
        ):
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()

            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock(
                side_effect=RuntimeError("Refinement failed")
            )
            retriever.variant_generator = Mock()
            type(retriever)._instance = retriever
            result = type(retriever).optimize_query("test query", llm=mock_llm)
            assert result["refined"] == "test query"

    def test_optimize_query_with_variant_failure(self, mock_llm: Mock):
        """Returns empty variants list if variant generation fails."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy", create=True) as mock_dspy,
        ):
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()
            mock_refined_result = Mock()
            mock_refined_result.refined_query = "Refined query"
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock(return_value=mock_refined_result)
            # Variant failures are caught for common runtime errors
            retriever.variant_generator = Mock(side_effect=RuntimeError("fail"))

            type(retriever)._instance = retriever
            result = type(retriever).optimize_query(
                "test query", llm=mock_llm, enable_variants=True
            )
            assert result["variants"] == []

    def test_optimize_query_variants_disabled(self, mock_llm: Mock):
        """Skips variant generation when explicitly disabled."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy", create=True) as mock_dspy,
        ):
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()
            mock_refined_result = Mock()
            mock_refined_result.refined_query = "Refined query"
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock(return_value=mock_refined_result)
            retriever.variant_generator = Mock()

            type(retriever)._instance = retriever
            result = type(retriever).optimize_query(
                "test query", llm=mock_llm, enable_variants=False
            )
            assert result["variants"] == []

    def test_optimize_query_max_variants_limit(self, mock_llm: Mock):
        """Respects the max_variants cap when generating variants."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy", create=True) as mock_dspy,
        ):
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()
            mock_refined_result = Mock()
            mock_refined_result.refined_query = "Refined query"
            mock_variant_result = Mock()
            mock_variant_result.variant1 = "Variant 1"
            mock_variant_result.variant2 = "Variant 2"

            retriever = DSPyLlamaIndexRetriever(llm=mock_llm, max_variants=1)
            retriever.optimization_enabled = True
            retriever.query_refiner = Mock(return_value=mock_refined_result)
            retriever.variant_generator = Mock(return_value=mock_variant_result)

            type(retriever)._instance = retriever
            result = type(retriever).optimize_query(
                "test query", llm=mock_llm, enable_variants=True
            )
            assert len(result["variants"]) == 1

    def test_optimize_retrieval_pipeline_with_dspy_available(self, mock_llm: Mock):
        """Compiles a bootstrapped refiner and updates the retriever."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy", create=True) as mock_dspy,
            patch(
                "src.dspy_integration.BootstrapFewShot", create=True
            ) as mock_bootstrap,
        ):
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()
            mock_dspy.Example = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
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
            assert result == mock_retriever
            mock_bootstrap.assert_called_once_with(max_bootstrapped_demos=3)
            mock_optimizer.compile.assert_called_once()
            assert retriever.query_refiner == mock_optimized_refiner

    def test_optimize_retrieval_pipeline_with_dspy_unavailable(self, mock_llm: Mock):
        """Returns the original retriever when DSPy is unavailable."""
        with patch("src.dspy_integration.DSPY_AVAILABLE", False):
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            mock_retriever = Mock()
            queries = ["What is AI?"]
            assert (
                retriever.optimize_retrieval_pipeline(mock_retriever, queries)
                == mock_retriever
            )

    def test_optimize_retrieval_pipeline_error_handling(self, mock_llm: Mock):
        """Keeps original retriever if bootstrapping examples fail."""
        with (
            patch("src.dspy_integration.DSPY_AVAILABLE", True),
            patch("src.dspy_integration.dspy", create=True) as mock_dspy,
            patch("src.dspy_integration.BootstrapFewShot", create=True),
        ):
            mock_dspy.configure = Mock()
            mock_dspy.ChainOfThought = Mock()
            # Use RuntimeError to match production exception handling
            mock_dspy.Example = Mock(
                side_effect=RuntimeError("Example creation failed")
            )
            retriever = DSPyLlamaIndexRetriever(llm=mock_llm)
            retriever.optimization_enabled = True
            mock_retriever = Mock()
            queries = ["What is AI?"]
            assert (
                retriever.optimize_retrieval_pipeline(mock_retriever, queries)
                == mock_retriever
            )


class TestGlobalInstance:
    """Global instance creation/reuse behavior for the DSPy retriever."""

    def test_get_dspy_retriever_creates_instance(self, mock_llm: Mock):
        """Creates a new instance when none exists and returns it."""
        with patch("src.dspy_integration._DSPY_CACHE", {"inst": None}):
            retriever = get_dspy_retriever(llm=mock_llm)
            assert isinstance(retriever, DSPyLlamaIndexRetriever)
            assert retriever.llm == mock_llm

    def test_get_dspy_retriever_reuses_instance(self, mock_llm: Mock):
        """Returns the same instance on subsequent calls."""
        first_retriever = get_dspy_retriever(llm=mock_llm)
        second_retriever = get_dspy_retriever()
        assert first_retriever is second_retriever

    def test_get_dspy_retriever_without_llm(self):
        """Creates an instance without an explicit LLM argument."""
        with patch("src.dspy_integration._DSPY_CACHE", {"inst": None}):
            retriever = get_dspy_retriever()
            assert isinstance(retriever, DSPyLlamaIndexRetriever)


@pytest.fixture(autouse=True)
def _reset_singleton_instance():
    """Ensure class-level singleton does not leak state across tests."""
    from src.dspy_integration import DSPyLlamaIndexRetriever

    # Reset before each test
    if hasattr(DSPyLlamaIndexRetriever, "_instance"):
        DSPyLlamaIndexRetriever._instance = None
    yield
    # Reset after each test as well for safety
    if hasattr(DSPyLlamaIndexRetriever, "_instance"):
        DSPyLlamaIndexRetriever._instance = None
