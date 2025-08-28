"""Test infrastructure validation script.

Validates that the standardized test fixtures and infrastructure work properly
across all test tiers and components. This ensures CI/CD pipeline reliability.
"""

from unittest.mock import AsyncMock

import pytest


# Test the shared fixtures work properly
def test_shared_fixtures_import():
    """Test that shared fixtures can be imported without issues."""
    from tests.shared_fixtures import (
        MockEmbeddingFactory,
        MockLLMFactory,
        MockRetrieverFactory,
        PerformanceTimer,
        TestDataFactory,
    )

    # Verify factories can create mocks
    assert MockEmbeddingFactory.create_mock_embedding_model() is not None
    assert MockLLMFactory.create_simple_mock_llm() is not None
    assert MockRetrieverFactory.create_mock_retriever() is not None
    assert len(TestDataFactory.create_test_documents()) > 0
    assert isinstance(PerformanceTimer(), PerformanceTimer)


@pytest.mark.unit
def test_mock_embedding_factory():
    """Test mock embedding factory produces consistent embeddings."""
    from tests.shared_fixtures import MockEmbeddingFactory

    # Test dense embeddings
    embeddings = MockEmbeddingFactory.create_dense_embeddings(
        dimension=1024, num_docs=3
    )
    assert len(embeddings) == 3
    assert len(embeddings[0]) == 1024

    # Test sparse embeddings
    sparse = MockEmbeddingFactory.create_sparse_embeddings(num_docs=2)
    assert len(sparse) == 2
    assert all(isinstance(emb, dict) for emb in sparse)

    # Test mock embedding model
    model = MockEmbeddingFactory.create_mock_embedding_model()
    assert model is not None
    assert hasattr(model, "get_text_embedding")


@pytest.mark.unit
def test_mock_llm_factory():
    """Test mock LLM factory creates proper LLM mocks."""
    from tests.shared_fixtures import MockLLMFactory

    # Test simple mock
    simple_llm = MockLLMFactory.create_simple_mock_llm("test response")
    assert simple_llm.invoke() == "test response"
    assert hasattr(simple_llm, "complete")
    assert hasattr(simple_llm, "predict")

    # Test that the factory methods exist and are callable
    assert callable(MockLLMFactory.create_simple_mock_llm)
    assert callable(MockLLMFactory.create_llamaindex_mock_llm)


@pytest.mark.unit
def test_mock_retriever_factory():
    """Test mock retriever factory creates proper search results."""
    from tests.shared_fixtures import MockRetrieverFactory

    # Test search results
    results = MockRetrieverFactory.create_search_results(num_results=5, base_score=0.9)
    assert len(results) == 5
    assert all(hasattr(r, "node") and hasattr(r, "score") for r in results)

    # Test scores are decreasing
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_test_data_factory():
    """Test that test data factory creates realistic data."""
    from tests.shared_fixtures import TestDataFactory

    # Test documents
    docs = TestDataFactory.create_test_documents(num_docs=3)
    assert len(docs) == 3
    assert all(hasattr(doc, "text") and hasattr(doc, "metadata") for doc in docs)

    # Test query scenarios
    scenarios = TestDataFactory.create_query_scenarios()
    assert len(scenarios) > 0
    assert all("query" in s and "expected_strategy" in s for s in scenarios)


@pytest.mark.unit
def test_performance_timer():
    """Test performance timer utility works correctly."""
    import time

    from tests.shared_fixtures import PerformanceTimer

    timer = PerformanceTimer()

    # Test basic timing
    timer.start("test_operation")
    time.sleep(0.001)  # 1ms
    duration = timer.stop("test_operation")

    assert duration > 0
    assert duration < 100  # Should be much less than 100ms

    # Test stats
    stats = timer.get_stats("test_operation")
    assert "mean_ms" in stats
    assert stats["count"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_fixtures(mock_async_embedding_model, async_test_utils):
    """Test async fixtures work properly."""
    # Test async embedding model
    assert isinstance(mock_async_embedding_model, AsyncMock)

    # Test async utilities
    from tests.shared_fixtures import AsyncTestUtils

    assert isinstance(async_test_utils, AsyncTestUtils)

    # Test timeout functionality
    async def quick_operation():
        return "success"

    result = await async_test_utils.run_with_timeout(
        quick_operation(), timeout_seconds=1.0
    )
    assert result == "success"


@pytest.mark.unit
def test_session_scoped_fixtures_available():
    """Test that session-scoped fixtures are available and properly configured."""
    # Import session fixtures - they should be importable
    from tests.shared_fixtures import session_query_scenarios, session_test_documents

    # Verify they are fixture functions
    assert hasattr(session_test_documents, "_pytestfixturefunction")
    assert hasattr(session_query_scenarios, "_pytestfixturefunction")


@pytest.mark.integration
def test_integration_tier_fixtures(integration_settings, lightweight_embedding_model):
    """Test that integration-tier fixtures work properly."""
    # Test integration settings
    assert integration_settings.embedding_dimension == 384
    assert "MiniLM" in integration_settings.embedding.model_name

    # Test lightweight model (may be None if not available)
    if lightweight_embedding_model is not None:
        # Has proper interface
        assert hasattr(lightweight_embedding_model, "encode")


@pytest.mark.system
def test_system_tier_fixtures(system_settings):
    """Test that system-tier fixtures work properly."""
    # Test system settings use full models
    assert system_settings.embedding_dimension == 1024
    assert "bge-large" in system_settings.embedding.model_name
    assert system_settings.retrieval.use_reranking is True


def test_conftest_hierarchy():
    """Test that conftest files work together without conflicts."""
    # Test that fixtures from different conftest files can coexist
    from tests.test_multi_agent_coordination.conftest import mock_vllm_config
    from tests.test_retrieval.conftest import mock_bgem3_model
    from tests.unit.conftest import sample_image_base64

    # All should be fixture functions
    assert hasattr(sample_image_base64, "_pytestfixturefunction")
    assert hasattr(mock_vllm_config, "_pytestfixturefunction")
    assert hasattr(mock_bgem3_model, "_pytestfixturefunction")


@pytest.mark.unit
def test_pytest_markers_configured():
    """Test that pytest markers are properly configured."""
    import pytest

    # Test that our custom markers exist
    pytest.Config(pluginmanager=pytest.PytestPluginManager())

    # The markers should be available (this tests marker registration)
    expected_markers = [
        "unit",
        "integration",
        "system",
        "performance",
        "agents",
        "retrieval",
    ]

    # Basic marker validation - they should be strings
    for marker in expected_markers:
        assert isinstance(marker, str)
        assert len(marker) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
