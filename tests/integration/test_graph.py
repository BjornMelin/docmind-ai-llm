"""GraphRAG public-surface integration tests."""

from __future__ import annotations

import pytest

from src.retrieval import graph_config


@pytest.mark.integration
class TestGraphImports:
    """Test the required GraphRAG public surface."""

    def test_property_graph_entrypoints_are_callable(self) -> None:
        """Required graph construction and export entrypoints are callable."""
        assert callable(graph_config.build_graph_query_engine)
        assert callable(graph_config.export_graph_jsonl)
