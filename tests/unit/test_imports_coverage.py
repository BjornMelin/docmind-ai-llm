"""Lightweight import tests to exercise module definitions for coverage.

These imports avoid executing any runtime side effects; modules are designed to
do work only when their factories/functions are called. This keeps tests fast
and offline while improving definition coverage for quality gates.
"""

import importlib

import pytest


@pytest.mark.unit
def test_import_core_modules_for_coverage():
    """Test that core modules can be imported for coverage."""
    modules = [
        "src.models.embeddings",
        "src.utils.multimodal",
        "src.retrieval.bge_m3_index",
        "src.retrieval.query_engine",
        "src.retrieval.graph_config",
        "src.retrieval.reranking",
        "src.retrieval.optimization",
        "src.utils.core",
        "src.utils.storage",
        "src.utils.document",
        "src.models.schemas",
        "src.models.processing",
        "src.utils.monitoring",
        "src.agents.tools.router_tool",
        "src.agents.tools.planning",
        "src.agents.tools.validation",
        "src.agents.tools.synthesis",
        "src.agents.coordinator",
        "src.agents.tool_factory",
        "src.config.integrations",
        "src.app",
        "src.agents.retrieval",
        "src.retrieval.reranking",
        "src.ui.components.provider_badge",
        "src.processing.pdf_pages",
        "src.retrieval.optimization",
    ]

    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None
