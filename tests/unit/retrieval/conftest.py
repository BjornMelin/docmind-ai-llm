"""Shared utilities for retrieval unit tests."""

from __future__ import annotations

import pytest
from llama_index.core.query_engine import RouterQueryEngine

from src.config.settings import DocMindSettings


@pytest.fixture
def router_settings() -> DocMindSettings:
    """Return canonical settings with optional router tools disabled."""
    settings = DocMindSettings()
    settings.retrieval.enable_image_retrieval = False
    settings.retrieval.enable_server_hybrid = False
    settings.retrieval.enable_keyword_tool = False
    settings.retrieval.use_reranking = False
    return settings


def get_router_tool_names(router: RouterQueryEngine) -> list[str]:
    """Extract names from LlamaIndex's native router metadata."""
    return [metadata.name for metadata in router._metadatas]
