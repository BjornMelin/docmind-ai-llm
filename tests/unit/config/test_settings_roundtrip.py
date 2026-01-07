"""Settings round-trip tests for retrieval flags.

Ensures the nested retrieval.enable_server_hybrid flag is present and mutable.
"""

from __future__ import annotations

from src.config.settings import DocMindSettings


def test_retrieval_enable_server_hybrid_roundtrip() -> None:
    """Verify retrieval.enable_server_hybrid can be set and read back."""
    cfg = DocMindSettings()
    assert hasattr(cfg.retrieval, "enable_server_hybrid")
    # Default should be False
    assert cfg.retrieval.enable_server_hybrid is False
    # Set and read back
    cfg.retrieval.enable_server_hybrid = True
    assert cfg.retrieval.enable_server_hybrid is True


def test_graphrag_enablement_requires_both_flags() -> None:
    """GraphRAG enablement requires both the global and nested flags."""
    cfg = DocMindSettings()
    cfg.enable_graphrag = False
    cfg.graphrag_cfg.enabled = True
    assert cfg.is_graphrag_enabled() is False

    cfg.enable_graphrag = True
    cfg.graphrag_cfg.enabled = False
    assert cfg.is_graphrag_enabled() is False

    cfg.enable_graphrag = True
    cfg.graphrag_cfg.enabled = True
    assert cfg.is_graphrag_enabled() is True
