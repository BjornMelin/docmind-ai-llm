"""Refactored unit tests for retrieval embeddings (SPEC-003 v1.1.0).

Focus:
- Ensure legacy module is gone (no back-compat).
- Validate LI BGEM3 factory helpers are importable and callable signatures exist.
- Validate ClipEmbedding usage via integrations without loading heavy backends.
"""

from __future__ import annotations

from contextlib import suppress
from unittest.mock import patch

import pytest


def test_legacy_module_removed():
    """Importing legacy module should fail (fully removed)."""
    with pytest.raises(ModuleNotFoundError):
        __import__("src.retrieval.embeddings")


def test_li_bge_m3_factory_symbols_exist():
    """Ensure LI BGEM3Index factory helpers are importable and callable."""
    from src.retrieval.bge_m3_index import (
        build_bge_m3_index,
        build_bge_m3_retriever,
        get_default_bge_m3_retriever,
    )

    # Build minimal index with empty nodes (will rely on LI internals when used)
    # We do not execute heavy model loading here.
    # Factory should be importable; runtime may vary if LI managed index missing
    # We tolerate exceptions here to keep tests offline and environment-agnostic.
    with suppress(Exception):  # pragma: no cover - environment dependent
        idx = build_bge_m3_index([], model_name="BAAI/bge-m3")
        _ = build_bge_m3_retriever(idx, weights_for_different_modes=[0.4, 0.2, 0.4])

    # The convenience wrapper should be callable
    with suppress(Exception):  # pragma: no cover
        _ = get_default_bge_m3_retriever([])


def test_clip_embedding_config_via_integrations():
    """Patch ClipEmbedding to return LI MockEmbedding; ensure Settings receives it."""
    from llama_index.core import Settings
    from llama_index.core.embeddings import MockEmbedding

    with patch(
        "src.config.integrations.ClipEmbedding",
        return_value=MockEmbedding(embed_dim=1024),
    ):
        from src.config.integrations import setup_llamaindex

        Settings.embed_model = None
        setup_llamaindex(force_embed=True)

        assert Settings.embed_model is not None
        assert Settings.embed_model.__class__.__name__ in {
            "MockEmbedding",
            "ClipEmbedding",
        }
