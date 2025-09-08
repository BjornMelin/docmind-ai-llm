"""Refactored unit tests for retrieval embeddings (SPEC-003 v1.1.0).

Focus:
- Ensure legacy module is gone (no back-compat).
- Validate ClipEmbedding usage via integrations without loading heavy backends.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_legacy_module_removed():
    """Importing legacy module should fail (fully removed)."""
    with pytest.raises(ModuleNotFoundError):
        __import__("src.retrieval.embeddings")


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
