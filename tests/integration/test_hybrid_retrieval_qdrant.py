"""Integration test (skipped) for Qdrant server-side fusion toggles."""

import os

import pytest


@pytest.mark.skip(reason="requires local Qdrant instance")
def test_server_side_rrf_and_dbsf_toggle():
    """Test server-side RRF and DBSF toggle functionality."""
    from src.retrieval.query_engine import (  # type: ignore
        ServerHybridRetriever,
        _HybridParams,
    )

    os.environ["DOCMIND_FUSION"] = "rrf"
    retr = ServerHybridRetriever(_HybridParams(collection="docmind_docs"))
    # Smoke: ensure retrieve method exists
    assert callable(retr.retrieve)
    os.environ["DOCMIND_FUSION"] = "dbsf"
