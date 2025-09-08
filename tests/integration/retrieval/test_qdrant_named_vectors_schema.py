"""Optional smoke test for named vector schema in Qdrant (skipped if not available)."""

from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.getenv("DOCMIND_QDRANT_SCHEMA_SMOKE", "0") != "1",
    reason="Schema smoke disabled (set DOCMIND_QDRANT_SCHEMA_SMOKE=1 to enable)",
)
def test_named_vectors_schema_present():
    from src.config.settings import DocMindSettings
    from src.utils.storage import get_collection_info

    collection = DocMindSettings().database.qdrant_collection
    info = get_collection_info(collection)
    if not info.get("exists"):
        pytest.skip("Collection not present; skipping schema check")

    vectors = info.get("vectors_config")
    sparse = info.get("sparse_vectors_config")

    # Depending on client, configs may be dict-like; guard for attribute access
    keys_dense = set(
        getattr(vectors, "__dict__", getattr(vectors, "keys", lambda: [])()).keys()
        if hasattr(vectors, "__dict__")
        else getattr(vectors, "keys", lambda: [])()
    )
    keys_sparse = set(
        getattr(sparse, "__dict__", getattr(sparse, "keys", lambda: [])()).keys()
        if hasattr(sparse, "__dict__")
        else getattr(sparse, "keys", lambda: [])()
    )

    # If dict-like, actual keys for named vectors should include our names
    if keys_dense and keys_sparse:
        assert "text-dense" in keys_dense
        assert "text-sparse" in keys_sparse
