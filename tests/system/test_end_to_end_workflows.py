"""System E2E (skipped) for multimodal ingest→index→retrieve→rerank.

Requires a local Qdrant and optional GPU for reranking performance.
"""

import pytest


@pytest.mark.skip(reason="requires local services and sample data")
def test_e2e_multimodal_workflow():
    """Placeholder E2E test for full workflow."""
    assert True
