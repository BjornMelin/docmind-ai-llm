"""Intentional failing test to highlight missing ingestion pipeline."""

import pytest


@pytest.mark.integration
def test_ingestion_pipeline_pending() -> None:
    """Fail until the new ingestion pipeline is implemented."""
    pytest.fail(
        "Ingestion pipeline removed during refactor. Implement Phase 2 "
        "before removing this test."
    )
