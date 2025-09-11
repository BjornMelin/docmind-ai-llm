from __future__ import annotations

import pytest

"""Top-level pytest configuration for shared fixtures.

Loads the shared fixtures module so fixtures like `supervisor_stream_shim` are
available across test tiers without explicit imports in each test file.
"""

pytest_plugins = [
    "tests.shared_fixtures",
]


@pytest.fixture
def integration_settings():
    """Provide standardized integration settings with lightweight embedding.

    Ensures `embedding_dimension == 384` and model name contains "MiniLM" as
    asserted by infrastructure validation tests.
    """
    # Lazy import to avoid early import-time path issues during PyTest startup
    from tests.fixtures.test_settings import create_integration_settings

    s = create_integration_settings()
    # Ensure expected lightweight model naming for assertions
    s.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return s


@pytest.fixture
def system_settings():
    """Provide system-tier settings approximating production configuration."""
    from tests.fixtures.test_settings import create_system_settings

    return create_system_settings()


@pytest.fixture
def lightweight_embedding_model():
    """Provide a lightweight embedding model stub for integration tests.

    Mimics all-MiniLM-L6-v2 behavior by exposing an `encode` method that returns
    (N, 384) float32 embeddings.
    """
    import numpy as np

    class _MiniLM:
        def encode(self, items: list[str]):
            return np.zeros((len(items), 384), dtype=np.float32)

    return _MiniLM()
