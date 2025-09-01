"""Property-style invariants for similarity and embeddings.

These tests are lightweight and deterministic (seeded) to avoid flakiness.
"""

import numpy as np
import pytest


@pytest.mark.unit
class TestVectorSimilarityInvariants:
    """Basic invariants for cosine similarity and Euclidean distance."""

    def test_cosine_symmetry_and_bounds(self):
        """Cosine similarity is symmetric and within [-1, 1]."""
        np.random.seed(2025)
        v1 = np.random.randn(64)
        v2 = np.random.randn(64)

        dot12 = float(np.dot(v1, v2))
        dot21 = float(np.dot(v2, v1))
        assert abs(dot12 - dot21) < 1e-12

        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 > 0 and n2 > 0:
            cos = dot12 / (n1 * n2)
            assert -1.0 - 1e-12 <= cos <= 1.0 + 1e-12

    def test_distance_non_negative(self):
        """Euclidean distance is always non-negative."""
        np.random.seed(2026)
        v1 = np.random.randn(64)
        v2 = np.random.randn(64)
        d = float(np.linalg.norm(v1 - v2))
        assert d >= 0.0


@pytest.mark.unit
class TestEmbeddingDimensionInvariant:
    """Embedding dimension invariant using LlamaIndex MockEmbedding."""

    def test_mock_embedding_dimension_1024(self):
        """MockEmbedding returns vectors with the requested dimension."""
        from llama_index.core.embeddings import MockEmbedding

        emb = MockEmbedding(embed_dim=1024)
        vec = emb.get_text_embedding("dimension check")
        assert isinstance(vec, list)
        assert len(vec) == 1024
