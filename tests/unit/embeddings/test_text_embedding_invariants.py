"""Text embedding invariants tests for normalization and determinism.

Uses a light stub to avoid heavy backend.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_l2_normalize_invariants():
    from src.models.embeddings import _l2_normalize

    arr = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    norm = _l2_normalize(arr)
    assert np.isclose(np.linalg.norm(norm[0]), 1.0)
    # Zero row remains zero
    assert np.allclose(norm[1], np.zeros(2))
