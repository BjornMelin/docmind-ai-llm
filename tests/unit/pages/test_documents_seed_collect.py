"""Seed selection tests for Documents page utilities.

Verifies deterministic seed derivation and capping to 32 results.
"""

from __future__ import annotations

from src.retrieval.graph_config import get_export_seed_ids


def test_collect_seed_ids_deterministic() -> None:
    # With no indices available, fallback is deterministic numeric seeds
    seeds = get_export_seed_ids(None, None, cap=32)
    assert len(seeds) == 32
    assert seeds[0] == "0"
    assert seeds[-1] == "31"
