"""Seed selection tests for Documents page utilities.

Verifies deterministic seed derivation and capping to 32 results.
"""

from __future__ import annotations

import importlib

docs_mod = importlib.import_module("src.pages.02_documents")


def test_collect_seed_ids_deterministic() -> None:
    seeds = docs_mod._collect_seed_ids(object(), cap=32)
    assert len(seeds) == 32
    assert seeds[0] == "0"
    assert seeds[-1] == "31"
