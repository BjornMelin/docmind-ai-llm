"""Tests for get_postprocessors wiring in reranking module."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_get_postprocessors_disabled_returns_none():
    rr = importlib.import_module("src.retrieval.reranking")
    assert rr.get_postprocessors("vector", use_reranking=False) is None
    assert rr.get_postprocessors("kg", use_reranking=False) is None


@pytest.mark.unit
def test_get_postprocessors_vector_and_kg():
    rr = importlib.import_module("src.retrieval.reranking")

    vec_pp = rr.get_postprocessors("vector", use_reranking=True)
    assert isinstance(vec_pp, list) and vec_pp, "expected reranker list for vector"
    assert any(type(p).__name__ == "MultimodalReranker" for p in vec_pp)

    # KG uses text reranker; pass explicit top_n to assert passthrough
    kg_pp = rr.get_postprocessors("kg", use_reranking=True, top_n=7)
    assert isinstance(kg_pp, list) and kg_pp
    assert hasattr(kg_pp[0], "postprocess_nodes")
    assert getattr(kg_pp[0], "top_n", None) == 7


@pytest.mark.unit
def test_get_postprocessors_unknown_mode_returns_none():
    rr = importlib.import_module("src.retrieval.reranking")
    assert rr.get_postprocessors("unknown-mode", use_reranking=True) is None

