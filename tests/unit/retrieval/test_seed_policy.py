"""Unit tests for seed policy helper (retriever-preferred + fallback)."""

from __future__ import annotations

from types import SimpleNamespace

from src.retrieval.graph_config import get_export_seed_ids


class _NWS:
    def __init__(self, nid: str) -> None:
        self.node = SimpleNamespace(id_=nid)


class _Retriever:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def retrieve(self, _q: str):  # type: ignore[no-untyped-def]
        return [_NWS(x) for x in self._ids]


class _PgIdx:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def as_retriever(self, include_text=False, path_depth=1, similarity_top_k=32):  # type: ignore[no-untyped-def]
        del include_text, path_depth, similarity_top_k
        return _Retriever(self._ids)


class _VecIdx:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def as_retriever(self, similarity_top_k=32):  # type: ignore[no-untyped-def]
        del similarity_top_k
        return _Retriever(self._ids)


def test_seeds_prefer_pg_index_retriever() -> None:
    pg = _PgIdx(["1", "2", "3"])
    vec = _VecIdx(["9", "8", "7"])
    out = get_export_seed_ids(pg, vec, cap=2)
    assert out == ["1", "2"]


def test_seeds_fallback_to_vector() -> None:
    pg = None
    vec = _VecIdx(["4", "5"])
    out = get_export_seed_ids(pg, vec, cap=2)
    assert out == ["4", "5"]


def test_seeds_deterministic_fallback() -> None:
    pg = None
    vec = None
    out = get_export_seed_ids(pg, vec, cap=3)
    assert out == ["0", "1", "2"]
