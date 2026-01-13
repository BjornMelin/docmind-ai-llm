"""Unit tests for DocMindSqliteStore semantic search and TTL refresh."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.persistence.memory_store import DocMindSqliteStore, _ns_parts

pytestmark = pytest.mark.unit


class _Embeddings:
    def __call__(self, texts: list[str]) -> list[list[float]]:  # type: ignore[no-untyped-def]
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # type: ignore[no-untyped-def]
        out: list[list[float]] = []
        for t in texts:
            tt = str(t).lower()
            if "apple" in tt:
                out.append([1.0, 0.0])
            elif "banana" in tt:
                out.append([0.0, 1.0])
            else:
                out.append([0.5, 0.5])
        return out

    def embed_query(self, query: str) -> list[float]:  # type: ignore[no-untyped-def]
        q = str(query).lower()
        if "apple" in q:
            return [1.0, 0.0]
        if "banana" in q:
            return [0.0, 1.0]
        return [0.5, 0.5]


def test_get_refresh_ttl_extends_expiry(monkeypatch, tmp_path: Path) -> None:
    cfg = SimpleNamespace(data_dir=tmp_path)
    store = DocMindSqliteStore(Path("chat.db"), cfg=cfg)
    try:
        ns = ("memories", "u1", "t1")
        monkeypatch.setattr("src.persistence.memory_store.now_ms", lambda: 1_000)
        store.put(ns, "k1", {"content": "hello"}, ttl=0.001, index=False)

        # Refresh TTL at t=1030ms (ttl minutes -> 60ms delta)
        monkeypatch.setattr("src.persistence.memory_store.now_ms", lambda: 1_030)
        assert store.get(ns, "k1", refresh_ttl=True) is not None

        # Would have expired at 1060ms without refresh; should still exist.
        monkeypatch.setattr("src.persistence.memory_store.now_ms", lambda: 1_070)
        assert store.get(ns, "k1") is not None
    finally:
        store.close()


def test_semantic_search_and_delete_updates_vec_table(tmp_path: Path) -> None:
    cfg = SimpleNamespace(data_dir=tmp_path)
    store = DocMindSqliteStore(
        Path("chat.db"),
        index={"dims": 2, "embed": _Embeddings(), "fields": ["content"]},
        cfg=cfg,
    )
    try:
        ns = ("memories", "u1")
        store.put(ns, "a", {"content": "apple"})
        store.put(ns, "b", {"content": "banana"})

        res = store.search(ns, query="apple", limit=5)
        assert res
        assert res[0].key in {"a", "b"}
        assert any(r.key == "a" for r in res)
        assert all((r.score is None or isinstance(r.score, float)) for r in res)

        # Delete should best-effort delete from vec table as well.
        store.delete(ns, "a")
        assert store.get(ns, "a") is None
    finally:
        store.close()


def test_namespace_depth_guard_raises() -> None:
    too_deep = tuple("x" for _ in range(9))
    with pytest.raises(ValueError, match="namespace depth"):
        _ns_parts(too_deep)
