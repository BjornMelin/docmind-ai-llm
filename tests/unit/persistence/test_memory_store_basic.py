"""Unit tests for DocMindSqliteStore (LangGraph BaseStore adapter)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.persistence.memory_store import DocMindSqliteStore

pytestmark = pytest.mark.unit


def test_put_get_search_delete_roundtrip(tmp_path: Path) -> None:
    cfg = SimpleNamespace(data_dir=tmp_path)
    store = DocMindSqliteStore(Path("chat.db"), cfg=cfg)

    ns = ("memories", "u1", "t1")
    store.put(ns, "k1", {"content": "hello", "kind": "fact"}, index=False)

    item = store.get(ns, "k1")
    assert item is not None
    assert item.value["content"] == "hello"

    # Prefix search includes children.
    results = store.search(("memories", "u1"), query=None, limit=10)
    assert any(r.key == "k1" for r in results)

    # Filter search (Python-side) still works.
    filtered = store.search(("memories", "u1"), query=None, filter={"kind": "fact"})
    assert len(filtered) == 1
    assert filtered[0].key == "k1"

    store.delete(ns, "k1")
    assert store.get(ns, "k1") is None

    store.close()


def test_ttl_expiry_and_filter_operators(monkeypatch, tmp_path: Path) -> None:
    cfg = SimpleNamespace(data_dir=tmp_path)
    store = DocMindSqliteStore(Path("chat.db"), cfg=cfg)

    ns = ("memories", "u1", "t1")
    monkeypatch.setattr("src.persistence.memory_store.now_ms", lambda: 1_000)
    store.put(ns, "k1", {"n": 2, "tags": ["x", "y"]}, ttl=0.001, index=False)

    # Before expiry (ttl minutes -> ms)
    monkeypatch.setattr("src.persistence.memory_store.now_ms", lambda: 1_010)
    assert store.get(ns, "k1") is not None

    # After expiry
    monkeypatch.setattr("src.persistence.memory_store.now_ms", lambda: 2_000)
    assert store.get(ns, "k1") is None

    # Recreate without ttl and exercise filter ops
    monkeypatch.setattr("src.persistence.memory_store.now_ms", lambda: 3_000)
    store.put(ns, "k2", {"n": 2, "tags": ["x", "y"]}, index=False)

    gt = store.search(ns, query=None, filter={"n": {"$gt": 1}})
    assert [r.key for r in gt] == ["k2"]
    tag_match = store.search(ns, query=None, filter={"tags": {"$eq": "x"}})
    assert [r.key for r in tag_match] == ["k2"]

    with pytest.raises(ValueError, match="Unsupported filter operator"):
        store.search(ns, query=None, filter={"n": {"$nope": 1}})

    store.close()


def test_list_namespaces_prefix_suffix_and_max_depth(tmp_path: Path) -> None:
    cfg = SimpleNamespace(data_dir=tmp_path)
    store = DocMindSqliteStore(Path("chat.db"), cfg=cfg)

    store.put(("memories", "u1", "t1"), "k1", {"x": 1}, index=False)
    store.put(("memories", "u1", "t2"), "k2", {"x": 1}, index=False)
    store.put(("other", "u2", "t3"), "k3", {"x": 1}, index=False)

    ns = store.list_namespaces(prefix=("memories", "u1"))
    assert ("memories", "u1", "t1") in ns
    assert ("memories", "u1", "t2") in ns
    assert all(x[0] == "memories" for x in ns if x)

    suf = store.list_namespaces(suffix=("t3",))
    assert suf == [("other", "u2", "t3")]

    capped = store.list_namespaces(prefix=("memories",), max_depth=2)
    assert ("memories", "u1") in capped

    store.close()
