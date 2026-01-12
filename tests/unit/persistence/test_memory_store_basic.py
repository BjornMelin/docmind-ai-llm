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
