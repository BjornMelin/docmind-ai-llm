"""Unit tests for native LangGraph SQLite semantic memory search."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import InvalidNamespaceError

from src.config.settings import DocMindSettings
from src.persistence.checkpoint_identity import memory_namespace
from src.persistence.memory_store import close_memory_store, open_memory_store

pytestmark = pytest.mark.unit


class _Embeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = str(text).lower()
        if "apple" in lowered:
            return [1.0, 0.0]
        if "banana" in lowered:
            return [0.0, 1.0]
        return [0.5, 0.5]


def test_semantic_search_filter_and_vector_cascade(tmp_path: Path) -> None:
    cfg = cast(DocMindSettings, SimpleNamespace(data_dir=tmp_path))
    store = open_memory_store(
        Path("chat.db"),
        index={"dims": 2, "embed": _Embeddings(), "fields": ["content"]},
        cfg=cfg,
    )
    try:
        namespace = memory_namespace(user_id="u1", thread_id="t1")
        prefix = ".".join(namespace)
        store.put(namespace, "apple", {"content": "apple", "kind": "fact"})
        store.put(
            namespace,
            "banana",
            {"content": "banana", "kind": "preference"},
        )

        results = store.search(
            namespace,
            query="apple",
            filter={"kind": "fact"},
            limit=5,
        )
        assert [result.key for result in results] == ["apple"]
        assert results[0].score is not None
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM store_vectors WHERE prefix=? AND key=?;",
                (prefix, "apple"),
            ).fetchone()[0]
            == 1
        )

        store.delete(namespace, "apple")
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM store_vectors WHERE prefix=? AND key=?;",
                (prefix, "apple"),
            ).fetchone()[0]
            == 0
        )
    finally:
        close_memory_store(store)


def test_native_namespace_validation_rejects_periods(tmp_path: Path) -> None:
    cfg = cast(DocMindSettings, SimpleNamespace(data_dir=tmp_path))
    store = open_memory_store(Path("chat.db"), cfg=cfg)
    try:
        with pytest.raises(InvalidNamespaceError):
            store.put(("bad.namespace",), "key", {"content": "no"}, index=False)
    finally:
        close_memory_store(store)


def test_hashed_namespaces_isolate_prefixes_and_sql_wildcards(tmp_path: Path) -> None:
    cfg = cast(DocMindSettings, SimpleNamespace(data_dir=tmp_path))
    store = open_memory_store(Path("chat.db"), cfg=cfg)
    namespaces = (
        memory_namespace(user_id="u", thread_id="thread"),
        memory_namespace(user_id="u2", thread_id="thread"),
        memory_namespace(user_id="u%_", thread_id="thread%_"),
        memory_namespace(user_id="u", thread_id="thread-2"),
    )
    try:
        for index, namespace in enumerate(namespaces):
            store.put(
                namespace,
                f"key-{index}",
                {"content": f"memory {index}"},
                index=False,
            )

        for index, namespace in enumerate(namespaces):
            assert [item.key for item in store.search(namespace)] == [f"key-{index}"]
    finally:
        close_memory_store(store)
