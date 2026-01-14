from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import UUID

import pytest

from src.agents.tools import memory as memory_tools

pytestmark = pytest.mark.unit


@dataclass
class _SearchItem:
    key: str
    value: object
    score: float | None = None


class _Store:
    def __init__(self) -> None:
        self.put_calls: list[
            tuple[tuple[str, ...], str, dict[str, object], object]
        ] = []
        self.search_calls: list[tuple[tuple[str, ...], str, int]] = []
        self.delete_calls: list[tuple[tuple[str, ...], str]] = []

    def put(self, ns: tuple[str, ...], key: str, value: dict[str, object], index=None):  # type: ignore[no-untyped-def]
        self.put_calls.append((ns, key, value, index))

    def search(self, ns: tuple[str, ...], query: str, limit: int):  # type: ignore[no-untyped-def]
        self.search_calls.append((ns, query, limit))
        return [
            _SearchItem(key="m1", value={"content": "c", "kind": "fact"}, score=0.9),
            _SearchItem(key="m2", value="not-a-dict", score=0.1),
        ]

    def delete(self, ns: tuple[str, ...], key: str):  # type: ignore[no-untyped-def]
        self.delete_calls.append((ns, key))


def _runtime(
    store: _Store | None, *, user_id: str = "u", thread_id: str = "t"
) -> object:
    return SimpleNamespace(
        store=store,
        config={"configurable": {"user_id": user_id, "thread_id": thread_id}},
    )


def test_namespace_from_config_scopes() -> None:
    cfg = {"configurable": {"user_id": "u1", "thread_id": "t1"}}
    assert memory_tools._namespace_from_config(cfg, scope="session") == (  # type: ignore[attr-defined]
        "memories",
        "u1",
        "t1",
    )
    assert memory_tools._namespace_from_config(cfg, scope="user") == (  # type: ignore[attr-defined]
        "memories",
        "u1",
    )


def test_remember_returns_error_when_store_missing() -> None:
    out = json.loads(memory_tools.remember.func("hi", runtime=_runtime(None)))  # type: ignore[attr-defined]
    assert out["ok"] is False


def test_remember_persists_payload_and_emits_id(monkeypatch) -> None:
    store = _Store()
    events: list[dict[str, object]] = []
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda ev: events.append(ev))

    out = json.loads(
        memory_tools.remember.func(  # type: ignore[attr-defined]
            "hello",
            kind="fact",
            importance=0.5,
            tags=("x", "y"),
            scope="session",
            runtime=_runtime(store),
        )
    )
    assert out["ok"] is True
    UUID(out["memory_id"])  # validates uuid

    assert len(store.put_calls) == 1
    ns, key, value, index = store.put_calls[0]
    assert ns == ("memories", "u", "t")
    assert key == out["memory_id"]
    assert value["content"] == "hello"
    assert value["kind"] == "fact"
    assert value["importance"] == 0.5
    assert value["tags"] == ["x", "y"]
    assert index == ["content"]

    assert events
    assert events[-1].get("chat.memory_saved") is True


def test_remember_ignores_string_tags(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _ev: None)
    out = json.loads(
        memory_tools.remember.func("hi", tags="oops", runtime=_runtime(store))  # type: ignore[attr-defined]
    )
    assert out["ok"] is True
    _ns, _key, value, _index = store.put_calls[0]
    assert value["tags"] is None


def test_recall_memories_filters_non_dict_values(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _ev: None)
    out = json.loads(memory_tools.recall_memories.func("q", runtime=_runtime(store)))  # type: ignore[attr-defined]
    assert out["ok"] is True
    assert out["memories"] == [
        {
            "id": "m1",
            "content": "c",
            "kind": "fact",
            "importance": None,
            "tags": None,
            "score": 0.9,
        }
    ]


def test_forget_memory_deletes_and_suppresses_telemetry_errors(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(
        memory_tools,
        "log_jsonl",
        lambda _ev: (_ for _ in ()).throw(RuntimeError("x")),
    )
    out = json.loads(memory_tools.forget_memory.func("m1", runtime=_runtime(store)))  # type: ignore[attr-defined]
    assert out == {"ok": True}
    assert store.delete_calls == [(("memories", "u", "t"), "m1")]


def test_suppress_telemetry_context_manager_swallows_runtimeerror() -> None:
    with memory_tools._SuppressTelemetry():  # type: ignore[attr-defined]
        raise RuntimeError("boom")
