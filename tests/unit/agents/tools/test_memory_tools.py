from __future__ import annotations

import json
import math
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

from src.agents.models import MemoryNamespaceGenerations
from src.agents.tools import memory as memory_tools
from src.persistence.checkpoint_identity import memory_namespace

pytestmark = pytest.mark.unit


class _MemoryState(MessagesState):
    deadline_ts: float
    memory_generations: MemoryNamespaceGenerations


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

    def get(self, _ns: tuple[str, ...], _key: str) -> None:
        return None

    def delete(self, ns: tuple[str, ...], key: str):  # type: ignore[no-untyped-def]
        self.delete_calls.append((ns, key))


def _runtime(store: Any | None, *, user_id: str = "u", thread_id: str = "t") -> object:
    return SimpleNamespace(
        store=store,
        config={"configurable": {"user_id": user_id, "thread_id": thread_id}},
    )


def _state(*, user_id: str = "u", thread_id: str = "t") -> dict[str, Any]:
    return {
        "deadline_ts": time.monotonic() + 60.0,
        "memory_generations": memory_tools.capture_memory_namespace_generations(
            user_id=user_id,
            thread_id=thread_id,
        ),
    }


def test_namespace_from_config_scopes() -> None:
    cfg = {
        "configurable": {
            "user_id": "u1",
            "thread_id": "private-checkpoint-id",
            "public_thread_id": "t1",
        }
    }
    assert memory_tools._namespace_from_config(  # type: ignore[attr-defined]
        cfg, scope="session"
    ) == memory_namespace(
        user_id="u1",
        thread_id="t1",
    )
    assert memory_tools._namespace_from_config(  # type: ignore[attr-defined]
        cfg, scope="user"
    ) == memory_namespace(
        user_id="u1",
    )
    assert memory_tools._ids_from_config(cfg) == ("u1", "t1")  # type: ignore[attr-defined]


def test_native_inmemory_namespaces_isolate_user_and_sessions(monkeypatch) -> None:
    """User-global recall excludes sessions, and sessions exclude each other."""
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _event: None)
    store = InMemoryStore()
    user_id = "isolation-user"
    thread_a = "thread-a"
    thread_b = "thread-b"
    user_namespace = memory_namespace(user_id=user_id)
    session_a = memory_namespace(user_id=user_id, thread_id=thread_a)
    session_b = memory_namespace(user_id=user_id, thread_id=thread_b)
    for namespace, key, content in (
        (user_namespace, "user", "user global"),
        (session_a, "a", "session a"),
        (session_b, "b", "session b"),
    ):
        store.put(namespace, key, {"content": content, "kind": "fact"})

    assert [item.key for item in store.search(user_namespace)] == ["user"]
    assert [item.key for item in store.search(session_a)] == ["a"]
    assert [item.key for item in store.search(session_b)] == ["b"]

    runtime = _runtime(store, user_id=user_id, thread_id=thread_a)
    user_result = json.loads(
        memory_tools.recall_memories.func(  # type: ignore[attr-defined]
            "memory",
            scope="user",
            state={"deadline_ts": time.monotonic() + 60.0},
            runtime=runtime,
        )
    )
    session_result = json.loads(
        memory_tools.recall_memories.func(  # type: ignore[attr-defined]
            "memory",
            scope="session",
            state={"deadline_ts": time.monotonic() + 60.0},
            runtime=runtime,
        )
    )
    assert [item["content"] for item in user_result["memories"]] == ["user global"]
    assert [item["content"] for item in session_result["memories"]] == ["session a"]


@pytest.mark.parametrize(
    ("memory_tool", "visible_fields"),
    [
        (
            memory_tools.remember,
            {"content", "kind", "importance", "tags", "scope"},
        ),
        (memory_tools.recall_memories, {"query", "limit", "scope"}),
        (memory_tools.forget_memory, {"memory_id", "scope"}),
    ],
)
def test_memory_tool_schemas_hide_injected_arguments(
    memory_tool, visible_fields: set[str]
) -> None:
    schema = memory_tool.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == visible_fields


def test_langgraph_injects_memory_state_and_runtime(monkeypatch) -> None:
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _event: None)
    store = InMemoryStore()
    builder = StateGraph(_MemoryState)
    builder.add_node("tools", ToolNode([memory_tools.remember]))
    builder.set_entry_point("tools")
    builder.set_finish_point("tools")
    graph = builder.compile(store=store)

    result = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "remember",
                            "args": {"content": "Cats"},
                            "id": "memory-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "deadline_ts": time.monotonic() + 60.0,
            "memory_generations": (
                memory_tools.capture_memory_namespace_generations(
                    user_id="u",
                    thread_id="t",
                )
            ),
        },
        {"configurable": {"user_id": "u", "thread_id": "t"}},
    )

    payload = json.loads(result["messages"][-1].content)
    assert payload["ok"] is True
    assert len(store.search(memory_namespace(user_id="u", thread_id="t"))) == 1


def test_langgraph_injects_recall_deadline_state(monkeypatch) -> None:
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _event: None)
    store = InMemoryStore()
    namespace = memory_namespace(user_id="u", thread_id="t")
    store.put(namespace, "memory", {"content": "Cats", "kind": "fact"})
    builder = StateGraph(_MemoryState)
    builder.add_node("tools", ToolNode([memory_tools.recall_memories]))
    builder.set_entry_point("tools")
    builder.set_finish_point("tools")
    graph = builder.compile(store=store)

    result = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "recall_memories",
                            "args": {"query": "cats"},
                            "id": "recall-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "deadline_ts": time.monotonic() + 60.0,
            "memory_generations": (
                memory_tools.capture_memory_namespace_generations(
                    user_id="u",
                    thread_id="t",
                )
            ),
        },
        {"configurable": {"user_id": "u", "thread_id": "t"}},
    )

    payload = json.loads(result["messages"][-1].content)
    assert payload["ok"] is True
    assert [item["content"] for item in payload["memories"]] == ["Cats"]


def test_remember_returns_error_when_store_missing() -> None:
    out = json.loads(
        memory_tools.remember.func(  # type: ignore[attr-defined]
            "hi", state=_state(), runtime=_runtime(None)
        )
    )
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
            state=_state(),
            runtime=_runtime(store),
        )
    )
    assert out["ok"] is True
    assert out["memory_id"].startswith("mem-")

    assert len(store.put_calls) == 1
    ns, key, value, index = store.put_calls[0]
    assert ns == memory_namespace(user_id="u", thread_id="t")
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
        memory_tools.remember.func(  # type: ignore[attr-defined]
            "hi", state=_state(), tags="oops", runtime=_runtime(store)
        )
    )
    assert out["ok"] is True
    _ns, _key, value, _index = store.put_calls[0]
    assert value["tags"] is None


def test_recall_memories_filters_non_dict_values(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _ev: None)
    out = json.loads(
        memory_tools.recall_memories.func(  # type: ignore[attr-defined]
            "q", state=_state(), runtime=_runtime(store)
        )
    )
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


def test_memory_tools_fail_closed_on_native_sqlite_errors(monkeypatch) -> None:
    class _LockedStore(_Store):
        def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise sqlite3.OperationalError("database is locked")

        def delete(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise sqlite3.OperationalError("database is locked")

    store = _LockedStore()
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _ev: None)

    recalled = json.loads(
        memory_tools.recall_memories.func(  # type: ignore[attr-defined]
            "q", state=_state(), runtime=_runtime(store)
        )
    )
    forgotten = json.loads(
        memory_tools.forget_memory.func(  # type: ignore[attr-defined]
            "memory-1", state=_state(), runtime=_runtime(store)
        )
    )

    assert recalled == {"ok": False, "error": "search failed", "memories": []}
    assert forgotten == {"ok": False, "error": "delete failed"}
    assert "database is locked" not in json.dumps((recalled, forgotten))


def test_remember_id_is_idempotent_and_content_addressed(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _ev: None)

    def remember_id(
        content: str,
        kind: Literal["fact", "preference", "todo", "project_state"] = "fact",
    ) -> str:
        result = json.loads(
            memory_tools.remember.func(  # type: ignore[attr-defined]
                content,
                kind=kind,
                state=_state(),
                runtime=_runtime(store),
            )
        )
        return str(result["memory_id"])

    first = remember_id("  Likes Cats  ")
    assert remember_id("likes cats") == first
    assert remember_id("likes dogs") != first
    assert remember_id("likes cats", "preference") != first


def test_explicit_capacity_allows_idempotence_and_rekey_but_rejects_new(
    monkeypatch,
) -> None:
    user_id = "private-capacity-user"
    thread_id = "private-capacity-thread"
    namespace = memory_namespace(user_id=user_id, thread_id=thread_id)
    store = InMemoryStore()
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        memory_tools.settings.chat,
        "memory_max_items_per_namespace",
        1,
    )
    monkeypatch.setattr(memory_tools, "log_jsonl", events.append)
    state = _state(user_id=user_id, thread_id=thread_id)
    runtime = _runtime(store, user_id=user_id, thread_id=thread_id)

    first = json.loads(
        memory_tools.remember.func(  # type: ignore[attr-defined]
            "Keep this",
            state=state,
            runtime=runtime,
        )
    )
    repeated = json.loads(
        memory_tools.remember.func(  # type: ignore[attr-defined]
            " keep THIS ",
            state=state,
            runtime=runtime,
        )
    )
    rejected = json.loads(
        memory_tools.remember.func(  # type: ignore[attr-defined]
            "Do not leak this content",
            state=state,
            runtime=runtime,
        )
    )

    assert first["ok"] is True
    assert repeated == first
    assert rejected == {"ok": False, "error": "memory capacity reached"}
    assert len(store.search(namespace)) == 1
    capacity_event = json.dumps(events[-1])
    assert events[-1]["error_type"] == "memory_capacity_reached"
    assert user_id not in capacity_event
    assert thread_id not in capacity_event
    assert "Do not leak this content" not in capacity_event

    rekeyed_id = memory_tools.save_memory(
        store,
        namespace,
        write=memory_tools.MemoryWrite(
            content="Replacement",
            kind="fact",
            importance=0.8,
            tags=None,
            origin="explicit",
        ),
        replace_memory_id=str(first["memory_id"]),
    )
    assert rekeyed_id is not None
    assert len(store.search(namespace)) == 1
    assert store.get(namespace, str(first["memory_id"])) is None
    assert store.get(namespace, rekeyed_id) is not None


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"deadline_ts": None},
        {"deadline_ts": "invalid"},
        {"deadline_ts": math.inf},
        {"deadline_ts": math.nan},
    ],
)
def test_invalid_deadline_blocks_memory_operations(
    monkeypatch, state: dict[str, Any]
) -> None:
    store = _Store()
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _ev: None)

    remembered = json.loads(
        memory_tools.remember.func(  # type: ignore[attr-defined]
            "late", state=state, runtime=_runtime(store)
        )
    )
    recalled = json.loads(
        memory_tools.recall_memories.func(  # type: ignore[attr-defined]
            "late", state=state, runtime=_runtime(store)
        )
    )
    forgotten = json.loads(
        memory_tools.forget_memory.func(  # type: ignore[attr-defined]
            "m1", state=state, runtime=_runtime(store)
        )
    )

    assert remembered == {"ok": False, "error": "operation deadline exceeded"}
    assert recalled == {
        "ok": False,
        "error": "operation deadline exceeded",
        "memories": [],
    }
    assert forgotten == {"ok": False, "error": "operation deadline exceeded"}
    assert store.put_calls == []
    assert store.search_calls == []
    assert store.delete_calls == []


def test_expired_deadline_blocks_memory_operations(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _ev: None)
    state = {"deadline_ts": time.monotonic() - 1.0}

    remembered = json.loads(
        memory_tools.remember.func(  # type: ignore[attr-defined]
            "late", state=state, runtime=_runtime(store)
        )
    )
    recalled = json.loads(
        memory_tools.recall_memories.func(  # type: ignore[attr-defined]
            "late", state=state, runtime=_runtime(store)
        )
    )
    forgotten = json.loads(
        memory_tools.forget_memory.func(  # type: ignore[attr-defined]
            "m1", state=state, runtime=_runtime(store)
        )
    )

    assert remembered["ok"] is False
    assert recalled["ok"] is False
    assert forgotten["ok"] is False
    assert store.put_calls == []
    assert store.search_calls == []
    assert store.delete_calls == []


def test_deadline_block_telemetry_is_metadata_only(monkeypatch) -> None:
    store = _Store()
    events: list[dict[str, object]] = []
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda event: events.append(event))

    memory_tools.remember.func(  # type: ignore[attr-defined]
        "private memory content",
        state={"deadline_ts": time.monotonic() - 1.0},
        runtime=_runtime(store, user_id="private-user", thread_id="private-thread"),
    )

    event_text = json.dumps(events[-1])
    assert events[-1]["chat.memory_saved"] is False
    assert events[-1]["error_type"] == "deadline_invalid_or_expired"
    assert "private memory content" not in event_text
    assert "private-user" not in event_text
    assert "private-thread" not in event_text


def test_delayed_remember_retry_upserts_one_logical_record(monkeypatch) -> None:
    class _DelayedStore(_Store):
        def __init__(self) -> None:
            super().__init__()
            self.first_put_started = Event()
            self.release_first_put = Event()
            self.records: dict[tuple[tuple[str, ...], str], dict[str, object]] = {}

        def put(self, ns, key, value, index=None):  # type: ignore[no-untyped-def]
            first_put = not self.first_put_started.is_set()
            self.put_calls.append((ns, key, value, index))
            if first_put:
                self.first_put_started.set()
                self.release_first_put.wait(timeout=2.0)
            self.records[(ns, key)] = value

    store = _DelayedStore()
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _ev: None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            memory_tools.remember.func,  # type: ignore[attr-defined]
            "Retry me",
            importance=0.2,
            state=_state(),
            runtime=_runtime(store),
        )
        try:
            assert store.first_put_started.wait(timeout=1.0)
            retry_future = executor.submit(
                memory_tools.remember.func,  # type: ignore[attr-defined]
                "retry me",
                importance=0.9,
                state=_state(),
                runtime=_runtime(store),
            )
            assert not retry_future.done()
        finally:
            store.release_first_put.set()
        initial = json.loads(first.result(timeout=1.0))
        retry = json.loads(retry_future.result(timeout=1.0))

    assert initial["memory_id"] == retry["memory_id"]
    assert len(store.records) == 1
    assert next(iter(store.records.values()))["importance"] == 0.9
    assert next(iter(store.records.values()))["origin"] == "explicit"


def test_rekey_preserves_old_on_put_failure_and_finishes_after_admission(
    monkeypatch,
) -> None:
    clock = [1.0]

    class _RekeyStore:
        def __init__(
            self, *, fail_put: bool = False, fail_delete: bool = False
        ) -> None:
            self.fail_put = fail_put
            self.fail_delete = fail_delete
            self.calls: list[tuple[str, str]] = []
            self.records = {
                "legacy": {
                    "content": "old content",
                    "kind": "fact",
                    "origin": "consolidation",
                }
            }

        def get(self, _namespace, key):  # type: ignore[no-untyped-def]
            value = self.records.get(key)
            return SimpleNamespace(value=value) if value is not None else None

        def put(self, _namespace, key, value, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(("put", key))
            if self.fail_put:
                raise RuntimeError("put failed")
            self.records[key] = value
            clock[0] = 11.0

        def delete(self, _namespace, key):  # type: ignore[no-untyped-def]
            self.calls.append(("delete", key))
            if self.fail_delete:
                raise RuntimeError("delete failed")
            self.records.pop(key, None)

    namespace = ("memories", "rekey-order", "session")
    write = memory_tools.MemoryWrite(
        content="new content",
        kind="fact",
        importance=0.8,
        tags=None,
        origin="explicit",
    )
    canonical_id = memory_tools.memory_id(write.content, write.kind)

    monkeypatch.setattr(memory_tools.time, "monotonic", lambda: clock[0])

    failed_put = _RekeyStore(fail_put=True)
    with pytest.raises(RuntimeError, match="put failed"):
        memory_tools.save_memory(
            cast(Any, failed_put),
            namespace,
            write=write,
            replace_memory_id="legacy",
            deadline_ts=10.0,
        )
    assert failed_put.calls == [("put", canonical_id)]
    assert set(failed_put.records) == {"legacy"}

    completed = _RekeyStore()
    assert (
        memory_tools.save_memory(
            cast(Any, completed),
            namespace,
            write=write,
            replace_memory_id="legacy",
            deadline_ts=10.0,
        )
        == canonical_id
    )
    assert clock[0] == 11.0
    assert set(completed.records) == {canonical_id}
    assert completed.calls == [
        ("put", canonical_id),
        ("delete", "legacy"),
    ]


def test_real_sqlite_rekey_insert_failure_preserves_old_record() -> None:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    store = SqliteStore(conn)
    store.setup()
    namespace = ("memories", "sqlite-rekey", "session")
    write = memory_tools.MemoryWrite(
        content="new content",
        kind="fact",
        importance=0.8,
        tags=None,
        origin="explicit",
    )
    canonical_id = memory_tools.memory_id(write.content, write.kind)
    store.put(
        namespace,
        "legacy",
        {"content": "old content", "kind": "fact", "origin": "explicit"},
        index=["content"],
    )
    conn.execute(
        f"""
        CREATE TRIGGER reject_canonical_memory
        BEFORE INSERT ON store
        WHEN NEW.key = '{canonical_id}'
        BEGIN
            SELECT RAISE(ABORT, 'injected insert failure');
        END;
        """
    )

    try:
        with pytest.raises(sqlite3.IntegrityError, match="injected insert failure"):
            memory_tools.save_memory(
                store,
                namespace,
                write=write,
                replace_memory_id="legacy",
            )
        assert store.get(namespace, "legacy") is not None
        assert store.get(namespace, canonical_id) is None
    finally:
        conn.close()


def test_generation_and_tombstone_fail_closed_inside_mutations() -> None:
    namespace = ("memories", "generation-unit", "session")
    store = InMemoryStore()
    write = memory_tools.MemoryWrite(
        content="generation guarded",
        kind="fact",
        importance=0.8,
        tags=None,
        origin="consolidation",
        source_checkpoint_id="checkpoint-1",
    )

    captured = memory_tools.memory_namespace_generation(namespace)
    current = memory_tools.advance_memory_namespace_generation(namespace)
    assert current == captured + 1
    assert (
        memory_tools.save_memory(
            store,
            namespace,
            write=write,
            expected_generation=captured,
        )
        is None
    )

    memory_key = memory_tools.save_memory(
        store,
        namespace,
        write=write,
        expected_generation=current,
    )
    assert memory_key is not None
    assert store.get(namespace, memory_key) is not None

    tombstone_generation = memory_tools.tombstone_memory_namespace(namespace)
    assert tombstone_generation == current + 1
    assert memory_tools.is_memory_namespace_tombstoned(namespace)
    assert memory_tools.save_memory(store, namespace, write=write) is None
    assert not memory_tools.delete_memory(
        store,
        namespace,
        memory_key,
        expected_generation=current,
    )
    assert store.get(namespace, memory_key) is not None


def test_forget_memory_deletes_and_suppresses_telemetry_errors(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(
        memory_tools,
        "log_jsonl",
        lambda _ev: (_ for _ in ()).throw(RuntimeError("x")),
    )
    out = json.loads(
        memory_tools.forget_memory.func(  # type: ignore[attr-defined]
            "m1", state=_state(), runtime=_runtime(store)
        )
    )
    assert out == {"ok": True}
    assert store.delete_calls == [(memory_namespace(user_id="u", thread_id="t"), "m1")]


def test_suppress_telemetry_context_manager_swallows_runtimeerror() -> None:
    with memory_tools._SuppressTelemetry():  # type: ignore[attr-defined]
        raise RuntimeError("boom")
