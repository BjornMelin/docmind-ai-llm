from __future__ import annotations

import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.base import BaseStore, SearchItem
from langgraph.store.memory import InMemoryStore
from pydantic import ValidationError

from src.agents.tools import memory as memory_module
from src.agents.tools.memory import (
    ConsolidationAction,
    ExtractedMemory,
    MemoryCandidate,
    MemoryConsolidationPolicy,
    MemoryExtractionResult,
    MemoryWrite,
    apply_consolidation_policy,
    consolidate_memory_candidates,
    extract_memory_candidates,
    memory_id,
    save_memory,
)


class _FakeStore:
    def __init__(self, items: list[SearchItem]) -> None:
        self.items = items
        self.put_calls: list[dict] = []
        self.records: dict[tuple[tuple[str, ...], str], dict] = {}
        self.deleted: list[str] = []

    def search(self, namespace, *, limit=10, offset=0, query=None, **_kwargs):
        return self.items[offset : offset + limit]

    def put(self, namespace, key, value, index=None, ttl=None):
        self.put_calls.append({"key": key, "value": value, "index": index, "ttl": ttl})
        self.records[(namespace, str(key))] = value

    def get(self, namespace, key):
        value = self.records.get((namespace, str(key)))
        return SimpleNamespace(value=value) if value is not None else None

    def delete(self, namespace, key):
        self.deleted.append(str(key))
        self.records.pop((namespace, str(key)), None)

    def batch(self, operations):
        for operation in operations:
            if operation.value is None:
                self.delete(operation.namespace, operation.key)
            else:
                self.put(
                    operation.namespace,
                    operation.key,
                    operation.value,
                    index=operation.index,
                    ttl=operation.ttl,
                )
        return [None for _ in operations]


def test_memory_candidate_validation():
    """Test MemoryCandidate model validation."""
    cand = MemoryCandidate(
        content="User prefers dark mode",
        kind="preference",
        importance=0.9,
        source_checkpoint_id="chk_1",
        tags=["ui"],
    )
    assert cand.content == "User prefers dark mode"

    with pytest.raises(ValidationError):
        MemoryCandidate.model_validate(
            {
                "content": "test",
                "kind": "invalid",
                "importance": 0.5,
                "source_checkpoint_id": "chk_1",
            }
        )


def test_consolidate_memory_candidates_new():
    """Test consolidation when no similar memory exists."""
    store = MagicMock()
    store.search.return_value = []

    cand = MemoryCandidate(
        content="New fact", kind="fact", importance=0.5, source_checkpoint_id="chk_1"
    )

    actions = consolidate_memory_candidates([cand], store, ("ns",))
    assert len(actions) == 1
    assert actions[0].action == "ADD"
    assert actions[0].candidate == cand


def test_consolidate_memory_candidates_noop():
    """Test consolidation when identical memory exists and importance is lower."""
    store = MagicMock()
    existing_item = MagicMock()
    existing_item.key = "mem_1"
    existing_item.value = {
        "content": "Existing fact",
        "kind": "fact",
        "importance": 0.9,
    }
    existing_item.score = 0.95
    store.search.return_value = [existing_item]

    cand = MemoryCandidate(
        content="Existing fact",
        kind="fact",
        importance=0.5,
        source_checkpoint_id="chk_1",
    )

    actions = consolidate_memory_candidates([cand], store, ("ns",))
    assert len(actions) == 1
    assert actions[0].action == "NOOP"
    assert actions[0].existing_id == "mem_1"


def test_consolidate_memory_candidates_update():
    """Test consolidation when similar memory exists.

    Candidate is higher importance.
    """
    store = MagicMock()
    existing_item = MagicMock()
    existing_item.key = "mem_1"
    existing_item.value = {
        "content": "Old preference",
        "kind": "preference",
        "importance": 0.2,
    }
    existing_item.score = 0.9
    store.search.return_value = [existing_item]

    cand = MemoryCandidate(
        content="New preference",
        kind="preference",
        importance=0.8,
        source_checkpoint_id="chk_1",
    )

    actions = consolidate_memory_candidates([cand], store, ("ns",))
    assert len(actions) == 1
    assert actions[0].action == "UPDATE"
    assert actions[0].existing_id == "mem_1"
    assert actions[0].candidate == cand


def test_consolidate_memory_candidates_score_none_fallback():
    """When semantic scores are unavailable, fall back to exact match check."""
    store = MagicMock()
    existing_item = MagicMock()
    existing_item.key = "mem_1"
    existing_item.value = {
        "content": "Existing fact",
        "kind": "fact",
        "importance": 0.7,
    }
    existing_item.score = None
    store.search.return_value = [existing_item]

    cand = MemoryCandidate(
        content="Existing fact",
        kind="fact",
        importance=0.4,
        source_checkpoint_id="chk_1",
    )

    actions = consolidate_memory_candidates([cand], store, ("ns",))
    assert len(actions) == 1
    assert actions[0].action == "NOOP"


def test_apply_consolidation_policy_ttl_and_eviction():
    """Apply TTL to low-importance items and evict when over cap."""
    now = datetime.now(UTC)
    namespace = ("memories", "user1")
    items = [
        SearchItem(
            namespace,
            "k1",
            {"importance": 0.1, "origin": "consolidation"},
            now,
            now,
            score=None,
        ),
        SearchItem(
            namespace,
            "k2",
            {"importance": 0.9, "origin": "consolidation"},
            now,
            now,
            score=None,
        ),
        SearchItem(
            namespace,
            "k3",
            {"importance": 0.2, "origin": "consolidation"},
            now,
            now,
            score=None,
        ),
    ]
    store = _FakeStore(items)

    policy = MemoryConsolidationPolicy(
        similarity_threshold=0.85,
        low_importance_threshold=0.3,
        low_importance_ttl_minutes=60,
        max_items_per_namespace=2,
        max_candidates_per_turn=8,
    )

    actions = [
        ConsolidationAction(
            action="ADD",
            candidate=MemoryCandidate(
                content="Add me", kind="fact", importance=0.2, source_checkpoint_id="c1"
            ),
        )
    ]

    count = apply_consolidation_policy(
        cast(BaseStore, store),
        namespace,
        actions,
        deadline_ts=time.monotonic() + 60.0,
        policy=policy,
    )
    assert count >= 2
    assert store.put_calls
    assert store.put_calls[0]["ttl"] == 60
    assert "k1" in store.deleted


def test_consolidation_add_retries_upsert_deterministic_key():
    """A replayed candidate remains one record across checkpoint ids."""
    namespace = ("memories", "user1", "thread1")
    store = _FakeStore([])
    policy = MemoryConsolidationPolicy(max_items_per_namespace=200)

    for content, checkpoint_id in ((" Remember Cats ", "c1"), ("remember cats", "c2")):
        actions = [
            ConsolidationAction(
                action="ADD",
                candidate=MemoryCandidate(
                    content=content,
                    kind="fact",
                    importance=0.7,
                    source_checkpoint_id=checkpoint_id,
                ),
            )
        ]
        assert (
            apply_consolidation_policy(
                cast(BaseStore, store),
                namespace,
                actions,
                policy=policy,
                deadline_ts=time.monotonic() + 60.0,
            )
            == 1
        )

    assert len(store.records) == 1
    assert store.put_calls[0]["key"] == store.put_calls[1]["key"]


def test_stale_consolidation_action_cannot_overwrite_explicit_memory():
    """A stale ADD decision yields to a later explicit write."""
    namespace = ("memories", "user1", "thread1")
    store = _FakeStore([])
    candidate = MemoryCandidate(
        content="Remember cats",
        kind="fact",
        importance=0.2,
        source_checkpoint_id="old-checkpoint",
        tags=["stale"],
    )
    stale_actions = consolidate_memory_candidates(
        [candidate], cast(BaseStore, store), namespace
    )

    explicit_id = save_memory(
        cast(BaseStore, store),
        namespace,
        write=MemoryWrite(
            content="remember cats",
            kind="fact",
            importance=0.95,
            tags=["explicit"],
            origin="explicit",
        ),
    )
    assert explicit_id is not None

    assert (
        apply_consolidation_policy(
            cast(BaseStore, store),
            namespace,
            stale_actions,
            deadline_ts=time.monotonic() + 60.0,
        )
        == 0
    )
    assert store.records[(namespace, explicit_id)] == {
        "content": "remember cats",
        "kind": "fact",
        "importance": 0.95,
        "tags": ["explicit"],
        "origin": "explicit",
    }


def test_namespace_cap_never_evicts_explicit_memory_for_derived_candidate():
    namespace = ("memories", "cap-user", "cap-thread")
    store = InMemoryStore()
    explicit_id = save_memory(
        store,
        namespace,
        write=MemoryWrite(
            content="user-authored",
            kind="fact",
            importance=0.1,
            tags=None,
            origin="explicit",
        ),
    )
    assert explicit_id is not None
    candidate = MemoryCandidate(
        content="derived",
        kind="fact",
        importance=0.9,
        source_checkpoint_id="checkpoint-1",
    )

    changes = apply_consolidation_policy(
        store,
        namespace,
        [ConsolidationAction(action="ADD", candidate=candidate)],
        deadline_ts=time.monotonic() + 60.0,
        policy=MemoryConsolidationPolicy(max_items_per_namespace=1),
    )

    assert changes >= 1
    records = store.search(namespace, limit=10)
    assert [record.key for record in records] == [explicit_id]
    assert records[0].value["origin"] == "explicit"


def test_consolidation_update_rekeys_to_canonical_identity():
    namespace = ("memories", "user1", "thread1")
    store = _FakeStore([])
    old_id = "old-random-key"
    store.records[(namespace, old_id)] = {
        "content": "Old preference",
        "kind": "preference",
        "importance": 0.2,
        "origin": "consolidation",
    }
    candidate = MemoryCandidate(
        content="New preference",
        kind="preference",
        importance=0.8,
        source_checkpoint_id="new-checkpoint",
    )

    assert (
        apply_consolidation_policy(
            cast(BaseStore, store),
            namespace,
            [
                ConsolidationAction(
                    action="UPDATE", existing_id=old_id, candidate=candidate
                )
            ],
            deadline_ts=time.monotonic() + 60.0,
        )
        == 1
    )
    canonical_id = memory_id(candidate.content, candidate.kind)
    assert (namespace, old_id) not in store.records
    assert (namespace, canonical_id) in store.records


def test_consolidation_deadline_stops_later_writes(monkeypatch):
    namespace = ("memories", "user1", "thread1")
    clock = [1.0]

    class _ExpiringStore(_FakeStore):
        def put(self, namespace, key, value, index=None, ttl=None):
            super().put(namespace, key, value, index=index, ttl=ttl)
            clock[0] = 11.0

    store = _ExpiringStore([])
    monkeypatch.setattr(memory_module.time, "monotonic", lambda: clock[0])
    actions = [
        ConsolidationAction(
            action="ADD",
            candidate=MemoryCandidate(
                content=content,
                kind="fact",
                importance=0.7,
                source_checkpoint_id="checkpoint",
            ),
        )
        for content in ("first", "second")
    ]

    assert (
        apply_consolidation_policy(
            cast(BaseStore, store),
            namespace,
            actions,
            deadline_ts=10.0,
        )
        == 1
    )
    assert [call["value"]["content"] for call in store.put_calls] == ["first"]


def test_consolidation_deadline_stops_later_decisions(monkeypatch):
    namespace = ("memories", "user1", "thread1")
    clock = [1.0]
    searches: list[str] = []

    class _ExpiringStore(_FakeStore):
        def search(self, namespace, *, limit=10, offset=0, query=None, **kwargs):
            del namespace, limit, offset, kwargs
            searches.append(str(query))
            clock[0] = 11.0
            return []

    store = _ExpiringStore([])
    monkeypatch.setattr(memory_module.time, "monotonic", lambda: clock[0])
    candidates = [
        MemoryCandidate(
            content=content,
            kind="fact",
            importance=0.7,
            source_checkpoint_id="checkpoint",
        )
        for content in ("first", "second")
    ]

    actions = consolidate_memory_candidates(
        candidates,
        cast(BaseStore, store),
        namespace,
        deadline_ts=10.0,
    )

    assert searches == ["first"]
    assert [action.candidate.content for action in actions if action.candidate] == [
        "first"
    ]


def test_consolidation_deadline_stops_later_evictions(monkeypatch):
    namespace = ("memories", "user1", "thread1")
    now = datetime.now(UTC)
    clock = [1.0]

    class _ExpiringStore(_FakeStore):
        def delete(self, namespace, key):
            super().delete(namespace, key)
            clock[0] = 11.0

    items = [
        SearchItem(
            namespace,
            f"memory-{index}",
            {"importance": 0.1, "origin": "consolidation"},
            now,
            now,
            score=None,
        )
        for index in range(3)
    ]
    store = _ExpiringStore(items)
    monkeypatch.setattr(memory_module.time, "monotonic", lambda: clock[0])

    assert (
        apply_consolidation_policy(
            cast(BaseStore, store),
            namespace,
            [],
            deadline_ts=10.0,
            policy=MemoryConsolidationPolicy(max_items_per_namespace=1),
        )
        == 1
    )
    assert len(store.deleted) == 1


def test_consolidation_logs_exclude_raw_identity_and_content(monkeypatch):
    namespace = ("memories", "private-user", "private-thread")
    raw_content = "private-memory-content"
    raw_error = "private-store-error"
    captured: list[object] = []

    class _FailingStore(_FakeStore):
        def put(self, namespace, key, value, index=None, ttl=None):
            raise RuntimeError(raw_error)

    fake_logger = SimpleNamespace(
        warning=lambda *args, **_kwargs: captured.extend(args),
        info=lambda *args, **_kwargs: captured.extend(args),
        debug=lambda *args, **_kwargs: captured.extend(args),
    )
    monkeypatch.setattr(memory_module, "logger", fake_logger)
    store = _FailingStore([])

    apply_consolidation_policy(
        cast(BaseStore, store),
        namespace,
        [
            ConsolidationAction(
                action="ADD",
                candidate=MemoryCandidate(
                    content=raw_content,
                    kind="fact",
                    importance=0.7,
                    source_checkpoint_id="private-checkpoint",
                ),
            )
        ],
        deadline_ts=time.monotonic() + 60.0,
    )

    logged = " ".join(str(value) for value in captured)
    assert raw_content not in logged
    assert raw_error not in logged
    assert "private-user" not in logged
    assert "private-thread" not in logged
    assert memory_id(raw_content, "fact") not in logged


def test_extract_memory_candidates_empty():
    """Test extraction with empty input."""
    assert extract_memory_candidates([], "chk_1") == []
    assert extract_memory_candidates([HumanMessage(content="hi")], "chk_1", None) == []


def test_extract_memory_candidates_structured_output():
    """Test extraction using structured output path."""
    structured = MagicMock()
    structured.invoke.return_value = MemoryExtractionResult(
        memories=[
            ExtractedMemory(content="fact1", kind="fact", importance=0.5, tags=["t"])
        ]
    )

    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    messages = [HumanMessage(content="Remember that I like cats")]
    candidates = extract_memory_candidates(messages, "chk_1", llm)

    assert len(candidates) == 1
    assert candidates[0].content == "fact1"
    assert candidates[0].source_checkpoint_id == "chk_1"


def test_extract_memory_candidates_passes_remaining_deadline_to_provider(
    monkeypatch,
):
    structured = MagicMock()
    structured.invoke.return_value = MemoryExtractionResult(
        memories=[ExtractedMemory(content="fact1", kind="fact", importance=0.5)]
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(memory_module.time, "monotonic", lambda: 100.0)

    candidates = extract_memory_candidates(
        [HumanMessage(content="Remember this")],
        "chk_1",
        llm,
        deadline_ts=103.5,
    )

    assert len(candidates) == 1
    assert structured.invoke.call_args.kwargs["timeout"] == 3.5


def test_extract_memory_candidates_skips_expired_provider_call(monkeypatch):
    llm = MagicMock()
    monkeypatch.setattr(memory_module.time, "monotonic", lambda: 100.0)

    assert (
        extract_memory_candidates(
            [HumanMessage(content="Remember this")],
            "chk_1",
            llm,
            deadline_ts=100.0,
        )
        == []
    )
    llm.with_structured_output.assert_not_called()


def test_extract_memory_candidates_json_fallback():
    """Test extraction when structured output is unavailable."""
    llm = MagicMock(spec=["invoke"])
    llm.invoke.return_value = MagicMock(
        content=(
            '{"memories": [{"content": "fact1", "kind": "fact", "importance": 0.5}]}'
        )
    )

    messages = [HumanMessage(content="Remember that I like cats")]
    candidates = extract_memory_candidates(messages, "chk_1", llm)

    assert len(candidates) == 1
    assert candidates[0].content == "fact1"
    assert candidates[0].source_checkpoint_id == "chk_1"
