from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.base import SearchItem
from pydantic import ValidationError

from src.agents.tools.memory import (
    ConsolidationAction,
    ExtractedMemory,
    MemoryCandidate,
    MemoryConsolidationPolicy,
    MemoryExtractionResult,
    apply_consolidation_policy,
    consolidate_memory_candidates,
    extract_memory_candidates,
)


class _FakeStore:
    def __init__(self, items: list[SearchItem]) -> None:
        self.items = items
        self.put_calls: list[dict] = []
        self.deleted: list[str] = []

    def search(self, namespace, *, limit=10, offset=0, query=None, filt=None):
        return self.items[offset : offset + limit]

    def put(self, namespace, key, value, index=None, ttl=None):
        self.put_calls.append({"key": key, "value": value, "index": index, "ttl": ttl})

    def delete(self, namespace, key):
        self.deleted.append(str(key))


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
        MemoryCandidate(
            content="test", kind="invalid", importance=0.5, source_checkpoint_id="chk_1"
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
        SearchItem(namespace, "k1", {"importance": 0.1}, now, now, score=None),
        SearchItem(namespace, "k2", {"importance": 0.9}, now, now, score=None),
        SearchItem(namespace, "k3", {"importance": 0.2}, now, now, score=None),
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

    count = apply_consolidation_policy(store, namespace, actions, policy=policy)
    assert count >= 2
    assert store.put_calls
    assert store.put_calls[0]["ttl"] == 60
    assert "k1" in store.deleted


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
