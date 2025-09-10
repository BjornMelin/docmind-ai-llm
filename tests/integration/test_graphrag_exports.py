"""Integration tests for GraphRAG exports (JSONL/Parquet)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.retrieval.graph_config import export_graph_jsonl, export_graph_parquet


class _Node:
    def __init__(self, node_id: str, **props: Any) -> None:
        self.id = node_id
        self.name = node_id
        self.properties = props


class _Store:
    def get(self, ids=None, properties=None):
        del properties
        return [_Node(str(i), source_id=f"src-{i}") for i in ids or []]

    def get_nodes(self):
        yield from [
            _Node("A", source_id="a"),
            _Node("B", source_id="b"),
        ]

    def get_rel_map(self, nodes, depth=1):
        del depth
        items = list(nodes)
        if len(items) < 2:
            return []
        return [[items[0], items[1]]]


class _PgIndex:
    def __init__(self) -> None:
        self.property_graph_store = _Store()


@pytest.mark.integration
def test_jsonl_export(tmp_path: Path) -> None:
    idx = _PgIndex()
    out = tmp_path / "graph.jsonl"
    export_graph_jsonl(idx, out, seed_ids=["A", "B"], depth=1)  # type: ignore[arg-type]
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    row = json.loads(lines[0])
    assert row["subject"] == "A"
    assert row["object"] == "B"


@pytest.mark.integration
def test_parquet_export_conditional(tmp_path: Path) -> None:
    idx = _PgIndex()
    out = tmp_path / "graph.parquet"
    export_graph_parquet(idx, out, seed_ids=["A", "B"], depth=1)  # type: ignore[arg-type]
    # Parquet may or may not be present; assert no exception and best-effort file
    # When pyarrow exists, file should exist
    if _has_pyarrow():
        assert out.exists()


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False
