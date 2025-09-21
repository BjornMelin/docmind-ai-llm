"""Unit tests for graph traversal and export helpers (library-first).

Covers JSONL schema and Parquet optional export behavior.
"""

from __future__ import annotations

import builtins
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

    def get_rel_map(self, node_ids=None, depth=1, **_kwargs):
        items = list(node_ids or [])
        if len(items) < 2:
            return []
        return [
            json.dumps(
                {
                    "subject": items[0],
                    "relation": "related",
                    "object": items[1],
                    "depth": depth,
                    "path_id": 0,
                    "source_ids": [f"src-{items[0]}", f"src-{items[1]}"]
                    if items
                    else [],
                }
            )
        ]

    class _Frame:
        def __init__(self, rows: list[str]) -> None:
            self._rows = rows

        def to_parquet(self, path: Path) -> None:
            try:
                import pyarrow  # type: ignore  # noqa: F401
            except ImportError as exc:
                raise ImportError("pyarrow missing") from exc
            Path(path).write_bytes(b"parquet-stub")

    def store_rel_map_df(self, node_ids=None, depth=1, **_kwargs):
        rows = self.get_rel_map(node_ids=node_ids, depth=depth)
        return self._Frame(rows)


class _PgIndex:
    def __init__(self) -> None:
        self.property_graph_store = _Store()


@pytest.mark.unit
def test_export_jsonl_schema(tmp_path: Path) -> None:
    idx = _PgIndex()
    out = tmp_path / "graph.jsonl"
    export_graph_jsonl(
        property_graph_index=idx,
        output_path=out,
        seed_node_ids=["A", "B"],
        depth=1,
    )  # type: ignore[arg-type]
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    row = json.loads(lines[0])
    assert {"subject", "relation", "object", "depth", "path_id", "source_ids"}.issubset(
        row.keys()
    )
    assert row["subject"] == "A"
    assert row["object"] == "B"
    assert row["relation"] == "related"
    assert row["depth"] == 1
    assert row["path_id"] == 0
    assert row["source_ids"]


@pytest.mark.unit
def test_export_parquet_optional(monkeypatch, tmp_path: Path) -> None:
    # Simulate missing pyarrow by forcing ImportError
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.startswith("pyarrow"):
            raise ImportError("no pyarrow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    idx = _PgIndex()
    out = tmp_path / "graph.parquet"
    export_graph_parquet(
        property_graph_index=idx,
        output_path=out,
        seed_node_ids=["A", "B"],
        depth=1,
    )  # type: ignore[arg-type]
    # Should not raise; and file should not exist
    assert not out.exists()


def test_export_jsonl_preserves_relation_label(tmp_path: Path) -> None:
    class _Rel:
        def __init__(self, label: str) -> None:
            self.label = label

    class _StoreWithLabels(_Store):
        def get_rel_map(self, node_ids=None, depth=1, **_kwargs):  # type: ignore[override]
            items = list(node_ids or [])
            if len(items) < 2:
                return []
            return [
                json.dumps(
                    {
                        "subject": items[0],
                        "relation": "USES",
                        "object": items[1],
                        "depth": depth,
                        "path_id": 0,
                        "source_ids": ["src-A", "src-B"],
                    }
                )
            ]

    class _PgIndexLabel:
        def __init__(self) -> None:
            self.property_graph_store = _StoreWithLabels()

    idx = _PgIndexLabel()
    out = tmp_path / "graph.jsonl"
    export_graph_jsonl(
        property_graph_index=idx,
        output_path=out,
        seed_node_ids=["A", "B"],
        depth=1,
    )  # type: ignore[arg-type]
    line = out.read_text(encoding="utf-8").strip().splitlines()[0]
    row = json.loads(line)
    assert row["relation"] == "USES"
