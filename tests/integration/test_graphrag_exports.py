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
        """Return nodes by ids, ignoring requested properties in tests."""
        del properties
        return [_Node(str(i), source_id=f"src-{i}") for i in ids or []]

    def get_nodes(self):
        """Yield a small set of nodes for export tests."""
        yield from [
            _Node("A", source_id="a"),
            _Node("B", source_id="b"),
        ]

    def get_rel_map(self, node_ids=None, depth=1, **_kwargs):
        """Return a minimal relation path from the given nodes."""
        depth_value = int(depth)
        items = list(node_ids or [])
        if len(items) < 2:
            return []
        return [
            json.dumps(
                {
                    "subject": items[0],
                    "relation": "related",
                    "object": items[1],
                    "depth": depth_value,
                }
            )
        ]

    class _Frame:
        def __init__(self, rows: list[str]) -> None:
            self._rows = rows

        def to_parquet(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"PAR1")

    def store_rel_map_df(self, node_ids=None, depth=1, **_kwargs):
        rows = self.get_rel_map(node_ids=node_ids, depth=depth)
        return self._Frame(rows)


class _PgIndex:
    def __init__(self) -> None:
        self.property_graph_store = _Store()


@pytest.mark.integration
def test_jsonl_export(tmp_path: Path) -> None:
    """Export minimal relation path to JSONL and validate schema."""
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
    assert row["subject"] == "A"
    assert row["object"] == "B"


@pytest.mark.integration
def test_parquet_export_conditional(tmp_path: Path) -> None:
    """Export minimal relation path to Parquet when PyArrow is available."""
    idx = _PgIndex()
    out = tmp_path / "graph.parquet"
    export_graph_parquet(  # type: ignore[arg-type]
        property_graph_index=idx,
        output_path=out,
        seed_node_ids=["A", "B"],
        depth=1,
    )
    # Parquet may or may not be present; assert no exception and best-effort file
    # When pyarrow exists, file should exist
    if _has_pyarrow():
        assert out.exists()


def _has_pyarrow() -> bool:
    """Return whether pyarrow can be imported in the environment."""
    import importlib

    try:
        importlib.import_module("pyarrow")
        return True
    except ImportError:
        return False
