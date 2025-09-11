"""Unit tests for portable utilities in graph_config.

Focus on pure helpers and documented store APIs (`get`, `get_rel_map`) without
requiring real LlamaIndex graph construction.
"""

import pytest


@pytest.mark.unit
def test_create_tech_schema_contains_expected_entities_relations():
    """Test that tech schema contains expected entities and relations."""
    from src.retrieval.graph_config import create_tech_schema

    schema = create_tech_schema()
    assert set(schema.keys()) == {"entities", "relations"}
    assert "FRAMEWORK" in schema["entities"]
    assert "ORG" in schema["entities"]
    assert "USES" in schema["relations"]
    assert "SUPPORTS" in schema["relations"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_portable_helpers_minimal_behaviors(monkeypatch):
    """Test portable helpers that use get/get_rel_map and retriever.

    Builds a small stub index and store to validate helper behavior without
    constructing a real PropertyGraphIndex.
    """
    from src.retrieval.graph_config import (
        extract_entities,
        extract_relationships,
        traverse_graph,
    )

    class _Node:
        def __init__(self, node_id: str) -> None:
            self.id = node_id
            self.name = node_id

    class _Store:
        def get(self, ids=None, properties=None):
            del properties
            return [_Node(str(i)) for i in ids or []]

        def get_rel_map(self, nodes, depth=1):
            del depth
            items = list(nodes)
            if len(items) < 2:
                return []
            return [[items[0], items[1]]]

    class _Retriever:
        def __init__(self, result):
            self._result = result

        def retrieve(self, query: str):
            del query
            return self._result

    class _Index:
        def __init__(self):
            self.property_graph_store = _Store()

        def as_retriever(self, include_text=False, path_depth=1):
            del include_text, path_depth
            return _Retriever(result=[{"text": "ok"}])

    idx = _Index()

    entities = await extract_entities(idx, seed_ids=["A"])  # type: ignore[arg-type]
    assert isinstance(entities, list)

    rels = await extract_relationships(idx, seed_ids=["A", "B"])  # type: ignore[arg-type]
    assert rels
    assert {"head", "relation", "tail"}.issubset(rels[0].keys())

    nodes = await traverse_graph(idx, "query")  # type: ignore[arg-type]
    assert nodes
    assert isinstance(nodes, list)
