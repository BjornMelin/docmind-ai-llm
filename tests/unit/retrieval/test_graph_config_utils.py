"""Unit tests for lightweight utilities in graph_config.

Focus on pure functions and attachment helpers to increase coverage without
requiring real LlamaIndex graph construction.
"""

from types import SimpleNamespace

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
def test_calculate_entity_confidence_branches():
    """Test entity confidence calculation with different scenarios."""
    from src.retrieval.graph_config import (
        CONTEXT_LENGTH_THRESHOLD,
        MAX_CONFIDENCE_SCORE,
        calculate_entity_confidence,
    )

    schema = {"entities": ["MODEL", "HARDWARE"]}

    base_entity = {
        "type": "MODEL",
        "extractor": "SchemaLLMPathExtractor",
        "context": "x" * (CONTEXT_LENGTH_THRESHOLD + 5),
    }
    score = calculate_entity_confidence(base_entity, schema)
    assert 0.0 < score <= MAX_CONFIDENCE_SCORE

    # Different extractor path
    simple_entity = {
        "type": "HARDWARE",
        "extractor": "SimpleLLMPathExtractor",
        "context": "short",
    }
    score2 = calculate_entity_confidence(simple_entity, schema)
    assert score2 < score  # less context bonus

    # Unknown type should not error and remain within [0,1]
    unknown = {"type": "PERSON", "extractor": "Other", "context": ""}
    score3 = calculate_entity_confidence(unknown, schema)
    assert 0.0 <= score3 <= 1.0


@pytest.mark.unit
def test_extend_property_graph_index_attaches_helpers():
    """Test that property graph index extension attaches helper methods."""
    from src.retrieval.graph_config import extend_property_graph_index

    dummy_index = SimpleNamespace()
    extended = extend_property_graph_index(dummy_index)

    # Helpers are attached as callables
    assert callable(extended.extract_entities)
    assert callable(extended.extract_relationships)
    assert callable(extended.traverse_graph)
    assert callable(extended.query)
