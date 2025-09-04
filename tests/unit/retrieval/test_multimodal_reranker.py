"""Minimal tests for MultimodalReranker wiring.

These tests validate that the class imports and basic no-op behavior
does not raise, keeping unit scope light.
"""

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval.reranking import MultimodalReranker


def test_multimodal_reranker_no_query_bundle_returns_input() -> None:
    """Test MultimodalReranker returns input unchanged without query bundle."""
    reranker = MultimodalReranker()
    nodes = [NodeWithScore(node=TextNode(text="a", id_="a"), score=0.1)]
    out = reranker.postprocess_nodes(nodes, None)
    assert out == nodes


def test_multimodal_reranker_handles_empty_nodes() -> None:
    """Test MultimodalReranker handles empty node list correctly."""
    reranker = MultimodalReranker()
    qb = QueryBundle(query_str="q")
    out = reranker.postprocess_nodes([], qb)
    assert out == []
