import pytest
from llama_index.core.schema import NodeWithScore, TextNode


@pytest.fixture
def nws_factory():
    """Factory for NodeWithScore test objects."""
    def _nws(nid: str, score: float = 0.0):
        node = TextNode(text="", id_=nid)
        return NodeWithScore(node=node, score=score)
    return _nws
