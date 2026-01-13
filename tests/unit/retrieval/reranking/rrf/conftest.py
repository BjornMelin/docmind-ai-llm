from collections.abc import Callable

import pytest
from llama_index.core.schema import NodeWithScore, TextNode


@pytest.fixture
def nws_factory() -> Callable[[str, float], NodeWithScore]:
    """Factory for NodeWithScore test objects."""

    def _nws(nid: str, score: float = 0.0) -> NodeWithScore:
        node = TextNode(text="", id_=nid)
        return NodeWithScore(node=node, score=score)

    return _nws
