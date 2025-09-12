"""Test agent-layer hybrid fallback to vector when hybrid returns empty."""

from src.agents.tools import retrieval as rmod


class _FakeTool:
    def __init__(self, docs):
        self._docs = docs

    def call(self, _q):
        return self._docs


class _FakeFactory:
    def create_hybrid_search_tool(self, _retr):
        return _FakeTool([])  # empty to force fallback

    def create_vector_search_tool(self, _idx):
        return _FakeTool([{"content": "v"}])

    def create_hybrid_vector_tool(self, _idx):
        return _FakeTool([])


def test_hybrid_fallback_to_vector(monkeypatch):
    monkeypatch.setattr(rmod, "_get_tool_factory", lambda: _FakeFactory())
    docs, strategy, err = rmod._run_vector_hybrid(
        strategy="hybrid",
        retriever=object(),
        vector_index=object(),
        queries=["q"],
        primary_query="q",
    )
    assert err is None
    assert isinstance(docs, list)
    assert strategy == "vector"
    assert len(docs) > 0
