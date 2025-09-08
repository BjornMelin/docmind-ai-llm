"""Error-path coverage for text reranker backends.

Covers LI CrossEncoder and FlagEmbedding fall-open branches.
"""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval import reranking as rr


def _mk_nodes(n: int) -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text=f"t-{i}"), score=0.0) for i in range(n)]


def test_li_backend_batch_error_fall_open(monkeypatch):
    """When LI backend raises, adapter returns previous scores (fall-open)."""

    class _BadLI:
        top_n = 0

        def postprocess_nodes(self, *_a, **_k):  # pylint: disable=unused-argument
            raise RuntimeError("boom")

    # Force LI path selection
    monkeypatch.setattr(rr, "_build_text_reranker_cached", lambda top_n: _BadLI())
    # Also ensure FlagEmbedding import fails by shadowing __import__ in rr module
    real_import = __import__

    def fake_import(name, *a, **k):
        if name == "FlagEmbedding":
            raise ImportError("no flagembedding")
        return real_import(name, *a, **k)

    monkeypatch.setattr(rr, "__import__", fake_import, raising=False)

    adapter = rr.build_text_reranker(top_n=5, batch_size=3, timeout_ms=9999)
    nodes = _mk_nodes(5)
    out = adapter.postprocess_nodes(nodes, query_str="q")
    # Fall-open keeps previous (zero) scores and stable order by node_id ascending
    assert len(out) == 5
    assert all(getattr(n, "score", 0.0) == 0.0 for n in out)


def test_flag_backend_batch_error_fall_open(monkeypatch):
    """When FlagEmbedding raises, adapter returns previous scores (fall-open)."""

    class _FakeFlag:
        def ensure_loaded(self):
            return None

        def score_pairs(self, _pairs):
            raise RuntimeError("flag error")

    # Build any adapter, then force flag path
    adapter = rr.build_text_reranker(top_n=4, batch_size=2, timeout_ms=9999)
    adapter._flag = _FakeFlag()  # type: ignore[attr-defined]
    adapter._li = None  # type: ignore[attr-defined]

    nodes = _mk_nodes(4)
    out = adapter.postprocess_nodes(nodes, query_str="q")
    assert len(out) == 4
    assert all(getattr(n, "score", 0.0) == 0.0 for n in out)
