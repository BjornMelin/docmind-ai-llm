"""Changing batch size must not alter ordering under ample budget."""

from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval import reranking as rr


def _mk_nodes(n: int) -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text=f"t-{i}"), score=0.0) for i in range(n)]


def test_batch_size_independence(monkeypatch):
    class _Model:
        def compute_score(self, pairs):
            # Global, content-derived score independent of grouping
            return [float(int(str(t).split("-")[-1])) for _q, t in pairs]

    def ensure_loaded(self):
        self._model = _Model()  # type: ignore[attr-defined]

    monkeypatch.setattr(rr._FlagTextReranker, "_ensure_loaded", ensure_loaded)

    nodes = _mk_nodes(25)
    base = rr.build_text_reranker(top_n=25, batch_size=2, timeout_ms=9999)
    out_a = base.postprocess_nodes(list(nodes), query_str="q")

    alt = rr.build_text_reranker(top_n=25, batch_size=7, timeout_ms=9999)
    out_b = alt.postprocess_nodes(list(nodes), query_str="q")

    assert [n.node.node_id for n in out_a] == [n.node.node_id for n in out_b]
