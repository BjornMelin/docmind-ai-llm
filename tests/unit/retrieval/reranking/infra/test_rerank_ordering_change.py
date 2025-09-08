from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval import reranking as rr


def test_rerank_changes_order_when_text_scores_differ(monkeypatch):
    # Input nodes in order a,b; reranker will swap
    a = NodeWithScore(node=TextNode(text="A"), score=0.1)
    b = NodeWithScore(node=TextNode(text="B"), score=0.2)
    nodes = [a, b]

    class _Dummy:
        def postprocess_nodes(self, nodes, query_str):
            return list(reversed(nodes))

    monkeypatch.setattr(rr, "build_text_reranker", lambda *a, **k: _Dummy())
    out = rr.MultimodalReranker()._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert out[0].node.text == "B"  # reversed order
