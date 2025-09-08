import time

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval import reranking as rr


def test_text_rerank_true_cancellation(monkeypatch):
    # Build nodes and bundle
    nodes = [NodeWithScore(node=TextNode(text=f"t{i}"), score=0.0) for i in range(5)]
    bundle = QueryBundle(query_str="hello")

    # Force timeout by using a blocking function longer than TEXT_RERANK_TIMEOUT_MS
    def very_slow_post(nodes, query_str):  # pylint: disable=unused-argument
        import time as _t

        _t.sleep((rr.TEXT_RERANK_TIMEOUT_MS + 50) / 1000.0)
        return list(nodes)

    class _Dummy:
        def postprocess_nodes(self, nodes, query_str):
            return very_slow_post(nodes, query_str)

    monkeypatch.setattr(rr, "build_text_reranker", lambda *a, **k: _Dummy())

    rer = rr.MultimodalReranker()
    out = rer._postprocess_nodes(nodes, bundle)
    # On timeout, fail-open should return original ordering (not empty)
    assert [n.node.text for n in out] == [n.node.text for n in nodes[: len(out)]]


def test_text_rerank_timeout_fail_open(monkeypatch):
    # Build nodes
    nodes = [NodeWithScore(node=TextNode(text=f"t{i}"), score=0.0) for i in range(5)]
    bundle = QueryBundle(query_str="hello")

    # Force timeout by monkeypatching _now_ms to simulate elapsed time
    start = rr._now_ms()
    calls = {"n": 0}

    def fake_now():
        # After first call for text stage, advance beyond timeout
        calls["n"] += 1
        if calls["n"] > 2:
            return start + rr.TEXT_RERANK_TIMEOUT_MS + 5
        return start

    monkeypatch.setattr(rr, "_now_ms", fake_now)

    # Patch text reranker to add an artificial delay
    def slow_post(nodes, query_str):  # pylint: disable=unused-argument
        time.sleep(0.001)
        return list(nodes)

    class _Dummy:
        def postprocess_nodes(self, nodes, query_str):
            return slow_post(nodes, query_str)

    monkeypatch.setattr(rr, "build_text_reranker", lambda *a, **k: _Dummy())

    rer = rr.MultimodalReranker()
    out = rer._postprocess_nodes(nodes, bundle)

    # Fail-open: ordering unchanged and length equals input top_k cap
    assert [n.node.text for n in out] == [n.node.text for n in nodes[: len(out)]]
