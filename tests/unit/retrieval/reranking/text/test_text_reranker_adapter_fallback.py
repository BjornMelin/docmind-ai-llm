"""Adapter fallback: FlagEmbedding missing -> LI wrapper; cancellation works.

We stub the LI cached builder to avoid heavy deps and assert cooperative
cancellation stats are emitted even when using the fallback path.
"""

from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval import reranking as rr


class _FakeLI:
    def __init__(self, top_n: int) -> None:
        self.top_n = int(top_n)

    def postprocess_nodes(self, nodes, query_str):
        # Assign deterministic scores from text suffix number
        out = []
        for n in nodes:
            try:
                idx = int(str(getattr(n.node, "text", "0")).split("-")[-1])
            except Exception:
                idx = 0
            n.score = float(idx)
            out.append(n)
        # Return the original nodes with scores set
        return out


def test_fallback_li_adapter_cancellation(monkeypatch):
    # Force ImportError for FlagEmbedding only by shadowing __import__ in rr module
    real_import = __import__

    def fake_import(name, *a, **k):
        if name == "FlagEmbedding":
            raise ImportError("no flagembedding")
        return real_import(name, *a, **k)

    monkeypatch.setattr(rr, "__import__", fake_import, raising=False)

    # Stub LI builder
    monkeypatch.setattr(rr, "_build_text_reranker_cached", lambda top_n: _FakeLI(top_n))

    nodes = [NodeWithScore(node=TextNode(text=f"t-{i}"), score=0.0) for i in range(10)]

    start = rr._now_ms()
    calls = {"n": 0}

    def fake_now():
        calls["n"] += 1
        # After one batch (size 4), exceed budget
        return start if calls["n"] <= 5 else (start + rr.TEXT_RERANK_TIMEOUT_MS + 1)

    monkeypatch.setattr(rr, "_now_ms", fake_now)

    adapter = rr.build_text_reranker(
        top_n=10, batch_size=4, timeout_ms=rr.TEXT_RERANK_TIMEOUT_MS
    )
    out = adapter.postprocess_nodes(nodes, query_str="q")

    # Expect only first batch processed and returned (size 4)
    assert len(out) == 4
    assert adapter.last_stats["timeout"] is True
    assert adapter.last_stats["processed_batches"] == 1
