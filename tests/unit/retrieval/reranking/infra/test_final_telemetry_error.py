"""Final telemetry error path: log_jsonl raises during final stage.

MultimodalReranker should handle telemetry errors without crashing.
"""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval import reranking as rr


def test_final_telemetry_log_error_is_caught(monkeypatch):
    # Simple text-only adapter that returns reversed nodes
    class _TextOnly:
        def postprocess_nodes(self, nodes, query_str):
            return list(reversed(nodes))

    # Patch builder
    monkeypatch.setattr(rr, "build_text_reranker", lambda *a, **k: _TextOnly())

    # Only raise on final stage
    def _log(ev):
        if ev.get("rerank.stage") == "final":
            raise RuntimeError("telemetry failure")
        return None

    monkeypatch.setattr(rr, "log_jsonl", _log)

    nodes = [
        NodeWithScore(node=TextNode(text="A"), score=0.1),
        NodeWithScore(node=TextNode(text="B"), score=0.2),
    ]

    out = rr.MultimodalReranker()._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    # Still returns correct ordering and result despite telemetry failure
    assert [n.node.text for n in out] == ["B", "A"]
