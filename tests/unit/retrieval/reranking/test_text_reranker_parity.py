"""Determinism: ample budget yields consistent ordering across runs."""

from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval import reranking as rr


def _mk_nodes(n: int) -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text=f"t-{i}"), score=0.0) for i in range(n)]


def test_parity_across_runs_with_ample_budget(monkeypatch):
    # Backend scores based on text suffix number, independent of batching
    class _FastModel:
        def compute_score(self, pairs):  # pairs: [[q, text], ...]
            out = []
            for _q, text in pairs:
                try:
                    idx = int(str(text).split("-")[-1])
                except Exception:
                    idx = 0
                out.append(float(idx))
            return out

    def ensure_loaded(self):
        self._model = _FastModel()  # type: ignore[attr-defined]

    monkeypatch.setattr(rr._FlagTextReranker, "_ensure_loaded", ensure_loaded)

    # First run
    nodes1 = _mk_nodes(12)
    a = rr.build_text_reranker(top_n=12, batch_size=3, timeout_ms=10_000)
    out1 = a.postprocess_nodes(nodes1, query_str="q")

    # Second run with different batch size; should be identical ordering
    nodes2 = _mk_nodes(12)
    b = rr.build_text_reranker(top_n=12, batch_size=5, timeout_ms=10_000)
    out2 = b.postprocess_nodes(nodes2, query_str="q")

    # Compare by text content to avoid Node ID variability across LI versions
    texts1 = [getattr(n.node, "text", "") for n in out1]
    texts2 = [getattr(n.node, "text", "") for n in out2]
    assert texts1 == texts2
