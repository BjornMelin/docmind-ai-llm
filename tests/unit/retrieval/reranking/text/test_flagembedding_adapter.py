"""Exercise FlagEmbedding adapter path in text reranker.

Injects a dummy FlagEmbedding module and asserts its compute_score is called.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace


def test_flagembedding_adapter_invoked(monkeypatch):
    # Create dummy FlagEmbedding module
    dummy = ModuleType("FlagEmbedding")
    calls = []

    class FlagReranker:
        def __init__(self, model_id, use_fp16=False):
            self.model_id = model_id
            self.use_fp16 = use_fp16

        def compute_score(self, pairs):
            calls.append(list(pairs))
            # Return a simple increasing score
            return [float(i + 1) for i in range(len(pairs))]

    dummy.FlagReranker = FlagReranker
    sys.modules["FlagEmbedding"] = dummy

    rr = importlib.import_module("src.retrieval.reranking")
    # Build reranker; should choose FlagEmbedding backend
    adapter = rr.build_text_reranker(top_n=3)

    # Prepare nodes
    nodes = []
    for i in range(4):
        node = SimpleNamespace(node_id=f"n{i}", text=f"text {i}", metadata={})
        nodes.append(SimpleNamespace(node=node, score=0.0))

    out = adapter.postprocess_nodes(nodes, query_str="q")
    # ensure dummy backend was used
    assert calls, "FlagEmbedding backend not invoked"
    assert len(out) == 3, "top_n not applied"
