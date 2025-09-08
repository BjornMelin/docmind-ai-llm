from typing import Any

import numpy as np
from llama_index.core.schema import ImageNode, NodeWithScore, QueryBundle

from src.retrieval import reranking as rr


def test_siglip_rescore_orders_by_cosine(monkeypatch):
    # Prepare two image nodes with fake paths (not used by mock)
    a = NodeWithScore(
        node=ImageNode(image_path="/dev/null", metadata={"modality": "pdf_page_image"}),
        score=0.0,
    )
    b = NodeWithScore(
        node=ImageNode(image_path="/dev/null", metadata={"modality": "pdf_page_image"}),
        score=0.0,
    )
    nodes = [a, b]
    bundle = QueryBundle(query_str="cat")

    # Mock loader to avoid transformers/torch
    class _M:
        def get_text_features(self, **_: Any):
            # Unit vector
            return np.array([[1.0, 0.0]], dtype=np.float32)

        def get_image_features(self, **kwargs: Any):
            # Map first image closer to text, second farther
            return np.array([[1.0, 0.0], [0.1, 0.9]], dtype=np.float32)

    class _P:
        def __call__(self, *a: Any, **k: Any):
            return self

        def to(self, *_: Any, **__: Any):
            return self

        def __getitem__(self, k):
            return None

    monkeypatch.setattr(rr, "_load_siglip", lambda: (_M(), _P(), "cpu"))

    scored = rr._siglip_rescore(bundle.query_str, nodes, budget_ms=9999)
    # First node should be ahead after cosine
    assert scored[0] is a
