from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval.multimodal_fusion import (
    ImageSearchParams,
    ImageSiglipRetriever,
    MultimodalFusionRetriever,
)

pytestmark = pytest.mark.unit


def _nws(*, node_id: str, score: float, page_id: str) -> NodeWithScore:
    node = TextNode(text="x", id_=node_id)
    node.metadata["page_id"] = page_id
    return NodeWithScore(node=node, score=float(score))


def test_multimodal_fusion_dedups_by_key_and_prefers_best_score(monkeypatch) -> None:
    fused = [
        _nws(node_id="n1", score=0.1, page_id="p1"),
        _nws(node_id="n2", score=0.9, page_id="p1"),
        _nws(node_id="n3", score=0.2, page_id="p2"),
    ]

    class _Text:
        def retrieve(self, _q: str):  # type: ignore[no-untyped-def]
            return []

        def close(self) -> None:
            return None

    class _Img:
        def retrieve(self, _q: str):  # type: ignore[no-untyped-def]
            return []

        def close(self) -> None:
            return None

    # Force a deterministic fused list to exercise only this module's logic.
    monkeypatch.setitem(
        MultimodalFusionRetriever.retrieve.__globals__,
        "rrf_merge",
        lambda _xs, k_constant=60: fused,
    )

    mm = MultimodalFusionRetriever(
        text_retriever=_Text(),  # type: ignore[arg-type]
        image_retriever=_Img(),  # type: ignore[arg-type]
        fused_top_k=10,
        dedup_key="page_id",
    )
    out = mm.retrieve("q")
    assert [n.node.node_id for n in out] == ["n2", "n3"]


def test_image_siglip_retriever_queries_qdrant_and_wraps_results(monkeypatch) -> None:
    called = {"ensure": 0}

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        def query_points(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return SimpleNamespace(points=[])

        def close(self) -> None:
            return None

    class _Embed:
        def get_text_embedding(self, _q: str):  # type: ignore[no-untyped-def]
            return np.asarray([0.1, 0.2], dtype=np.float32)

    monkeypatch.setitem(
        ImageSiglipRetriever.retrieve.__globals__,
        "nodes_from_query_result",
        lambda _res, **_k: [_nws(node_id="n1", score=1.0, page_id="p1")],
    )
    monkeypatch.setattr(
        "src.retrieval.image_index.ensure_siglip_image_collection",
        lambda *_a, **_k: called.__setitem__("ensure", called["ensure"] + 1),
    )

    retr = ImageSiglipRetriever(
        ImageSearchParams(collection="img", top_k=3),
        client=_Client(),  # type: ignore[arg-type]
        embedder=_Embed(),  # type: ignore[arg-type]
    )
    out = retr.retrieve(QueryBundle(query_str="q"))
    assert len(out) == 1
    assert out[0].node.node_id == "n1"
    assert called["ensure"] == 1
