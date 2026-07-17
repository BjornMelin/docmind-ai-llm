from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from qdrant_client import models as qmodels

from src.config import settings
from src.retrieval import multimodal_fusion as multimodal_module
from src.retrieval import vector_contract
from src.retrieval.image_index import (
    ImageCollectionIncompatibleError,
    canonical_image_collection_metadata,
)
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


def test_multimodal_constructor_does_not_leak_text_retriever_on_image_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release internally acquired resources when optional image setup fails."""

    class _Text:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    text = _Text()

    def _fail_image(_params: ImageSearchParams) -> None:
        raise ImageCollectionIncompatibleError("image_collection_missing")

    monkeypatch.setattr(vector_contract, "sparse_retrieval_enabled", lambda: True)
    monkeypatch.setattr(multimodal_module, "ServerHybridRetriever", lambda _p: text)
    monkeypatch.setattr(multimodal_module, "ImageSiglipRetriever", _fail_image)

    with pytest.raises(ImageCollectionIncompatibleError):
        MultimodalFusionRetriever(text_collection="text", image_collection="images")

    assert text.closed is True


@pytest.mark.parametrize(
    ("fused_top_k", "expected_ids"),
    [(10, ["n2", "n3"]), (1, ["n2"])],
)
def test_multimodal_fusion_dedups_by_key_and_prefers_best_score(
    monkeypatch,
    fused_top_k: int,
    expected_ids: list[str],
) -> None:
    fused = [
        _nws(node_id="n1", score=0.1, page_id="p1"),
        _nws(node_id="n2", score=0.9, page_id="p1"),
        _nws(node_id="n3", score=0.2, page_id="p2"),
    ]

    class _Text:
        def retrieve(self, _q: str):  # type: ignore[no-untyped-def]
            return []

        async def aretrieve(self, _q: str):  # type: ignore[no-untyped-def]
            return []

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    class _Img:
        def retrieve(self, _q: str):  # type: ignore[no-untyped-def]
            return []

        async def aretrieve(self, _q: str):  # type: ignore[no-untyped-def]
            return []

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    # Force a deterministic fused list to exercise only this module's logic.
    monkeypatch.setitem(
        MultimodalFusionRetriever._fuse.__globals__,
        "rrf_merge",
        lambda _xs, k_constant=60: fused,
    )
    events: list[dict[str, object]] = []
    monkeypatch.setitem(
        MultimodalFusionRetriever._fuse.__globals__, "log_jsonl", events.append
    )

    mm = MultimodalFusionRetriever(
        text_retriever=_Text(),  # type: ignore[arg-type]
        image_retriever=_Img(),  # type: ignore[arg-type]
        fused_top_k=fused_top_k,
        dedup_key="page_id",
    )
    out = mm.retrieve("q")
    assert [n.node.node_id for n in out] == expected_ids
    assert events
    event = events[-1]
    assert event["dedup.key"] == "page_id"
    assert event["dedup.before"] == 3
    assert event["dedup.after"] == 2
    assert event["dedup.dropped"] == 1


async def test_multimodal_fusion_uses_native_async_retriever_contract() -> None:
    text_node = _nws(node_id="text", score=1.0, page_id="p1")
    image_node = _nws(node_id="image", score=1.0, page_id="p2")

    class _Text:
        def retrieve(self, _q: str) -> list[NodeWithScore]:
            return [text_node]

        async def aretrieve(self, _q: str) -> list[NodeWithScore]:
            return [text_node]

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    class _Image:
        def retrieve(self, _q: str) -> list[NodeWithScore]:
            return [image_node]

        async def aretrieve(self, _q: str) -> list[NodeWithScore]:
            return [image_node]

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    retriever = MultimodalFusionRetriever(
        text_retriever=_Text(),  # type: ignore[arg-type]
        image_retriever=_Image(),  # type: ignore[arg-type]
        fused_top_k=2,
    )

    assert [node.node.node_id for node in await retriever.aretrieve("query")] == [
        "image",
        "text",
    ]


async def test_multimodal_retrieval_is_not_capped_by_reranker_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_node = _nws(node_id="text", score=1.0, page_id="p1")
    image_node = _nws(node_id="image", score=1.0, page_id="p2")

    class _Slow:
        def __init__(self, node: NodeWithScore) -> None:
            self.node = node

        def retrieve(self, _query: str) -> list[NodeWithScore]:
            return [self.node]

        async def aretrieve(self, _query: str) -> list[NodeWithScore]:
            await asyncio.sleep(0.01)
            return [self.node]

        def close(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    text = _Slow(text_node)
    image = _Slow(image_node)
    retriever = MultimodalFusionRetriever(
        text_retriever=text,  # type: ignore[arg-type]
        image_retriever=image,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(settings.retrieval, "text_rerank_timeout_ms", 1)
    monkeypatch.setattr(settings.retrieval, "siglip_timeout_ms", 1)

    assert [node.node.node_id for node in await retriever.aretrieve("query")] == [
        "image",
        "text",
    ]


def test_image_siglip_retriever_queries_qdrant_and_wraps_results(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.calls = 0
            self.create_calls = 0

        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            params = SimpleNamespace(
                vectors={
                    "siglip": qmodels.VectorParams(
                        size=768,
                        distance=qmodels.Distance.COSINE,
                    )
                }
            )
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=params,
                    metadata=canonical_image_collection_metadata(dim=768),
                )
            )

        def create_collection(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self.create_calls += 1

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
    client = _Client()
    retr = ImageSiglipRetriever(
        ImageSearchParams(collection="img", top_k=3),
        client=client,  # type: ignore[arg-type]
        embedder=_Embed(),  # type: ignore[arg-type]
    )
    out = retr.retrieve(QueryBundle(query_str="q"))
    assert len(out) == 1
    assert out[0].node.node_id == "n1"
    assert client.create_calls == 0
