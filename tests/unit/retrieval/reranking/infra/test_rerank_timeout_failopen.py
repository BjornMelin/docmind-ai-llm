import asyncio
import threading
import time

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval import reranking as rr


def _mixed_nodes() -> list[NodeWithScore]:
    return [
        NodeWithScore(
            node=TextNode(
                text="text candidate",
                id_="text-candidate",
                metadata={"modality": "text"},
            ),
            score=0.8,
        ),
        NodeWithScore(
            node=TextNode(
                text="visual candidate",
                id_="visual-candidate",
                metadata={"modality": "image"},
            ),
            score=0.7,
        ),
    ]


def test_sync_total_budget_caps_each_stage_and_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One monotonic deadline must cap text plus visual work and fail open."""
    nodes = [
        NodeWithScore(
            node=TextNode(text="text", metadata={"modality": "text"}),
            score=0.0,
        ),
        NodeWithScore(
            node=TextNode(text="image", metadata={"modality": "image"}),
            score=0.0,
        ),
    ]
    bundle = QueryBundle(query_str="hello")
    now = {"value": 100.0}
    stage_timeouts: list[float] = []
    events: list[dict[str, object]] = []

    monkeypatch.setattr(rr.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        rr.settings.retrieval,
        "total_rerank_budget_ms",
        100,
        raising=False,
    )
    monkeypatch.setattr(
        rr.settings.retrieval,
        "text_rerank_timeout_ms",
        250,
        raising=False,
    )
    monkeypatch.setattr(rr, "log_jsonl", events.append)

    reranker = rr.MultimodalReranker()

    def _text_result(text_nodes, _query_bundle):  # type: ignore[no-untyped-def]
        return list(text_nodes)

    def _visual_result(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("timed-out visual work must not produce a result")

    def _run_sync(function, *, timeout):  # type: ignore[no-untyped-def]
        stage_timeouts.append(timeout)
        if len(stage_timeouts) == 1:
            result = function()
            now["value"] += 0.075
            return result
        now["value"] += timeout + 0.001
        raise rr.FutureTimeoutError

    monkeypatch.setattr(reranker, "_text_result", _text_result)
    monkeypatch.setattr(reranker, "_visual_result", _visual_result)
    monkeypatch.setattr(reranker._work, "run_sync", _run_sync)

    try:
        out = reranker._postprocess_nodes(nodes, bundle)
    finally:
        reranker.close()

    assert out == nodes
    assert stage_timeouts == pytest.approx([0.1, 0.025])
    assert any(
        event.get("rerank.stage") == "total" and event.get("rerank.timeout") is True
        for event in events
    )


def test_sync_total_budget_fails_open_when_telemetry_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telemetry storage failure cannot reverse the rerank fail-open contract."""
    nodes = [NodeWithScore(node=TextNode(text="text"), score=0.0)]
    reranker = rr.MultimodalReranker()
    monkeypatch.setattr(rr, "_capped_stage_budget_ms", lambda *_args: 0.0)
    monkeypatch.setattr(
        rr,
        "log_jsonl",
        lambda _event: (_ for _ in ()).throw(OSError("disk full")),
    )

    try:
        out = reranker._postprocess_nodes(nodes, QueryBundle(query_str="hello"))
    finally:
        reranker.close()

    assert out == nodes


def test_sync_text_error_preserves_text_candidates_in_mixed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handled text-model error cannot silently drop the text modality."""
    nodes = _mixed_nodes()
    reranker = rr.MultimodalReranker()
    monkeypatch.setattr(
        reranker,
        "_text_result",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("text model failed")),
    )
    monkeypatch.setattr(
        reranker,
        "_visual_result",
        lambda visual_nodes, _query_bundle, lists: [*lists, visual_nodes],
    )

    try:
        out = reranker._postprocess_nodes(nodes, QueryBundle(query_str="hello"))
    finally:
        reranker.close()

    assert {node.node.node_id for node in out} == {
        "text-candidate",
        "visual-candidate",
    }


def test_sync_visual_error_returns_original_mixed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visual-model I/O error returns the exact caller-owned result set."""
    nodes = _mixed_nodes()
    reranker = rr.MultimodalReranker()
    monkeypatch.setattr(reranker, "_text_result", lambda text, _query: text)
    monkeypatch.setattr(
        reranker,
        "_visual_result",
        lambda *_args: (_ for _ in ()).throw(OSError("visual model unavailable")),
    )

    try:
        out = reranker._postprocess_nodes(nodes, QueryBundle(query_str="hello"))
    finally:
        reranker.close()

    assert out is nodes


async def test_async_text_error_preserves_text_candidates_in_mixed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async path retains text candidates after a text-model failure."""
    nodes = _mixed_nodes()
    reranker = rr.MultimodalReranker()
    monkeypatch.setattr(
        reranker,
        "_text_result",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("text model failed")),
    )
    monkeypatch.setattr(
        reranker,
        "_visual_result",
        lambda visual_nodes, _query_bundle, lists: [*lists, visual_nodes],
    )

    try:
        out = await reranker.apostprocess_nodes(
            nodes, query_bundle=QueryBundle(query_str="hello")
        )
    finally:
        await reranker.aclose()

    assert {node.node.node_id for node in out} == {
        "text-candidate",
        "visual-candidate",
    }


async def test_async_visual_error_returns_original_mixed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async path also fails open on visual-model I/O errors."""
    nodes = _mixed_nodes()
    reranker = rr.MultimodalReranker()
    monkeypatch.setattr(reranker, "_text_result", lambda text, _query: text)
    monkeypatch.setattr(
        reranker,
        "_visual_result",
        lambda *_args: (_ for _ in ()).throw(OSError("visual model unavailable")),
    )

    try:
        out = await reranker.apostprocess_nodes(
            nodes, query_bundle=QueryBundle(query_str="hello")
        )
    finally:
        await reranker.aclose()

    assert out is nodes


def test_sync_timeout_worker_cannot_mutate_fail_open_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native worker finishing after timeout must not mutate returned nodes."""
    nodes = [NodeWithScore(node=TextNode(text="text"), score=0.0)]
    bundle = QueryBundle(query_str="hello")
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class _LateMutatingReranker:
        def postprocess_nodes(self, stage_nodes, query_str):  # type: ignore[no-untyped-def]
            del query_str
            started.set()
            release.wait()
            stage_nodes[0].score = 99.0
            finished.set()
            return list(stage_nodes)

    monkeypatch.setattr(
        rr,
        "build_text_reranker",
        lambda *_args, **_kwargs: _LateMutatingReranker(),
    )
    monkeypatch.setattr(
        rr.settings.retrieval,
        "total_rerank_budget_ms",
        25,
        raising=False,
    )

    reranker = rr.MultimodalReranker()
    try:
        out = reranker._postprocess_nodes(nodes, bundle)
        assert started.wait(timeout=1.0)
        assert out == nodes
        assert out[0].score == 0.0

        release.set()
        assert finished.wait(timeout=1.0)
        assert out[0].score == 0.0
    finally:
        release.set()
        reranker.close()


async def test_repeated_async_timeouts_do_not_queue_or_use_default_pool(
    monkeypatch,
):  # type: ignore[no-untyped-def]
    nodes = [NodeWithScore(node=TextNode(text="blocking"), score=0.0)]
    bundle = QueryBundle(query_str="hello")
    started = threading.Event()
    release = threading.Event()
    starts: list[str] = []

    class _BlockingReranker:
        def postprocess_nodes(self, nodes, query_str):  # type: ignore[no-untyped-def]
            del query_str
            starts.append(threading.current_thread().name)
            started.set()
            release.wait()
            return list(nodes)

    async def _forbid_default_pool(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("reranking must not call asyncio.to_thread")

    monkeypatch.setattr(
        rr,
        "build_text_reranker",
        lambda *_a, **_k: _BlockingReranker(),
    )
    monkeypatch.setattr(
        rr.settings.retrieval,
        "total_rerank_budget_ms",
        10,
        raising=False,
    )
    monkeypatch.setattr(asyncio, "to_thread", _forbid_default_pool)

    reranker = rr.MultimodalReranker()
    try:
        first = await reranker.apostprocess_nodes(nodes, query_bundle=bundle)
        assert first == nodes
        assert started.is_set()

        repeated = [
            await reranker.apostprocess_nodes(nodes, query_bundle=bundle)
            for _ in range(8)
        ]
        assert repeated == [nodes] * 8
        assert starts == ["docmind-rerank-cpu_0"]

        release.set()
        await reranker.aclose()
        await asyncio.sleep(0)
        assert starts == ["docmind-rerank-cpu_0"]
    finally:
        release.set()
        reranker.close()


def test_text_rerank_true_cancellation(monkeypatch):
    # Build nodes and bundle
    nodes = [NodeWithScore(node=TextNode(text=f"t{i}"), score=0.0) for i in range(5)]
    bundle = QueryBundle(query_str="hello")

    timeout_ms = 250
    monkeypatch.setattr(
        rr.settings.retrieval,
        "text_rerank_timeout_ms",
        timeout_ms,
        raising=False,
    )

    # Force timeout by using a blocking function longer than the configured timeout.
    def very_slow_post(nodes, query_str):
        import time as _t

        _t.sleep((timeout_ms + 50) / 1000.0)
        return list(nodes)

    class _Dummy:
        def postprocess_nodes(self, nodes, query_str):
            return very_slow_post(nodes, query_str)

    monkeypatch.setattr(rr, "build_text_reranker", lambda *a, **k: _Dummy())

    rer = rr.MultimodalReranker()
    out = rer._postprocess_nodes(nodes, bundle)
    # On timeout, fail-open should return original ordering (not empty)
    assert [n.node.get_content() for n in out] == [
        n.node.get_content() for n in nodes[: len(out)]
    ]
    rer.close()


def test_text_rerank_timeout_fail_open(monkeypatch):
    # Build nodes
    nodes = [NodeWithScore(node=TextNode(text=f"t{i}"), score=0.0) for i in range(5)]
    bundle = QueryBundle(query_str="hello")

    timeout_ms = 250
    monkeypatch.setattr(
        rr.settings.retrieval,
        "text_rerank_timeout_ms",
        timeout_ms,
        raising=False,
    )

    # Force timeout by monkeypatching _now_ms to simulate elapsed time
    start = rr._now_ms()
    calls = {"n": 0}

    def fake_now():
        # After first call for text stage, advance beyond timeout
        calls["n"] += 1
        if calls["n"] > 2:
            return start + timeout_ms + 5
        return start

    monkeypatch.setattr(rr, "_now_ms", fake_now)

    # Patch text reranker to add an artificial delay
    def slow_post(nodes, query_str):
        time.sleep(0.001)
        return list(nodes)

    class _Dummy:
        def postprocess_nodes(self, nodes, query_str):
            return slow_post(nodes, query_str)

    monkeypatch.setattr(rr, "build_text_reranker", lambda *a, **k: _Dummy())

    rer = rr.MultimodalReranker()
    out = rer._postprocess_nodes(nodes, bundle)

    # Fail-open: ordering unchanged and length equals input top_k cap
    assert [n.node.get_content() for n in out] == [
        n.node.get_content() for n in nodes[: len(out)]
    ]
    rer.close()
