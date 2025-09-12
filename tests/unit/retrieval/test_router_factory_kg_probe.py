"""Router factory KG probe behavior tests.

Ensure that zero-result probe does not disable the knowledge_graph tool.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.retrieval.router_factory import build_router_engine


class _VecIndex:
    def as_query_engine(self, **_kwargs: Any) -> Any:  # pragma: no cover - trivial
        return object()


class _PGIndex:
    def __init__(self, probe_empty: bool = True) -> None:
        self.property_graph_store = object()
        self._probe_empty = probe_empty

    def as_retriever(self, **_kwargs: Any) -> Any:  # pragma: no cover - trivial
        class _R:
            def __init__(self, empty: bool) -> None:
                self._empty = empty

            def retrieve(self, _q: str) -> list[Any]:  # pragma: no cover - trivial
                return [] if self._empty else [1]

        return _R(self._probe_empty)


@pytest.mark.unit
def test_router_registers_kg_tool_even_on_empty_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Minimal settings shape for build_router_engine
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        retrieval=SimpleNamespace(
            top_k=3,
            use_reranking=False,
            enable_server_hybrid=False,
            reranking_top_k=3,
        ),
        database=SimpleNamespace(qdrant_collection="col"),
        graphrag_cfg=SimpleNamespace(default_path_depth=1),
    )

    # Patch get_postprocessors to return empty post list
    import src.retrieval.reranking as rr

    monkeypatch.setattr(rr, "get_postprocessors", lambda *_a, **_k: [], raising=True)

    vec = _VecIndex()
    pg = _PGIndex(probe_empty=True)
    engine = build_router_engine(vec, pg_index=pg, settings=cfg, llm=None)
    tool_list = getattr(engine, "query_engine_tools", [])
    tool_names = [t.metadata.name for t in tool_list]
    assert "knowledge_graph" in tool_names
