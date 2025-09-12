"""Tests for postprocessor_utils engine builders.

Covers fallback behavior when node_postprocessors are unsupported.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def test_build_vector_query_engine_fallback():  # type: ignore[no-untyped-def]
    m = importlib.import_module("src.retrieval.postprocessor_utils")

    class _Vec:
        # Deliberately restrict signature so unexpected kwargs raise TypeError
        def as_query_engine(self, similarity_top_k=None):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                engine=True, kwargs={"similarity_top_k": similarity_top_k}
            )

    eng = m.build_vector_query_engine(_Vec(), post=[object()], similarity_top_k=3)
    assert getattr(eng, "engine", False) is True
    assert "node_postprocessors" not in eng.kwargs


def test_build_retriever_query_engine_fallback():  # type: ignore[no-untyped-def]
    m = importlib.import_module("src.retrieval.postprocessor_utils")

    class _Ret:
        pass

    class _RQE:
        @classmethod
        def from_args(
            cls,
            *,
            retriever,  # type: ignore[no-untyped-def]
            llm=None,  # type: ignore[no-untyped-def]
            response_mode=None,  # type: ignore[no-untyped-def]
            verbose=None,  # type: ignore[no-untyped-def]
        ):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                engine=True,
                kwargs={
                    "retriever": retriever,
                    "llm": llm,
                    "response_mode": response_mode,
                    "verbose": verbose,
                },
            )

    eng = m.build_retriever_query_engine(
        retriever=_Ret(), post=[object()], engine_cls=_RQE
    )
    assert getattr(eng, "engine", False) is True
    assert "node_postprocessors" not in eng.kwargs
