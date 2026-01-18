"""Unit tests for the analysis service orchestration."""

from __future__ import annotations

import threading

import pytest

from src.analysis.models import DocumentRef
from src.analysis.service import AnalysisCancelledError, auto_select_mode, run_analysis
from tests.fixtures.test_settings import create_test_settings
from tests.fixtures.vector_index import _FakeVectorIndex

pytestmark = pytest.mark.unit


def test_auto_select_mode_prefers_separate_for_small_sets() -> None:
    """Auto-selects separate mode for small document sets.

    Args:
        None.

    Returns:
        None.
    """
    cfg = create_test_settings()
    cfg.analysis.max_workers = 4
    assert auto_select_mode(doc_count=2, cfg=cfg)[0] == "separate"
    assert auto_select_mode(doc_count=10, cfg=cfg)[0] == "combined"


def test_run_analysis_combined_mode() -> None:
    """Runs analysis in combined mode and returns a single response.

    Args:
        None.

    Returns:
        None.
    """
    cfg = create_test_settings()
    result = run_analysis(
        query="q",
        mode="combined",
        vector_index=_FakeVectorIndex(),
        documents=[],
        cfg=cfg,
    )
    assert result.mode == "combined"
    assert result.combined
    assert result.combined.startswith("answer:")
    assert result.per_doc == []


def test_run_analysis_separate_mode() -> None:
    """Runs analysis in separate mode and returns per-document results.

    Args:
        None.

    Returns:
        None.
    """
    cfg = create_test_settings()
    docs = [
        DocumentRef(doc_id="doc-a", doc_name="a.txt"),
        DocumentRef(doc_id="doc-b", doc_name="b.txt"),
    ]
    result = run_analysis(
        query="q",
        mode="separate",
        vector_index=_FakeVectorIndex(),
        documents=docs,
        cfg=cfg,
    )
    assert result.mode == "separate"
    assert len(result.per_doc) == 2
    answers = {r.answer for r in result.per_doc}
    assert "answer:doc-a" in answers
    assert "answer:doc-b" in answers


def test_run_analysis_respects_cancellation() -> None:
    """Raises when the analysis is canceled before execution.

    Args:
        None.

    Returns:
        None.
    """
    cfg = create_test_settings()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(AnalysisCancelledError):
        run_analysis(
            query="q",
            mode="combined",
            vector_index=_FakeVectorIndex(),
            documents=[],
            cfg=cfg,
            cancel_event=cancel,
        )
