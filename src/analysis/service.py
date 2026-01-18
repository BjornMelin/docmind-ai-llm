"""Document analysis mode service (SPEC-036).

Implements separate/combined/auto analysis flows using the existing LlamaIndex
vector index. The service is Streamlit-agnostic and safe for background worker
threads (no Streamlit API usage).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger

from src.analysis.models import (
    AnalysisMode,
    AnalysisResult,
    DocumentRef,
    PerDocResult,
    ResolvedAnalysisMode,
)
from src.config.settings import DocMindSettings, settings
from src.processing.ingestion_api import generate_stable_id
from src.utils.log_safety import build_pii_log_entry
from src.utils.telemetry import log_jsonl


class AnalysisCancelledError(RuntimeError):
    """Raised when cooperative cancellation is requested."""


def discover_uploaded_documents(uploads_dir: Path) -> list[DocumentRef]:
    """Discover uploaded documents and compute deterministic doc ids.

    Args:
        uploads_dir: Directory containing uploaded files.

    Returns:
        List of documents with stable ids and display names.
    """
    try:
        if not uploads_dir.exists():
            return []
    except OSError:
        return []

    docs: list[DocumentRef] = []
    for path in sorted(uploads_dir.glob("*"), key=lambda p: p.name):
        if not path.is_file():
            continue
        try:
            doc_id = generate_stable_id(path)
        except Exception as exc:  # pragma: no cover - best-effort
            redaction = build_pii_log_entry(str(exc), key_id="analysis.discover_doc_id")
            logger.debug(
                "Skipping document '{}' (error_type={}, error={})",
                path.name,
                type(exc).__name__,
                redaction.redacted,
            )
            continue
        docs.append(DocumentRef(doc_id=str(doc_id), doc_name=str(path.name)))
    return docs


def auto_select_mode(
    *, doc_count: int, cfg: DocMindSettings
) -> tuple[ResolvedAnalysisMode, str]:
    """Select an analysis mode when requested mode is ``auto``.

    Args:
        doc_count: Number of documents selected for analysis.
        cfg: Settings controlling analysis concurrency.

    Returns:
        Tuple of (resolved_mode, reason) for auto-selection.
    """
    if doc_count <= 0:
        return "combined", "no_docs_selected"
    if doc_count == 1:
        return "combined", "single_doc_selected"
    if doc_count <= 3 and int(cfg.analysis.max_workers) >= 2:
        return "separate", "small_doc_set"
    return "combined", "too_many_docs"


def resolve_mode(
    requested: AnalysisMode,
    *,
    doc_count: int,
    cfg: DocMindSettings,
) -> tuple[ResolvedAnalysisMode, str | None]:
    """Resolve the effective analysis mode and optional auto decision reason.

    Args:
        requested: Requested analysis mode (auto/separate/combined).
        doc_count: Number of documents selected for analysis.
        cfg: Settings controlling analysis concurrency.

    Returns:
        Tuple of (resolved_mode, auto_reason). Reason is None when not auto.
    """
    if requested == "auto":
        mode, reason = auto_select_mode(doc_count=doc_count, cfg=cfg)
        return mode, reason
    return requested, None  # type: ignore[return-value]


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelledError()


def _safe_log_jsonl(payload: dict[str, Any], *, key_id: str) -> None:
    try:
        log_jsonl(payload)
    except Exception as exc:  # pragma: no cover - best effort
        redaction = build_pii_log_entry(str(exc), key_id=key_id)
        logger.debug(
            "telemetry skipped (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )


def _clamp_pct(pct: int) -> int:
    return max(0, min(100, int(pct)))


def _progress(
    pct: int,
    message: str,
    *,
    report_progress: Callable[[int, str], None] | None,
) -> None:
    if report_progress is None:
        return
    try:
        report_progress(_clamp_pct(pct), str(message)[:200])
    except Exception:  # pragma: no cover - best effort
        return


def _build_doc_filters(doc_ids: list[str]) -> Any | None:
    if not doc_ids:
        return None
    try:
        from llama_index.core.vector_stores import (
            FilterCondition,
            MetadataFilter,
            MetadataFilters,
        )
    except Exception:  # pragma: no cover - optional llama-index
        return None

    filters: list[Any] = []
    for doc_id in doc_ids:
        filters.append(MetadataFilter(key="doc_id", value=str(doc_id)))
        filters.append(MetadataFilter(key="document_id", value=str(doc_id)))
    return MetadataFilters(filters=filters, condition=FilterCondition.OR)


def _query_vector_index(
    *,
    vector_index: Any,
    query: str,
    cfg: DocMindSettings,
    filters: Any | None,
    allow_unfiltered_fallback: bool = False,
) -> tuple[str, list[Mapping[str, object]]]:
    """Query a vector index and return (answer, citations)."""
    engine: Any
    try:
        engine = vector_index.as_query_engine(
            similarity_top_k=int(cfg.retrieval.top_k),
            filters=filters,
            response_mode="compact",
        )
    except TypeError as exc:
        if filters is not None and not allow_unfiltered_fallback:
            raise ValueError(
                "Vector index does not support metadata filters. "
                "Enable allow_unfiltered_fallback to proceed without filters."
            ) from exc
        if filters is not None:
            logger.warning(
                "Vector index does not support metadata filters; "
                "falling back to an unfiltered query."
            )
        engine = vector_index.as_query_engine()

    response = engine.query(str(query))
    answer = str(getattr(response, "response", response))

    citations: list[Mapping[str, object]] = []
    source_nodes = getattr(response, "source_nodes", None)
    if isinstance(source_nodes, list):
        for item in source_nodes[:25]:
            node = getattr(item, "node", None) or item
            meta = getattr(node, "metadata", None)
            if isinstance(meta, dict) and meta:
                citations.append({"metadata": meta})
    return answer, citations


def _run_combined_mode(
    *,
    vector_index: Any,
    query: str,
    cfg: DocMindSettings,
    doc_ids: list[str],
    warnings: list[str],
    auto_decision_reason: str | None,
    cancel_event: threading.Event | None,
    report_progress: Callable[[int, str], None] | None,
) -> AnalysisResult:
    _check_cancel(cancel_event)
    filters = _build_doc_filters(doc_ids)
    if doc_ids and filters is None:
        warnings.append(
            "filters_unavailable: running combined analysis without doc filters"
        )
    answer, _citations = _query_vector_index(
        vector_index=vector_index,
        query=query,
        cfg=cfg,
        filters=filters,
        allow_unfiltered_fallback=True,
    )
    _progress(100, "Done", report_progress=report_progress)
    return AnalysisResult(
        mode="combined",
        per_doc=[],
        combined=answer,
        reduce=None,
        warnings=warnings,
        auto_decision_reason=auto_decision_reason,
    )


def _run_one_doc(
    *,
    vector_index: Any,
    query: str,
    cfg: DocMindSettings,
    doc: DocumentRef,
    cancel_event: threading.Event | None,
) -> PerDocResult:
    _check_cancel(cancel_event)
    t0 = time.perf_counter()
    filters = _build_doc_filters([doc.doc_id])
    if filters is None:
        raise RuntimeError("filters unavailable")
    answer, citations = _query_vector_index(
        vector_index=vector_index,
        query=query,
        cfg=cfg,
        filters=filters,
    )
    _check_cancel(cancel_event)
    return PerDocResult(
        doc_id=doc.doc_id,
        doc_name=doc.doc_name,
        answer=answer,
        citations=citations,
        duration_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _run_separate_mode(
    *,
    vector_index: Any,
    query: str,
    cfg: DocMindSettings,
    documents: list[DocumentRef],
    warnings: list[str],
    auto_decision_reason: str | None,
    cancel_event: threading.Event | None,
    report_progress: Callable[[int, str], None] | None,
) -> AnalysisResult:
    if not documents:
        raise ValueError("separate mode requires selected documents")

    per_doc: list[PerDocResult] = []
    max_workers = min(int(cfg.analysis.max_workers), len(documents))
    if max_workers <= 1:
        for idx, doc in enumerate(documents, start=1):
            per_doc.append(
                _run_one_doc(
                    vector_index=vector_index,
                    query=query,
                    cfg=cfg,
                    doc=doc,
                    cancel_event=cancel_event,
                )
            )
            _progress(
                int(idx / max(1, len(documents)) * 100),
                f"Analyzed {doc.doc_name}",
                report_progress=report_progress,
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(
                    _run_one_doc,
                    vector_index=vector_index,
                    query=query,
                    cfg=cfg,
                    doc=doc,
                    cancel_event=cancel_event,
                ): doc
                for doc in documents
            }
            for completed_count, future in enumerate(as_completed(futs), start=1):
                _check_cancel(cancel_event)
                try:
                    per_doc.append(future.result())
                except AnalysisCancelledError:
                    raise
                except Exception as exc:
                    doc = futs[future]
                    _safe_log_jsonl(
                        {
                            "event": "analysis_doc_failed",
                            "mode": "separate",
                            "doc_id": doc.doc_id,
                            "doc_name": doc.doc_name,
                            "error_type": type(exc).__name__,
                        },
                        key_id="analysis.doc_failed",
                    )
                    raise
                _progress(
                    int(completed_count / max(1, len(documents)) * 100),
                    f"Analyzed {futs[future].doc_name}",
                    report_progress=report_progress,
                )

    _progress(100, "Done", report_progress=report_progress)
    return AnalysisResult(
        mode="separate",
        per_doc=sorted(per_doc, key=lambda r: r.doc_name),
        combined=None,
        reduce=None,
        warnings=warnings,
        auto_decision_reason=auto_decision_reason,
    )


def run_analysis(
    *,
    query: str,
    mode: AnalysisMode,
    vector_index: Any,
    documents: list[DocumentRef],
    cfg: DocMindSettings | None = None,
    cancel_event: threading.Event | None = None,
    report_progress: Callable[[int, str], None] | None = None,
) -> AnalysisResult:
    """Run analysis in the requested mode.

    Args:
        query: User analysis question.
        mode: Requested analysis mode.
        vector_index: LlamaIndex vector index (or compatible stub).
        documents: Selected documents (empty means "entire corpus").
        cfg: Settings override.
        cancel_event: Optional cooperative cancellation token.
        report_progress: Optional callback receiving (percent, message).

    Returns:
        AnalysisResult containing combined and/or per-document results.

    Raises:
        AnalysisCancelledError: Raised when cancel_event triggers cooperative
            cancellation during analysis.
        ValueError: Raised when AnalysisMode.SEPARATE is requested but no
            documents are provided.
    """
    cfg = cfg or settings
    start = time.perf_counter()
    warnings: list[str] = []

    doc_ids = [d.doc_id for d in documents if d.doc_id]
    resolved, reason = resolve_mode(mode, doc_count=len(doc_ids), cfg=cfg)
    _safe_log_jsonl(
        {
            "event": "analysis_mode_selected",
            "mode": resolved,
            "doc_count": len(doc_ids),
            "auto_decision_reason": reason,
        },
        key_id="analysis.mode_selected",
    )

    try:
        _check_cancel(cancel_event)
        _progress(0, "Starting analysis", report_progress=report_progress)
        if resolved == "combined":
            result = _run_combined_mode(
                vector_index=vector_index,
                query=query,
                cfg=cfg,
                doc_ids=doc_ids,
                warnings=warnings,
                auto_decision_reason=reason,
                cancel_event=cancel_event,
                report_progress=report_progress,
            )
        else:
            result = _run_separate_mode(
                vector_index=vector_index,
                query=query,
                cfg=cfg,
                documents=documents,
                warnings=warnings,
                auto_decision_reason=reason,
                cancel_event=cancel_event,
                report_progress=report_progress,
            )

        duration_ms = (time.perf_counter() - start) * 1000.0
        _safe_log_jsonl(
            {
                "event": "analysis_completed",
                "mode": result.mode,
                "duration_ms": round(duration_ms, 2),
                "per_doc_count": len(result.per_doc),
                "success": True,
            },
            key_id="analysis.completed",
        )
        return result
    except AnalysisCancelledError:
        duration_ms = (time.perf_counter() - start) * 1000.0
        _safe_log_jsonl(
            {
                "event": "analysis_cancelled",
                "mode": resolved,
                "duration_ms": round(duration_ms, 2),
            },
            key_id="analysis.cancelled",
        )
        raise
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000.0
        _safe_log_jsonl(
            {
                "event": "analysis_failed",
                "mode": resolved,
                "duration_ms": round(duration_ms, 2),
                "error_type": type(exc).__name__,
            },
            key_id="analysis.failed",
        )
        raise


__all__ = [
    "AnalysisCancelledError",
    "auto_select_mode",
    "discover_uploaded_documents",
    "resolve_mode",
    "run_analysis",
]
