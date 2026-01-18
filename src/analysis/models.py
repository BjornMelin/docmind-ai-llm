"""Typed analysis result models (SPEC-036)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

AnalysisMode = Literal["auto", "separate", "combined"]
ResolvedAnalysisMode = Literal["separate", "combined"]


@dataclass(frozen=True, slots=True)
class DocumentRef:
    """A document selected for analysis.

    Args:
        doc_id: Document identifier.
        doc_name: Human-readable name.
    """

    doc_id: str
    doc_name: str


@dataclass(frozen=True, slots=True)
class PerDocResult:
    """Result for a single document analysis run.

    Args:
        doc_id: Document identifier.
        doc_name: Human-readable name.
        answer: Generated answer text.
        citations: Source citations for the answer.
        duration_ms: Execution time in milliseconds.
    """

    doc_id: str
    doc_name: str
    answer: str
    citations: list[Mapping[str, object]]
    duration_ms: float


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Outcome of an analysis execution.

    Args:
        mode: Resolved analysis mode.
        per_doc: Per-document results.
        combined: Combined answer when in combined mode.
        reduce: Reduction summary when applicable.
        warnings: Warning messages.
        auto_decision_reason: Reason used when auto-resolving mode.
    """

    mode: ResolvedAnalysisMode
    per_doc: list[PerDocResult]
    combined: str | None
    reduce: str | None
    warnings: list[str]
    auto_decision_reason: str | None


__all__ = [
    "AnalysisMode",
    "AnalysisResult",
    "DocumentRef",
    "PerDocResult",
    "ResolvedAnalysisMode",
]
