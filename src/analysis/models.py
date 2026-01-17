"""Typed analysis result models (SPEC-036)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AnalysisMode = Literal["auto", "separate", "combined"]
ResolvedAnalysisMode = Literal["separate", "combined"]


@dataclass(frozen=True, slots=True)
class DocumentRef:
    """A document selected for analysis."""

    doc_id: str
    doc_name: str


@dataclass(frozen=True, slots=True)
class PerDocResult:
    """Result for a single document analysis run."""

    doc_id: str
    doc_name: str
    answer: str
    citations: list[dict[str, Any]]
    duration_ms: float


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Outcome of an analysis execution."""

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
