"""Document analysis domain modules (SPEC-036)."""

from __future__ import annotations

from src.analysis.models import AnalysisMode, AnalysisResult, DocumentRef, PerDocResult
from src.analysis.service import (
    AnalysisCancelledError,
    discover_uploaded_documents,
    run_analysis,
)

__all__ = [
    "AnalysisCancelledError",
    "AnalysisMode",
    "AnalysisResult",
    "DocumentRef",
    "PerDocResult",
    "discover_uploaded_documents",
    "run_analysis",
]
