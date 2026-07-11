"""Canonical parser result contracts for local document ingestion."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ParserProfile(StrEnum):
    """Canonical document parser provenance label."""

    CPU_SAFE = "cpu_safe"


class ParserFramework(StrEnum):
    """Supported high-level document conversion frameworks."""

    DOCLING = "docling"
    DIRECT_TEXT = "direct_text"


class PdfBackendName(StrEnum):
    """Supported PDF inspection/rasterization backends."""

    PYPDFIUM2 = "pypdfium2"


class OcrEngineName(StrEnum):
    """Supported OCR engines."""

    RAPIDOCR = "rapidocr"


class ParsingArtifact(BaseModel):
    """PII-safe parser artifact reference."""

    kind: str = Field(..., min_length=1)
    artifact_id: str | None = None
    suffix: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageParseResult(BaseModel):
    """Parser output for one page or page-like section."""

    page_id: str
    page_index: int = Field(ge=0)
    text_markdown: str = ""
    ocr_applied: bool = False
    routing_reason: str = "unknown"


class ParserVersions(BaseModel):
    """Runtime package versions captured during parsing."""

    packages: dict[str, str] = Field(default_factory=dict)


class DocumentParseResult(BaseModel):
    """Canonical parser output consumed by the ingestion facade."""

    document_id: str
    source_filename: str
    source_hash: str
    profile: ParserProfile
    parser_framework: ParserFramework
    pdf_backend: PdfBackendName = PdfBackendName.PYPDFIUM2
    ocr_engine: OcrEngineName = OcrEngineName.RAPIDOCR
    versions: ParserVersions = Field(default_factory=ParserVersions)
    page_count: int = Field(default=0, ge=0)
    pages: list[PageParseResult] = Field(default_factory=list)
    artifacts: list[ParsingArtifact] = Field(default_factory=list)
    routing_decisions: list[dict[str, Any]] = Field(default_factory=list)
    config_hash: str = ""
    health: dict[str, Any] = Field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        """Return deterministic, manifest-safe parser provenance."""
        ocr_pages = [page.page_index for page in self.pages if bool(page.ocr_applied)]
        provenance = {
            "framework": self.parser_framework.value,
            "profile": self.profile.value,
            "package_versions": dict(self.versions.packages),
            "routing_decisions": list(self.routing_decisions),
            "page_routing": [
                {
                    "page_index": page.page_index,
                    "reason": page.routing_reason,
                }
                for page in self.pages
            ],
            "ocr_applied_pages": ocr_pages,
            "searchable_pdf_artifacts": [
                artifact.model_dump(mode="json")
                for artifact in self.artifacts
                if artifact.kind == "searchable_pdf"
            ],
            "config_hash": self.config_hash,
            "health": dict(self.health),
        }
        if self.source_filename.casefold().endswith(".pdf"):
            provenance["pdf_backend"] = self.pdf_backend.value
            provenance["ocr_engine"] = self.ocr_engine.value
        return provenance


__all__ = [
    "DocumentParseResult",
    "OcrEngineName",
    "PageParseResult",
    "ParserFramework",
    "ParserProfile",
    "ParserVersions",
    "ParsingArtifact",
    "PdfBackendName",
]
