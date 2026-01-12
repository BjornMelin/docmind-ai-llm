"""Canonical ingestion models for the DocMind pipeline.

These Pydantic models describe the configuration, input payloads, and
normalized outputs for the upcoming LlamaIndex-first ingestion system. They
replace the legacy Unstructured-specific models that previously existed in
this module.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ProcessingStrategy(str, Enum):
    """Enumerate OCR processing strategies for the ingestion pipeline."""

    FAST = "fast"
    OCR_ONLY = "ocr_only"
    HI_RES = "hi_res"


class IngestionConfig(BaseModel):
    """Declarative configuration for building the ingestion pipeline."""

    chunk_size: int = Field(default=1024, ge=1, description="Maximum tokens per node")
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        description="Tokens of overlap between contiguous nodes",
    )
    enable_image_encryption: bool = Field(
        default=False,
        description="Encrypt rendered page images with AES-GCM when true",
    )
    enable_image_indexing: bool = Field(
        default=True,
        description=(
            "Index rendered PDF page images into the multimodal image collection "
            "(SigLIP)."
        ),
    )
    image_index_batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Batch size used when embedding and indexing page images",
    )
    cache_dir: Path | None = Field(
        default=None, description="Optional directory for LlamaIndex cache"
    )
    cache_collection: str = Field(
        default="docmind_ingestion", description="Namespace for cache storage"
    )
    docstore_path: Path | None = Field(
        default=None, description="Optional filesystem docstore location"
    )
    enable_observability: bool = Field(
        default=False, description="Enable OpenTelemetry tracing/metrics"
    )
    observability_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sampling probability for telemetry events",
    )
    span_exporter_endpoint: str | None = Field(
        default=None,
        description="OTLP endpoint used when observability is enabled",
    )

    @model_validator(mode="after")
    def _validate(self) -> IngestionConfig:
        """Ensure overlap < chunk size and observability fields are consistent."""
        if self.chunk_overlap >= self.chunk_size:
            msg = "chunk_overlap must be strictly less than chunk_size"
            raise ValueError(msg)
        if self.enable_observability and not self.span_exporter_endpoint:
            msg = "span_exporter_endpoint required when observability is enabled"
            raise ValueError(msg)
        return self


class IngestionInput(BaseModel):
    """Normalized ingestion payload.

    Exactly one of ``source_path`` or ``payload_bytes`` must be provided.
    """

    document_id: str = Field(..., min_length=1, description="Stable document ID")
    source_path: Path | None = Field(
        default=None, description="Path to the document on disk"
    )
    payload_bytes: bytes | None = Field(
        default=None, description="In-memory document payload"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata for the document"
    )
    encrypt_images: bool = Field(
        default=False, description="Override to force page-image encryption"
    )

    @model_validator(mode="after")
    def _validate_payload(self) -> IngestionInput:
        """Validate mutual exclusivity of payload fields and normalise paths."""
        has_path = self.source_path is not None
        has_bytes = self.payload_bytes is not None
        if has_path == has_bytes:
            msg = "Provide exactly one of source_path or payload_bytes"
            raise ValueError(msg)
        if self.source_path is not None:
            self.source_path = self.source_path.expanduser()
        return self


class ExportArtifact(BaseModel):
    """Metadata describing an exported artifact (manifest entry, graph, etc.)."""

    name: str = Field(..., min_length=1, description="Human-readable artifact name")
    path: Path = Field(
        ...,
        description=(
            "Ephemeral filesystem path to the artifact (runtime-only; excluded "
            "from serialization to avoid persisting host paths)."
        ),
        exclude=True,
    )
    content_type: str = Field(
        default="application/octet-stream",
        description="MIME type of the artifact on disk",
    )
    size_bytes: int | None = Field(
        default=None, ge=0, description="Size of the artifact, if known"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional exporter metadata"
    )

    @model_validator(mode="after")
    def _resolve_path(self) -> ExportArtifact:
        """Ensure ``path`` is always stored as a resolved ``Path`` instance."""
        self.path = Path(self.path)
        return self


class ManifestSummary(BaseModel):
    """Summary of the snapshot manifest emitted by the ingestion pipeline."""

    corpus_hash: str = Field(..., min_length=8, description="Hash of source corpus")
    config_hash: str = Field(..., min_length=8, description="Hash of pipeline config")
    payload_count: int = Field(..., ge=0, description="Number of payload files")
    checksum: str | None = Field(
        default=None, description="Optional checksum covering manifest lines"
    )
    complete: bool = Field(
        default=False, description="Whether the manifest is marked complete"
    )


class IngestionResult(BaseModel):
    """Structured output from a pipeline execution."""

    nodes: list[Any] = Field(
        default_factory=list, description="LlamaIndex nodes yielded by ingestion"
    )
    documents: list[Any] = Field(
        default_factory=list,
        description="Source documents supplied to the ingestion pipeline",
    )
    manifest: ManifestSummary = Field(
        ..., description="Summary of the manifest written during ingestion"
    )
    exports: list[ExportArtifact] = Field(
        default_factory=list, description="Artifacts emitted alongside the nodes"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extra execution metadata"
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0, description="Total ingestion time in milliseconds"
    )

    def has_exports(self) -> bool:
        """Return ``True`` when one or more export artifacts are present."""
        return bool(self.exports)


__all__ = [
    "ExportArtifact",
    "IngestionConfig",
    "IngestionInput",
    "IngestionResult",
    "ManifestSummary",
    "ProcessingStrategy",
]
