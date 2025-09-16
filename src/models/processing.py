"""Canonical ingestion models for the DocMind pipeline.

These Pydantic models describe the configuration, input payloads, and
normalized outputs for the upcoming LlamaIndex-first ingestion system. They
replace the legacy Unstructured-specific models that previously existed in
this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


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
    def _validate(cls, values: IngestionConfig) -> IngestionConfig:  # noqa: N805  # pylint: disable=no-self-argument
        """Ensure overlap < chunk size and observability fields are consistent."""
        if values.chunk_overlap >= values.chunk_size:
            msg = "chunk_overlap must be strictly less than chunk_size"
            raise ValueError(msg)
        if values.enable_observability and not values.span_exporter_endpoint:
            msg = "span_exporter_endpoint required when observability is enabled"
            raise ValueError(msg)
        return values


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
    def _validate_payload(cls, values: IngestionInput) -> IngestionInput:  # noqa: N805  # pylint: disable=no-self-argument
        """Validate mutual exclusivity of payload fields and normalise paths."""
        has_path = values.source_path is not None
        has_bytes = values.payload_bytes is not None
        if has_path == has_bytes:
            msg = "Provide exactly one of source_path or payload_bytes"
            raise ValueError(msg)
        if values.source_path is not None:
            values.source_path = values.source_path.expanduser()
        return values


class ExportArtifact(BaseModel):
    """Metadata describing an exported artifact (manifest entry, graph, etc.)."""

    name: str = Field(..., min_length=1, description="Human-readable artifact name")
    path: Path = Field(..., description="Filesystem path to the artifact")
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
    def _resolve_path(cls, values: ExportArtifact) -> ExportArtifact:  # noqa: N805  # pylint: disable=no-self-argument
        """Ensure ``path`` is always stored as a resolved ``Path`` instance."""
        values.path = Path(values.path)
        return values


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
]
