"""Tests for the ingestion-focused models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models.processing import (
    ExportArtifact,
    IngestionConfig,
    IngestionInput,
    IngestionResult,
    ManifestSummary,
)


class TestIngestionConfig:
    """Coverage for ``IngestionConfig`` validation rules."""

    def test_defaults(self) -> None:
        """Config exposes sane defaults."""
        cfg = IngestionConfig()
        assert cfg.chunk_size == 1024
        assert cfg.chunk_overlap == 100
        assert not cfg.enable_observability

    def test_overlap_validation(self) -> None:
        """Overlap must be strictly less than chunk size."""
        with pytest.raises(ValidationError):
            IngestionConfig(chunk_size=128, chunk_overlap=128)

    def test_observability_requires_endpoint(self) -> None:
        """Observability flag enforces exporter endpoint."""
        with pytest.raises(ValidationError):
            IngestionConfig(enable_observability=True)

        cfg = IngestionConfig(
            enable_observability=True, span_exporter_endpoint="http://localhost:4317"
        )
        assert cfg.span_exporter_endpoint == "http://localhost:4317"


class TestIngestionInput:
    """Ensure payload validation for ``IngestionInput`` works."""

    def test_requires_exactly_one_payload(self, tmp_path: Path) -> None:
        """Inputs must provide exactly one payload variant."""
        with pytest.raises(ValidationError):
            IngestionInput(document_id="doc-1")

        with pytest.raises(ValidationError):
            IngestionInput(
                document_id="doc-1",
                source_path=tmp_path / "file.txt",
                payload_bytes=b"content",
            )

    def test_source_path_normalised(self, tmp_path: Path) -> None:
        """Source path inputs are normalised."""
        file_path = tmp_path / "doc.txt"
        file_path.write_text("hello")
        inp = IngestionInput(document_id="doc-1", source_path=file_path)
        assert inp.source_path == file_path.expanduser()

    def test_payload_bytes_allowed(self) -> None:
        """In-memory payloads are accepted."""
        inp = IngestionInput(document_id="doc-raw", payload_bytes=b"data")
        assert inp.payload_bytes == b"data"
        assert inp.source_path is None


class TestIngestionResult:
    """Basic behaviour for ``IngestionResult`` scaffolding."""

    def test_has_exports_property(self, tmp_path: Path) -> None:
        """has_exports helper reflects export presence."""
        manifest = ManifestSummary(
            corpus_hash="abc12345",
            config_hash="def67890",
            payload_count=2,
            complete=True,
        )
        artifact = ExportArtifact(name="manifest", path=tmp_path / "manifest.jsonl")
        result = IngestionResult(manifest=manifest, exports=[artifact])
        assert result.has_exports()

    def test_no_exports_returns_false(self) -> None:
        """Empty exports produce a falsey result."""
        manifest = ManifestSummary(
            corpus_hash="abc12345",
            config_hash="def67890",
            payload_count=0,
        )
        result = IngestionResult(manifest=manifest)
        assert not result.has_exports()
