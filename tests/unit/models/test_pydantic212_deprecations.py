"""Regression tests guarding against Pydantic 2.12 deprecation warnings."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from pydantic.warnings import PydanticDeprecatedSince212

from src.config.settings import DocMindSettings
from src.models.processing import ExportArtifact, IngestionConfig, IngestionInput

pytestmark = pytest.mark.unit


def test_no_pydantic212_deprecations_on_model_construction(tmp_path: Path) -> None:
    """Construct representative models and ensure no 2.12 deprecations fire."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PydanticDeprecatedSince212)

        _ = IngestionConfig(chunk_size=1024, chunk_overlap=100)
        _ = IngestionInput(document_id="doc-1", source_path=tmp_path / "a.txt")
        _ = ExportArtifact(name="artifact", path=tmp_path / "x.bin")

        _ = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
