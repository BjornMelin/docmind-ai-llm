"""Tests for Pydantic models and application settings.

This module tests the data models including AnalysisOutput, Settings configuration,
validation behavior, and environment variable overrides following 2025 best practices.
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from models import AnalysisOutput, Settings


def test_analysis_output_creation():
    """Test AnalysisOutput model creation and field access.

    Tests that AnalysisOutput can be created with valid data and fields
    are properly accessible.
    """
    output = AnalysisOutput(
        summary="Sum",
        key_insights=["insight"],
        action_items=["action"],
        open_questions=["question"],
    )
    assert output.summary == "Sum"


def test_analysis_output_validation():
    """Test AnalysisOutput model validation with invalid data.

    Tests that ValidationError is raised when invalid data types
    are provided to model fields.
    """
    with pytest.raises(ValidationError):
        AnalysisOutput(summary=123)  # Invalid type


def test_settings_default_values():
    """Test Settings model loads with expected default values.

    Tests that Settings model initializes with correct default
    configuration values for the application.
    """
    settings = Settings()
    assert settings.backend == "ollama"
    assert settings.context_size == 4096


@patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://test:1234"})
def test_settings_environment_override():
    """Test Settings model respects environment variable overrides.

    Tests that Settings model properly uses environment variables
    to override default configuration values.
    """
    settings = Settings()
    assert settings.ollama_base_url == "http://test:1234"
