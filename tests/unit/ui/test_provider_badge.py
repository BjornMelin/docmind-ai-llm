"""Unit test for provider_badge to bump coverage."""

import pytest

from src.config.settings import settings
from src.ui.components.provider_badge import provider_badge


@pytest.mark.unit
def test_provider_badge_renders_without_error():
    """Test that provider_badge renders without error."""
    provider_badge(settings)
