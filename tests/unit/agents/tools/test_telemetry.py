"""Unit tests for telemetry.log_event (non-failing)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.tools.telemetry import log_event, logger

pytestmark = pytest.mark.unit


def test_log_event_no_exception() -> None:
    """log_event executes without raising."""
    # Using real logger; should not raise
    log_event("test_event", foo="bar")


def test_log_event_swallows_logger_exception() -> None:
    """Internal logger failures are swallowed and do not propagate."""
    with patch.object(logger, "bind", side_effect=Exception("log bind error")):
        # Should not raise even if logger fails internally
        log_event("test_event", foo="bar")
