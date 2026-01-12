"""Shared helpers for coordinator integration tests."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def patch_supervisor_and_react(supervisor_stream_shim):
    """Patch supervisor/react agent creation to use the shim."""
    with (
        patch(
            "src.agents.coordinator.create_supervisor",
            return_value=supervisor_stream_shim,
        ),
        patch("src.agents.coordinator.create_react_agent"),
    ):
        yield
