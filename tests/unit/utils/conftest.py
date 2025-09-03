"""Utility test fixtures for monitoring tests (boundary patterns).

Provides lightweight fixtures used by monitoring tests to avoid excessive
patching within individual test cases and keep unit tests deterministic.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(autouse=True)
def rng_seed():
    """Seed RNGs for deterministic unit tests."""
    import random  # pylint: disable=C0415  # late import to speed test collection

    import numpy as np  # pylint: disable=C0415

    random.seed(1337)
    np.random.seed(1337)
    try:
        import torch  # pylint: disable=C0415

        if hasattr(torch, "manual_seed"):
            torch.manual_seed(1337)
    except ImportError:
        # Torch not installed in this environment; ignore torch seeding
        pass


@pytest.fixture
def perf_counter_boundary():
    """Deterministic perf_counter sequence for timing-sensitive tests."""
    seq = [0.0, 0.01, 0.02, 0.03, 0.05]
    with patch("time.perf_counter", side_effect=seq):
        yield {"sequence": seq}


@pytest.fixture
def network_guard():
    """Guard against accidental network calls in unit tests."""

    def _raise(*_a, **_k):
        raise RuntimeError("Network access disabled in unit tests")

    try:
        import httpx  # pylint: disable=C0415
        import requests  # pylint: disable=C0415

        with (
            patch.object(requests, "get", _raise),
            patch.object(requests, "post", _raise),
            patch.object(httpx, "get", _raise),
            patch.object(httpx, "post", _raise),
        ):
            yield
    except ImportError:
        # If libs are not present, yield without patching
        yield


@pytest.fixture(autouse=True)
def no_sleep():
    """Patch time.sleep and asyncio.sleep to avoid real delays in unit tests."""
    import asyncio as _asyncio  # pylint: disable=C0415
    import time as _time  # pylint: disable=C0415

    def _sleep_noop(_secs: float = 0.0):
        return None

    async def _async_sleep_noop(_secs: float = 0.0):
        return None

    with (
        patch.object(_time, "sleep", _sleep_noop),
        patch.object(_asyncio, "sleep", _async_sleep_noop),
    ):
        yield


@pytest.fixture
def logging_boundary():
    """Structured logging boundary for src.utils.monitoring logger.

    Yields a dict with the mocked logger and helper assertion functions.
    """
    mock_logger = Mock()

    def assert_info_called(message_contains: str | None = None) -> None:
        mock_logger.info.assert_called()
        if message_contains is not None:
            call_msg = mock_logger.info.call_args[0][0]
            assert message_contains in call_msg

    def assert_error_called(message_contains: str | None = None) -> None:
        mock_logger.error.assert_called()
        if message_contains is not None:
            call_msg = mock_logger.error.call_args[0][0]
            assert message_contains in call_msg

    def assert_warning_called(message_contains: str | None = None) -> None:
        mock_logger.warning.assert_called()
        if message_contains is not None:
            call_msg = mock_logger.warning.call_args[0][0]
            assert message_contains in call_msg

    with patch("src.utils.monitoring.logger", mock_logger):
        yield {
            "logger": mock_logger,
            "assert_info_called": assert_info_called,
            "assert_error_called": assert_error_called,
            "assert_warning_called": assert_warning_called,
            "call_count": lambda level: getattr(mock_logger, level).call_count,
        }


@pytest.fixture
def performance_boundary():
    """Deterministic timing boundary using a fixed perf_counter sequence."""
    seq = [0.0, 0.015, 0.030, 0.045, 0.060, 0.075]
    with patch("time.perf_counter", side_effect=seq):
        yield {
            "timing_sequence": seq,
            "expected_step_ms": 15.0,
        }


@pytest.fixture
def system_resource_boundary():
    """Boundary for psutil and OS resource calls with predictable values."""
    import os as _os  # pylint: disable=C0415

    class _Mem(SimpleNamespace):
        pass

    fake_proc = Mock()
    fake_proc.memory_info.return_value = SimpleNamespace(
        rss=100 * 1024 * 1024,  # 100 MB
        vms=150 * 1024 * 1024,  # 150 MB
    )
    fake_proc.memory_percent.return_value = 42.0

    with (
        patch("psutil.Process", return_value=fake_proc),
        patch("psutil.virtual_memory", return_value=SimpleNamespace(percent=65.0)),
        patch("psutil.cpu_percent", return_value=35.5),
        patch("psutil.disk_usage", return_value=SimpleNamespace(percent=45.0)),
        patch.object(_os, "getloadavg", return_value=(1.2, 1.5, 1.1), create=True),
    ):
        yield {
            "rss_mb": 100.0,
            "vms_mb": 150.0,
            "cpu_percent": 35.5,
            "disk_percent": 45.0,
            "loadavg": (1.2, 1.5, 1.1),
        }


@pytest.fixture
def ai_stack_boundary():
    """Boundary for AI components using LlamaIndex built-in mocks."""
    from llama_index.core import Settings  # pylint: disable=C0415
    from llama_index.core.embeddings import MockEmbedding  # pylint: disable=C0415
    from llama_index.core.llms import MockLLM  # pylint: disable=C0415

    original_llm = Settings.llm
    original_embed = Settings.embed_model

    mock_llm = MockLLM(max_tokens=256)
    mock_embed = MockEmbedding(embed_dim=1024)

    Settings.llm = mock_llm
    Settings.embed_model = mock_embed

    try:
        yield {
            "llm": mock_llm,
            "embed_model": mock_embed,
            "embed_dim": 1024,
            "max_tokens": 256,
        }
    finally:
        Settings.llm = original_llm
        Settings.embed_model = original_embed
