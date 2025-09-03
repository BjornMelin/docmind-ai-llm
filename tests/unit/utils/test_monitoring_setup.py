"""Additional monitoring tests to exercise setup and error branches."""

from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_setup_logging_adds_console_and_file_handlers(tmp_path):
    """Test that logging setup adds both console and file handlers."""
    from src.utils import monitoring

    log_file = tmp_path / "test.log"

    with (
        patch.object(monitoring.logger, "remove") as rm,
        patch.object(monitoring.logger, "add") as add,
    ):
        monitoring.setup_logging(log_level="DEBUG", log_file=str(log_file))

    # First add is console, second add is file
    assert rm.called
    assert add.call_count >= 2


@pytest.mark.unit
def test_get_system_info_error_returns_empty():
    """Test that system info collection handles errors by returning empty dict."""
    from src.utils import monitoring

    with patch("psutil.cpu_percent", side_effect=OSError("boom")):
        info = monitoring.get_system_info()
    assert info == {}
