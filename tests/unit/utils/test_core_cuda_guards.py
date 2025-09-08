"""Tests for CUDA guard utilities in utils.core.

Google-Style Docstrings:
    Validates error handling contexts with reraise=False and default returns.
"""

import pytest

pytestmark = pytest.mark.unit


def test_cuda_error_context_reraise_false_default_return():
    """Return default value and no exception when reraise is False.

    Uses a context manager where the block raises a RuntimeError. Expects the
    context to capture the error and set the result to the provided default.
    """
    from src.utils.core import cuda_error_context

    ctx: dict
    with cuda_error_context("op", reraise=False, default_return=0.0) as ctx:
        raise RuntimeError("CUDA device error")
    assert ctx.get("result") == 0.0
    assert "error" in ctx


def test_safe_cuda_operation_returns_default_without_logging():
    """Return default value on error with logging disabled.

    Wraps a lambda that raises a RuntimeError and asserts the default return
    value is provided when `log_errors=False`.
    """
    from src.utils.core import safe_cuda_operation

    out = safe_cuda_operation(
        lambda: (_ for _ in ()).throw(RuntimeError("x")), "op", 1, False
    )
    assert out == 1
