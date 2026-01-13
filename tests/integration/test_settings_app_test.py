"""Integration test for Settings page using Streamlit AppTest.

Ensures the Settings page renders without error and exposes the
server-side hybrid retrieval toggle.
"""

from __future__ import annotations

import os
import time

from streamlit.testing.v1 import AppTest


def test_settings_page_renders_and_has_hybrid_toggle() -> None:
    """Render the Settings page and assert the hybrid toggle is present.

    This is a light smoke test intended to avoid flaky network or
    external dependencies. It only verifies that the Settings page
    successfully renders and includes the expected retrieval toggle.
    """
    at = AppTest.from_file("src/pages/04_settings.py")
    start = time.monotonic()
    at.run(timeout=10)
    elapsed = time.monotonic() - start
    is_ci = bool(os.getenv("CI") or os.getenv("GITHUB_ACTIONS"))
    budget = 9.0 if is_ci else 8.0
    assert elapsed <= budget, (
        f"Settings AppTest render took {elapsed:.2f}s > {budget:.2f}s budget"
    )
    # Find the read-only field exposing the retrieval toggle state
    labels = [getattr(inp, "label", "") for inp in at.text_input]
    assert any(
        str(label).lower() == "server-side hybrid enabled" for label in labels
    ), "Expected server-side hybrid retrieval field to be present in Settings page"

    field = next(
        (
            inp
            for inp in at.text_input
            if str(getattr(inp, "label", "")).lower() == "server-side hybrid enabled"
        ),
        None,
    )
    assert field is not None
    value = str(getattr(field, "value", "")).strip().lower()
    assert value in {"true", "false"}, f"Unexpected field value: {value!r}"

    checkbox_labels = [str(getattr(cb, "label", "")).lower() for cb in at.checkbox]
    assert not any("server-side hybrid" in lbl for lbl in checkbox_labels), (
        "Expected server-side hybrid to be displayed as read-only text, not a checkbox"
    )
