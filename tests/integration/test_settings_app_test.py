"""Integration test for Settings page using Streamlit AppTest.

Ensures the Settings page renders without error and exposes the
server-side hybrid retrieval toggle.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_settings_page_renders_and_has_hybrid_toggle() -> None:
    """Render the Settings page and assert the hybrid toggle is present.

    This is a light smoke test intended to avoid flaky network or
    external dependencies. It only verifies that the Settings page
    successfully renders and includes the expected retrieval toggle.
    """
    at = AppTest.from_file("src/pages/04_settings.py")
    at.run()
    # Find the checkbox labeled with our retrieval toggle text
    labels = [getattr(cb, "label", "") for cb in at.checkbox]
    assert any(
        "server-side hybrid retrieval" in str(label).lower() for label in labels
    ), "Expected server-side hybrid retrieval toggle to be present in Settings page"
