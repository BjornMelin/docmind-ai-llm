"""Streamlit entrypoint that keeps the repository root on ``sys.path``.

DocMind uses absolute imports (``import src...``) throughout the codebase.
Streamlit modifies ``sys.path`` based on the script location; running a script
inside ``src/`` can remove the repository root from module search.

Run:
    uv run streamlit run app.py
"""

from __future__ import annotations

from src.app import main

if __name__ == "__main__":  # pragma: no cover
    main()
