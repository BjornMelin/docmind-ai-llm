#!/usr/bin/env python3
"""Validates documentation directory structure against actual src/ contents."""

import json
import re
import sys
from pathlib import Path


def extract_manifest(doc_path: Path) -> dict:
    """Extracts the JSON manifest from system-architecture.md."""
    content = doc_path.read_text()

    match = re.search(r"```json\s+(.*?)\s+```", content, re.DOTALL)
    if not match:
        raise ValueError("Could not find JSON manifest in system-architecture.md")
    return json.loads(match.group(1))


def verify_parity():
    """Main execution function for structural parity verification."""
    repo_root = Path(__file__).parent.parent
    doc_path = repo_root / "docs/developers/system-architecture.md"
    src_path = repo_root / "src"

    try:
        manifest = extract_manifest(doc_path)
    except Exception as e:
        print(f"Error extracting manifest: {e}")
        sys.exit(1)

    canonical = set(manifest.get("canonical_src", []))

    # Get actual top-level directories in src/ (ignoring __pycache__ and hidden)
    actual = {
        p.name
        for p in src_path.iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    }

    # Filter out egg-info or other non-package dirs if they exist
    actual = {d for d in actual if not d.endswith(".egg-info")}

    missing_in_docs = actual - canonical
    missing_in_src = canonical - actual

    errors = []
    if missing_in_docs:
        errors.append(
            f"Directories in src/ but missing from manifest: {missing_in_docs}"
        )
    if missing_in_src:
        errors.append(
            f"Directories in manifest but missing from src/: {missing_in_src}"
        )

    if errors:
        for err in errors:
            print(f"FAILED: {err}")
        sys.exit(1)
    else:
        print("SUCCESS: Documentation structural parity verified (v2.0.0).")
        sys.exit(0)


if __name__ == "__main__":
    verify_parity()
