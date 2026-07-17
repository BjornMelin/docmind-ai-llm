"""Fail when Streamlit page imports eagerly load model or retrieval stacks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROHIBITED_ROOTS = frozenset({"torch", "transformers", "llama_index", "qdrant_client"})
TARGETS = ("src.app", "src.pages.01_chat", "src.pages.02_documents")
ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import importlib
import json
import sys

module = importlib.import_module(sys.argv[1])
if sys.argv[1] == "src.pages.01_chat":
    module.provider_badge(module.settings)
    module._compute_snapshot_status()
elif sys.argv[1] == "src.pages.02_documents":
    module._render_latest_snapshot_summary()
roots = {name.partition('.')[0] for name in sys.modules}
print(json.dumps(sorted(roots & set(json.loads(sys.argv[2])))))
"""


def main() -> int:
    """Check each target in a fresh interpreter and report every violation."""
    failed = False
    prohibited = json.dumps(sorted(PROHIBITED_ROOTS))
    for target in TARGETS:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE, target, prohibited],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failed = True
            print(f"{target}: import failed", file=sys.stderr)
            print(result.stderr.strip(), file=sys.stderr)
            continue
        violations = json.loads(result.stdout.strip().splitlines()[-1])
        print(f"{target}: {violations}")
        failed = failed or bool(violations)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
