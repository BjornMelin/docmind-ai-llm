"""
Recalculate and update progress counts in a planning Markdown file.

Usage:
  uv run python scripts/update_progress_counts.py \
    agent-logs/2025-09-02/settings/002-settings-final-research-and-plan.md

The script updates the "## Progress Summary" block in-place.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Tuple


SECTION_HEADERS: Dict[str, str] = {
    "Libraries & Imports": "## Libraries & Imports (Ensure Available)",
    "ADR Updates": "## ADR Updates (Precise)",
    "Code Changes (A–F)": "## Code Changes — Step‑by‑Step (No changes applied yet)",
    "Tests": "## Tests (Add/Update)",
    "Final Checklist": "## Final Checklist (Execution Order)",
}


def _count_checkboxes(md: str, section_header: str) -> Tuple[int, int]:
    start = md.find(section_header)
    if start == -1:
        return 0, 0
    # Find end of this section (next H2 or EOF)
    next_h2 = re.search(r"\n## ", md[start + 1 :])
    end = (start + 1 + next_h2.start()) if next_h2 else len(md)
    section = md[start:end]

    matches = re.findall(r"^\s*-\s*\[([ xX])\]", section, flags=re.MULTILINE)
    total = len(matches)
    done = sum(1 for m in matches if m.lower() == "x")
    return done, total


def _update_summary_block(md: str, counts: Dict[str, Tuple[int, int]]) -> str:
    # Compute overall
    overall_done = sum(d for d, _ in counts.values())
    overall_total = sum(t for _, t in counts.values())

    def replace_line(label: str, done: int, total: int, s: str) -> str:
        pattern = rf"^- {re.escape(label)}: \d+/\d+$"
        repl = f"- {label}: {done}/{total}"
        return re.sub(pattern, repl, s, flags=re.MULTILINE)

    # Ensure Progress Summary block exists
    if "## Progress Summary" not in md:
        insert_after = md.find("Author:")
        if insert_after != -1:
            insert_pos = md.find("\n", insert_after)
            summary = (
                "\n\n## Progress Summary\n\n"
                f"- Overall: {overall_done}/{overall_total}\n"
                f"- Libraries & Imports: {counts['Libraries & Imports'][0]}/{counts['Libraries & Imports'][1]}\n"
                f"- ADR Updates: {counts['ADR Updates'][0]}/{counts['ADR Updates'][1]}\n"
                f"- Code Changes (A–F): {counts['Code Changes (A–F)'][0]}/{counts['Code Changes (A–F)'][1]}\n"
                f"- Tests: {counts['Tests'][0]}/{counts['Tests'][1]}\n"
                f"- Final Checklist: {counts['Final Checklist'][0]}/{counts['Final Checklist'][1]}\n"
            )
            return md[: insert_pos + 1] + summary + md[insert_pos + 1 :]
        # Else, prepend to file
        preface = (
            "## Progress Summary\n\n"
            f"- Overall: {overall_done}/{overall_total}\n"
            f"- Libraries & Imports: {counts['Libraries & Imports'][0]}/{counts['Libraries & Imports'][1]}\n"
            f"- ADR Updates: {counts['ADR Updates'][0]}/{counts['ADR Updates'][1]}\n"
            f"- Code Changes (A–F): {counts['Code Changes (A–F)'][0]}/{counts['Code Changes (A–F)'][1]}\n"
            f"- Tests: {counts['Tests'][0]}/{counts['Tests'][1]}\n"
            f"- Final Checklist: {counts['Final Checklist'][0]}/{counts['Final Checklist'][1]}\n\n"
        )
        return preface + md

    # Update existing block lines
    md = replace_line("Overall", overall_done, overall_total, md)
    for label in (
        "Libraries & Imports",
        "ADR Updates",
        "Code Changes (A–F)",
        "Tests",
        "Final Checklist",
    ):
        d, t = counts[label]
        md = replace_line(label, d, t, md)
    return md


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "agent-logs/2025-09-02/settings/002-settings-final-research-and-plan.md"
    )
    text = path.read_text(encoding="utf-8")

    counts: Dict[str, Tuple[int, int]] = {
        label: _count_checkboxes(text, header)
        for label, header in SECTION_HEADERS.items()
    }

    updated = _update_summary_block(text, counts)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

