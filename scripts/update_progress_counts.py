"""Recalculate and update progress counts in a planning Markdown file.

Usage:
  uv run python scripts/update_progress_counts.py \
    agent-logs/2025-09-02/settings/002-settings-final-research-and-plan.md

The script updates the "## Progress Summary" block in-place.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SECTION_HEADERS: dict[str, str] = {
    "Libraries & Imports": "## Libraries & Imports (Ensure Available)",
    "ADR Updates": "## ADR Updates (Precise)",
    "Code Changes (A–F)": "## Code Changes — Step‑by‑Step (No changes applied yet)",
    "Tests": "## Tests (Add/Update)",
    "Final Checklist": "## Final Checklist (Execution Order)",
}


def _count_checkboxes(md: str, section_header: str) -> tuple[int, int]:
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


def _update_summary_block(md: str, counts: dict[str, tuple[int, int]]) -> str:
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
            libs_done, libs_total = counts["Libraries & Imports"]
            adr_done, adr_total = counts["ADR Updates"]
            code_done, code_total = counts["Code Changes (A–F)"]
            tests_done, tests_total = counts["Tests"]
            checklist_done, checklist_total = counts["Final Checklist"]
            summary = (
                "\n\n## Progress Summary\n\n"
                f"- Overall: {overall_done}/{overall_total}\n"
                f"- Libraries & Imports: {libs_done}/{libs_total}\n"
                f"- ADR Updates: {adr_done}/{adr_total}\n"
                f"- Code Changes (A–F): {code_done}/{code_total}\n"
                f"- Tests: {tests_done}/{tests_total}\n"
                f"- Final Checklist: {checklist_done}/{checklist_total}\n"
            )
            return md[: insert_pos + 1] + summary + md[insert_pos + 1 :]
        # Else, prepend to file
        libs_done, libs_total = counts["Libraries & Imports"]
        adr_done, adr_total = counts["ADR Updates"]
        code_done, code_total = counts["Code Changes (A–F)"]
        tests_done, tests_total = counts["Tests"]
        checklist_done, checklist_total = counts["Final Checklist"]
        preface = (
            "## Progress Summary\n\n"
            f"- Overall: {overall_done}/{overall_total}\n"
            f"- Libraries & Imports: {libs_done}/{libs_total}\n"
            f"- ADR Updates: {adr_done}/{adr_total}\n"
            f"- Code Changes (A–F): {code_done}/{code_total}\n"
            f"- Tests: {tests_done}/{tests_total}\n"
            f"- Final Checklist: {checklist_done}/{checklist_total}\n\n"
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
    """Parse command line arguments and update progress counts in markdown file."""
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(
            "agent-logs/2025-09-02/settings/002-settings-final-research-and-plan.md"
        )
    )
    text = path.read_text(encoding="utf-8")

    counts: dict[str, tuple[int, int]] = {
        label: _count_checkboxes(text, header)
        for label, header in SECTION_HEADERS.items()
    }

    updated = _update_summary_block(text, counts)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
