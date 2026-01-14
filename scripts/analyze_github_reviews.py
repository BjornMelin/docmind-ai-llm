"""Summarize unresolved GitHub PR review threads into a console/CSV report."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

# Files already processed (exclude from output)
PROCESSED_FILES: frozenset[str] = frozenset(
    {"router_factory", "01_chat", "coordinator"}
)

# Sentinel value for missing line numbers (sorts to end of list)
LINE_NONE_SENTINEL: int = sys.maxsize


def should_exclude(file_path: str) -> bool:
    """Check if file is already processed."""
    basename = Path(file_path).stem
    return basename in PROCESSED_FILES


def extract_threads(json_file: Path) -> list[dict[str, Any]] | None:
    """Parse JSON and extract unresolved threads.

    Returns:
        List of unresolved review threads, or None if file not found or JSON is invalid.
    """
    try:
        with json_file.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file}", file=sys.stderr)
        return None
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON in {json_file}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data, dict):
        print(f"Error: Expected JSON object in {json_file}", file=sys.stderr)
        return None

    return data.get("unresolved_review_threads", [])


def get_code_block_indicator(thread: dict[str, Any]) -> bool:
    """Check if thread has suggested code."""
    return any(
        "```" in (comment.get("body") or "") for comment in thread.get("comments", [])
    )


def get_issue_summary(thread: dict[str, Any]) -> str:
    """Extract first comment as summary, limit to 80 chars."""
    comments = thread.get("comments", [])
    if not comments:
        return "No comment"
    body = (comments[0].get("body") or "").split("\n")[0]
    # Remove markdown code blocks and extra spaces
    body = body.replace("```", "").strip()
    return body[:80] if body else "No comment"


def get_line_number(location: dict[str, Any] | None) -> int | None:
    """Extract line number from location, or None if missing."""
    if isinstance(location, dict):
        return location.get("start_line")
    return None


def _get_issue_line_key(issue: dict[str, Any]) -> int:
    """Get sortable line number for an issue (LINE_NONE_SENTINEL when line is None)."""
    return issue["line"] if issue["line"] is not None else LINE_NONE_SENTINEL


def _resolve_json_path(arg_path: str | None) -> Path | None:
    """Resolve JSON input path from CLI arg or DOCMIND_REVIEW_JSON env var.

    Args:
        arg_path: Explicit JSON file path from CLI argument.

    Returns:
        Resolved Path object, or None if neither arg nor env var is provided.
    """
    if arg_path:
        return Path(arg_path)
    env_path = os.environ.get("DOCMIND_REVIEW_JSON")
    return Path(env_path) if env_path else None


def _print_console_report(by_file: dict[str, list[dict[str, Any]]]) -> None:
    """Print formatted console report of unresolved threads grouped by file."""
    print("=" * 120)
    excluded = ", ".join(sorted(PROCESSED_FILES))
    print(f"UNRESOLVED REVIEW THREADS (excluding {excluded})")
    print("=" * 120)
    print()

    for file_path in sorted(by_file.keys()):
        issues = by_file[file_path]
        print(f"\n{file_path}")
        print(f"  Count: {len(issues)} threads")
        print("-" * 120)

        for issue in sorted(issues, key=_get_issue_line_key):
            line = issue["line"]
            line_str = str(line) if line is not None else "N/A"
            summary = issue["summary"]
            has_code = issue["has_code"]
            comments = issue["comment_count"]
            print(
                "  Line "
                f"{line_str:6} | Code: {has_code:3} | Comments: {comments} | {summary}"
            )


def _generate_csv(by_file: dict[str, list[dict[str, Any]]]) -> str:
    """Generate CSV content from grouped issues."""
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["File", "Line", "Has Code Block", "Comments", "Issue Summary"])

    for file_path in sorted(by_file.keys()):
        for issue in sorted(by_file[file_path], key=_get_issue_line_key):
            line_str = str(issue["line"]) if issue["line"] is not None else "N/A"
            writer.writerow(
                [
                    file_path,
                    line_str,
                    issue["has_code"],
                    issue["comment_count"],
                    issue["summary"],
                ]
            )

    return csv_buffer.getvalue()


def _print_statistics(by_file: dict[str, list[dict[str, Any]]]) -> None:
    """Compute and print summary statistics."""
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    total_threads = sum(len(issues) for issues in by_file.values())
    total_with_code = sum(
        1
        for issues in by_file.values()
        for issue in issues
        if issue["has_code"] == "Yes"
    )
    print(f"Total unresolved threads (excl. processed): {total_threads}")
    if total_threads == 0:
        print("Threads with code suggestions: 0 (0%)")
        print("Threads without code: 0 (0%)")
    else:
        print(
            "Threads with code suggestions: "
            f"{total_with_code} ({100 * total_with_code / total_threads:.0f}%)"
        )
        print(
            "Threads without code: "
            f"{total_threads - total_with_code} "
            f"({100 * (total_threads - total_with_code) / total_threads:.0f}%)"
        )
    print(f"Files with issues: {len(by_file)}")

    print("\nTop files by thread count:")
    sorted_files = sorted(by_file.items(), key=lambda x: len(x[1]), reverse=True)
    for file_path, issues in sorted_files[:10]:
        code_count = sum(1 for i in issues if i["has_code"] == "Yes")
        print(f"  {len(issues):2} threads | {code_count:2} with code | {file_path}")


def main() -> None:
    """Run the review thread summarizer CLI."""
    parser = argparse.ArgumentParser(
        description="Summarize unresolved GitHub PR review threads."
    )
    parser.add_argument(
        "--json-file",
        help="Path to JSON output from fetch_unresolved_pr_review_comments.py "
        "(or set DOCMIND_REVIEW_JSON).",
    )
    parser.add_argument(
        "--output-csv",
        default=str(Path(tempfile.gettempdir()) / "unresolved_reviews.csv"),
        help="CSV output path (default: <tempdir>/unresolved_reviews.csv).",
    )
    args = parser.parse_args()

    json_file = _resolve_json_path(args.json_file)
    if json_file is None:
        print("Error: Provide --json-file or set DOCMIND_REVIEW_JSON.", file=sys.stderr)
        sys.exit(1)

    threads = extract_threads(json_file)
    if threads is None:
        sys.exit(1)
    print(f"Total unresolved threads in data: {len(threads)}\n")

    # Group by file and filter
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_processed = 0

    for thread in threads:
        file_path = thread.get("file", "unknown")

        if should_exclude(file_path):
            skipped_processed += 1
            continue

        line_number = get_line_number(thread.get("location"))
        summary = get_issue_summary(thread)
        has_code = get_code_block_indicator(thread)

        by_file[file_path].append(
            {
                "line": line_number,
                "summary": summary,
                "has_code": "Yes" if has_code else "No",
                "thread_id": thread.get("thread_id"),
                "comment_count": thread.get("comment_count", 0),
            }
        )

    print(f"Skipped (already processed): {skipped_processed}")
    print(f"Unresolved threads to handle: {sum(len(v) for v in by_file.values())}\n")

    _print_console_report(by_file)

    print("\n\n" + "=" * 120)
    print("CSV OUTPUT (for spreadsheet import)")
    print("=" * 120 + "\n")

    csv_content = _generate_csv(by_file)
    print(csv_content)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text(csv_content, encoding="utf-8")

    _print_statistics(by_file)

    print(f"\nCSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
