#!/usr/bin/env python3
"""Check for broken links in markdown files.

This script scans markdown files under the provided paths and reports
broken internal links. It respects `.gitignore` when run inside a git repo.
"""

import argparse
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from subprocess import DEVNULL, CalledProcessError, run

DEFAULT_SKIP_DIRS = {".git"}


def get_git_root(start_path: str) -> str | None:
    """Return the repository root for a path.

    Args:
        start_path: Filesystem path to probe.

    Returns:
        Absolute path to the git repository root, or None when unavailable.
    """
    try:
        result = run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def is_git_ignored(path: str, git_root: str | None) -> bool:
    """Check whether a path is ignored by git.

    Args:
        path: Filesystem path to test.
        git_root: Git repository root, or None when not in a repo.

    Returns:
        True if the path is ignored by git, otherwise False.
    """
    if not git_root:
        return False
    rel_path = os.path.relpath(path, git_root)
    if rel_path.startswith(".."):
        return False
    result = run(
        ["git", "check-ignore", "-q", rel_path],
        cwd=git_root,
        check=False,
        stdout=DEVNULL,
        stderr=DEVNULL,
    )
    return result.returncode == 0


def is_hidden_or_skipped(path: Path, root: Path) -> bool:
    """Return True if a path is under a hidden or skipped directory."""
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    parts_to_check = rel_parts
    if path.is_file():
        parts_to_check = rel_parts[:-1]
    for part in parts_to_check:
        if part in {".", ""}:
            continue
        if part.startswith(".") or part in DEFAULT_SKIP_DIRS:
            return True
    return False


def describe_path(path: Path) -> str:
    """Return a display-friendly path relative to the current directory."""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def iter_markdown_files(search_path: Path) -> Iterable[Path]:
    """Yield markdown files for a search path."""
    if search_path.is_file():
        if search_path.suffix == ".md":
            yield search_path
        return
    yield from search_path.rglob("*.md")


def read_markdown(file_path: Path) -> str | None:
    """Read a markdown file and return its content, or None on failure."""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"Warning: Skipping non-UTF-8 markdown file: {file_path}")
        return None
    except OSError as exc:
        print(f"Error reading {file_path}: {exc}")
        return None


def iter_internal_links(content: str) -> Iterable[tuple[str, str]]:
    """Yield (raw_link, link_path) pairs for internal markdown links."""
    stripped = re.sub(r"```.*?```", "", content, flags=re.S)
    found_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", stripped)
    for _, link in found_links:
        if link.startswith(("http://", "https://", "mailto:", "tel:")):
            continue
        if link.startswith("#"):
            continue
        link_path = link.split("#")[0]
        if not link_path:
            continue
        yield link, link_path


def is_valid_target(target_path: Path) -> bool:
    """Return True if a target path resolves to a valid markdown resource."""
    if target_path.is_file():
        return True
    if target_path.suffix != ".md" and target_path.with_suffix(".md").exists():
        return True
    return target_path.is_dir() and (
        (target_path / "index.md").exists()
        or (target_path / "_index.md").exists()
        or (target_path / "README.md").exists()
    )


def check_links(search_paths: Iterable[str]) -> list[dict[str, str]]:
    """Scan markdown files in paths for broken internal links.

    Args:
        search_paths: Paths to search for markdown files.

    Returns:
        A list of broken link records with source, link, and resolved path.
    """
    broken_links: list[dict[str, str]] = []

    for search_path in search_paths:
        abs_search_path = Path(search_path).resolve()
        if not abs_search_path.exists():
            print(f"Warning: Path {abs_search_path} does not exist. Skipping.")
            continue

        git_root_probe = abs_search_path
        if abs_search_path.is_file():
            git_root_probe = abs_search_path.parent
        git_root = get_git_root(str(git_root_probe))

        for file_path in iter_markdown_files(abs_search_path):
            if not file_path.is_file():
                continue
            if is_hidden_or_skipped(file_path, abs_search_path):
                continue
            if is_git_ignored(str(file_path), git_root):
                continue
            content = read_markdown(file_path)
            if content is None:
                continue

            for link, link_path in iter_internal_links(content):
                link_as_path = Path(link_path)
                if link_as_path.is_absolute():
                    # Treat absolute links as relative to git root or search path
                    base = Path(git_root) if git_root else abs_search_path
                    target_path = (base / link_path.lstrip("/")).resolve()
                else:
                    target_path = (file_path.parent / link_as_path).resolve()

                if is_valid_target(target_path):
                    continue
                broken_links.append(
                    {
                        "source": describe_path(file_path),
                        "link": link,
                        "resolved": target_path.as_posix(),
                    }
                )
    return broken_links


def main() -> None:
    """Main entry point for the link checker."""
    parser = argparse.ArgumentParser(
        description="Check for broken links in markdown files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Paths to check for markdown files "
            "(default: . or DOCMIND_CHECK_PATHS env var)"
        ),
    )
    args = parser.parse_args()

    # Get paths from CLI or env var fallback
    paths = args.paths
    if not paths:
        env_paths = os.environ.get("DOCMIND_CHECK_PATHS")
        paths = env_paths.split() if env_paths else ["."]

    print(f"Checking links in: {', '.join(paths)}")

    broken_links = check_links(paths)

    for bl in broken_links:
        print(
            f"Broken link in {bl['source']}: {bl['link']} "
            f"(Resolved to: {bl['resolved']})"
        )

    if broken_links:
        print(f"\nFound {len(broken_links)} broken links.")
        sys.exit(1)
    else:
        print("\nAll internal links are valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
