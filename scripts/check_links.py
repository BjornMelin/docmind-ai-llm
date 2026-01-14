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
from subprocess import DEVNULL, run

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
    except Exception:
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


def check_links(search_paths: Iterable[str]) -> list[dict[str, str]]:
    """Scan markdown files in paths for broken internal links.

    Args:
        search_paths: Paths to search for markdown files.

    Returns:
        A list of broken link records with source, link, and resolved path.
    """
    broken_links: list[dict[str, str]] = []

    for search_path in search_paths:
        abs_search_path = os.path.abspath(search_path)
        if not os.path.exists(abs_search_path):
            print(f"Warning: Path {abs_search_path} does not exist. Skipping.")
            continue

        git_root = get_git_root(abs_search_path)

        for root, dirs, files in os.walk(abs_search_path):
            # Skip hidden and vendor directories.
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in DEFAULT_SKIP_DIRS
                and not is_git_ignored(os.path.join(root, d), git_root)
            ]

            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    if is_git_ignored(file_path, git_root):
                        continue
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        print(f"Warning: Skipping non-UTF-8 markdown file: {file_path}")
                        continue
                    except OSError as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

                    # Strip fenced code blocks to avoid false positives.
                    content = re.sub(r"```.*?```", "", content, flags=re.S)

                    # Simple markdown link regex
                    # Matches standard [label](link) pairs, including quoted labels.
                    found_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)

                    for _, link in found_links:
                        # Skip web links
                        if link.startswith(("http://", "https://", "mailto:", "tel:")):
                            continue

                        # Skip anchor-only links
                        if link.startswith("#"):
                            continue
                        # Remove anchors
                        link_path = link.split("#")[0]
                        if not link_path:
                            continue

                        # Resolve relative path
                        target_path = os.path.abspath(os.path.join(root, link_path))

                        # Check if file exists with fallbacks
                        is_valid = False
                        if (
                            os.path.isfile(target_path)
                            or (
                                not target_path.endswith(".md")
                                and os.path.exists(target_path + ".md")
                            )
                            or (
                                os.path.isdir(target_path)
                                and (
                                    os.path.exists(
                                        os.path.join(target_path, "index.md")
                                    )
                                    or os.path.exists(
                                        os.path.join(target_path, "_index.md")
                                    )
                                    or os.path.exists(
                                        os.path.join(target_path, "README.md")
                                    )
                                )
                            )
                            or os.path.isdir(target_path)
                        ):
                            is_valid = True

                        if not is_valid:
                            broken_links.append(
                                {
                                    "source": os.path.relpath(file_path),
                                    "link": link,
                                    "resolved": target_path,
                                }
                            )
    return broken_links


def main():
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
