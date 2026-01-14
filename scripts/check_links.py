#!/usr/bin/env python3
"""Check for broken links in markdown files."""

import argparse
import os
import re
import sys


def check_links(search_paths: list[str]) -> list[dict]:
    """Scan markdown files in paths for broken internal links."""
    broken_links = []

    for search_path in search_paths:
        abs_search_path = os.path.abspath(search_path)
        if not os.path.exists(abs_search_path):
            print(f"Warning: Path {abs_search_path} does not exist. Skipping.")
            continue

        for root, dirs, files in os.walk(abs_search_path):
            # Skip hidden directories like .git
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

                    # Simple markdown link regex
                    # Improved regex to avoid false positives from code blocks
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
                                )
                            )
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
