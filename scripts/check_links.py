#!/usr/bin/env python3
"""Check for broken links in markdown files."""

import os
import re
import sys

root_dir = os.path.abspath("docs")
broken_links = []

for root, _dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".md"):
            file_path = os.path.join(root, file)
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                # Simple markdown link regex
                # Improved regex to avoid false positives from code blocks
                # Matches [label](link) where label doesn't start with quotes or [
                found_links = re.findall(r"\[([^\"'\[\]][^\]]*?)\]\((.*?)\)", content)

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

                    # Check if file exists
                    if not os.path.exists(target_path):
                        # Try adding .md if missing
                        if not target_path.endswith(".md") and os.path.exists(
                            target_path + ".md"
                        ):
                            continue
                        # Try if it's a directory and has index.md
                        if os.path.isdir(target_path) and os.path.exists(
                            os.path.join(target_path, "index.md")
                        ):
                            continue
                        if os.path.isdir(target_path) and os.path.exists(
                            os.path.join(target_path, "_index.md")
                        ):
                            continue

                        broken_links.append(
                            {"source": file_path, "link": link, "resolved": target_path}
                        )

for bl in broken_links:
    print(
        f"Broken link in {bl['source']}: {bl['link']} (Resolved to: {bl['resolved']})"
    )

if broken_links:
    print(f"\nFound {len(broken_links)} broken links.")
    sys.exit(1)
else:
    print("\nAll internal links are valid.")
    sys.exit(0)
