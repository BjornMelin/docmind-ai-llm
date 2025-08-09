#!/usr/bin/env python3
"""Automated pylint issue fixer for docmind-ai-llm project."""

import re
import shutil
import subprocess
from pathlib import Path


def find_executable(command: str) -> str | None:
    """Find the executable path for a given command.

    Args:
        command: Name of the command to find.

    Returns:
        Full path to the executable or None if not found.
    """
    return shutil.which(command)


def run_pylint(file_path: str) -> dict:
    """Run pylint and parse JSON output.

    Args:
        file_path: Path to the Python file to lint.

    Returns:
        Dictionary of pylint results.
    """
    try:
        pylint_cmd = find_executable("pylint")
        if not pylint_cmd:
            raise FileNotFoundError("pylint not found")

        result = subprocess.run(
            [pylint_cmd, "--rcfile=.pylintrc", file_path, "--output-format=json"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "{}"
    except Exception as e:
        print(f"Error running pylint on {file_path}: {e}")
        return "{}"


def add_docstrings(file_path: Path):
    """Add missing docstrings to functions and classes.

    Args:
        file_path: Path to the Python file to modify.
    """
    with open(file_path) as f:
        content = f.read()

    # Add module docstring if missing
    if not re.search(r'""".*?"""', content[:500], re.DOTALL):
        content = f'"""\nModule documentation for {file_path.stem}.\n"""\n' + content

    # Detect functions and classes without docstrings
    def add_docstring(match):
        def_type, name = match.group(1), match.group(2)
        if not re.search(r'""".*?"""', match.group(0), re.DOTALL):
            default_docstring = f'"""\n{def_type.capitalize()} documentation for {name}.\n\nTODO: Add detailed description.\n"""\n'
            return default_docstring + match.group(0)
        return match.group(0)

    # Regex to find functions and classes without docstrings
    content = re.sub(
        r"^(def|class)\s+(\w+)\s*\(.*?\):",
        add_docstring,
        content,
        flags=re.MULTILINE | re.DOTALL,
    )

    with open(file_path, "w") as f:
        f.write(content)


def fix_imports(file_path: Path):
    """Remove unused imports and organize import order.

    Args:
        file_path: Path to the Python file to modify.
    """
    autoflake_cmd = find_executable("autoflake")
    isort_cmd = find_executable("isort")

    if autoflake_cmd:
        subprocess.run(
            [
                autoflake_cmd,
                "--in-place",
                "--remove-all-unused-imports",
                str(file_path),
            ],
            check=False,
        )

    if isort_cmd:
        subprocess.run([isort_cmd, str(file_path)], check=False)


def format_code(file_path: Path):
    """Apply consistent formatting.

    Args:
        file_path: Path to the Python file to modify.
    """
    black_cmd = find_executable("black")
    if black_cmd:
        subprocess.run([black_cmd, "--line-length", "100", str(file_path)], check=False)


def main():
    """Main script to fix pylint issues across project files."""
    files = [
        "models.py",
        "utils/utils.py",
        "utils/index_builder.py",
        "utils/document_loader.py",
        "utils/qdrant_utils.py",
        "utils/model_manager.py",
        "agents/agent_utils.py",
        "agent_factory.py",
        "app.py",
        "prompts.py",
    ]

    base_path = Path("/home/bjorn/repos/agents/docmind-ai-llm")

    for file_name in files:
        file_path = base_path / file_name
        print(f"Processing {file_name}...")

        # Order of fixes is important
        add_docstrings(file_path)
        fix_imports(file_path)
        format_code(file_path)

        # Run pylint to verify
        try:
            pylint_output = run_pylint(str(file_path))
            print(f"Pylint output for {file_name}: {pylint_output}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    main()
