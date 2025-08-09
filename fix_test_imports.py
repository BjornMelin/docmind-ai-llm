#!/usr/bin/env python
"""Fix import issues across all test files by adding sys.path modifications."""

from pathlib import Path


def has_sys_path_fix(content: str) -> bool:
    """Check if the file already has sys.path.insert() fix."""
    return "sys.path.insert(0" in content


def add_import_fix(content: str) -> str:
    """Add import path fix to test file content."""
    import_fix = """import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent))

"""

    # Find the first import statement
    lines = content.split("\n")

    # Skip docstring and comments at the start
    insert_line = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if (
            line
            and not line.startswith('"""')
            and not line.startswith('"""')
            and not line.startswith("#")
        ):
            # If this is an import line, insert before it
            if line.startswith("import ") or line.startswith("from "):
                insert_line = i
                break

    # Insert the fix
    lines.insert(insert_line, import_fix.rstrip())
    return "\n".join(lines)


def fix_test_file(file_path: Path) -> bool:
    """Fix imports in a single test file."""
    try:
        content = file_path.read_text()

        # Skip if already has the fix
        if has_sys_path_fix(content):
            print(f"✓ {file_path.name} already has import fix")
            return False

        # Add the fix
        fixed_content = add_import_fix(content)
        file_path.write_text(fixed_content)
        print(f"✓ Added import fix to {file_path.name}")
        return True

    except Exception as e:
        print(f"✗ Error fixing {file_path.name}: {e}")
        return False


def main():
    """Fix all test files in the tests directory."""
    test_dir = Path(__file__).parent / "tests"
    test_files = list(test_dir.glob("test_*.py"))

    print(f"Found {len(test_files)} test files")
    print("=" * 50)

    fixed_count = 0

    for test_file in test_files:
        if fix_test_file(test_file):
            fixed_count += 1

    print("=" * 50)
    print(f"Fixed {fixed_count} test files")

    # Special handling for conftest.py (already has the fix but let's verify)
    conftest_path = test_dir / "conftest.py"
    if conftest_path.exists():
        content = conftest_path.read_text()
        if has_sys_path_fix(content):
            print("✓ conftest.py already has import fix")
        else:
            print("✗ conftest.py missing import fix")


if __name__ == "__main__":
    main()
