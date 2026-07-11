"""pypdfium2 PDF inspector tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.processing.parsing.errors import DocumentParseError
from src.processing.parsing.pdf_inspector import inspect_pdf

pytestmark = pytest.mark.unit


def test_inspect_pdf_samples_text_with_pypdfium2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("pypdfium2")

    class _TextPage:
        def get_text_range(self) -> str:
            return "hello"

        def close(self) -> None:
            return None

    class _Page:
        def get_textpage(self) -> _TextPage:
            return _TextPage()

        def get_size(self) -> tuple[int, int]:
            return (200, 100)

        def close(self) -> None:
            return None

    class _Doc:
        def __init__(self, _path: Path) -> None:
            return None

        def __enter__(self) -> _Doc:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def __len__(self) -> int:
            return 2

        def __getitem__(self, _index: int) -> _Page:
            return _Page()

    module.PdfDocument = _Doc  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdfium2", module)

    result = inspect_pdf(Path("sample.pdf"), sample_pages=1)

    assert result.page_count == 2
    assert result.has_native_text is True
    assert result.pages[0].width == 200


def test_inspect_pdf_rejects_malformed_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed PDF syntax is a typed failure, not an empty inspection."""
    module = ModuleType("pypdfium2")

    class _MalformedDocument:
        def __init__(self, _path: Path) -> None:
            raise ValueError("invalid PDF structure")

    module.PdfDocument = _MalformedDocument  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdfium2", module)

    with pytest.raises(DocumentParseError) as raised:
        inspect_pdf(Path("malformed.pdf"))

    assert raised.value.stage == "pdf_inspection"
    assert raised.value.reason == "invalid_or_unreadable_pdf"
    assert raised.value.source_suffix == ".pdf"


def test_inspect_pdf_preserves_page_limit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resource-limit failures must not be mislabeled as malformed files."""
    module = ModuleType("pypdfium2")

    class _DocumentOverLimit:
        def __init__(self, _path: Path) -> None:
            return None

        def __enter__(self) -> _DocumentOverLimit:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def __len__(self) -> int:
            return 2

    module.PdfDocument = _DocumentOverLimit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdfium2", module)

    with pytest.raises(DocumentParseError) as raised:
        inspect_pdf(Path("large.pdf"), max_pages=1)

    assert raised.value.stage == "pdf_inspection"
    assert raised.value.reason == "page_limit_exceeded"
