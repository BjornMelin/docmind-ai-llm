"""OCRmyPDF searchable-PDF utility backend."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

_killpg: Callable[[int, int], None] | None = getattr(os, "killpg", None)
_popen = subprocess.Popen


def ocrmypdf_health() -> dict[str, Any]:
    """Return availability diagnostics for OCRmyPDF and Tesseract."""
    return {
        "ocrmypdf": shutil.which("ocrmypdf") is not None,
        "tesseract": shutil.which("tesseract") is not None,
    }


def export_searchable_pdf(
    input_pdf: Path,
    output_pdf: Path,
    *,
    language: str = "eng",
    jobs: int = 1,
    timeout_seconds: float = 300.0,
) -> Path:
    """Create a searchable PDF artifact with OCRmyPDF.

    This utility is intentionally separate from the RAG parser path.
    """
    cmd = _ocrmypdf_command(
        input_pdf,
        output_pdf,
        language=language,
        jobs=jobs,
    )
    process = _popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except BaseException:
        _kill_process_group(process.pid)
        process.wait()
        raise
    if process.returncode:
        _kill_process_group(process.pid)
        raise subprocess.CalledProcessError(
            process.returncode,
            cmd,
            output=stdout,
            stderr=stderr,
        )
    return output_pdf


async def export_searchable_pdf_async(
    input_pdf: Path,
    output_pdf: Path,
    *,
    language: str = "eng",
    jobs: int = 1,
    timeout_seconds: float = 300.0,
) -> Path:
    """Create a searchable PDF in a cancellable subprocess group."""
    cmd = _ocrmypdf_command(
        input_pdf,
        output_pdf,
        language=language,
        jobs=jobs,
    )
    process = _popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    try:
        while process.poll() is None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, timeout_seconds)
            await asyncio.sleep(min(0.05, remaining))
    except BaseException:
        _kill_process_group(process.pid)
        process.wait()
        raise
    if process.returncode:
        _kill_process_group(process.pid)
        raise subprocess.CalledProcessError(
            process.returncode,
            cmd,
        )
    return output_pdf


def _ocrmypdf_command(
    input_pdf: Path,
    output_pdf: Path,
    *,
    language: str,
    jobs: int,
) -> list[str]:
    if os.name != "posix":
        raise RuntimeError("searchable PDF process containment requires POSIX")
    if shutil.which("ocrmypdf") is None:
        raise RuntimeError("ocrmypdf CLI is not installed")
    if shutil.which("tesseract") is None:
        raise RuntimeError("tesseract CLI is not installed")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    return [
        "ocrmypdf",
        "--skip-text",
        "--language",
        language,
        "--jobs",
        str(jobs),
        str(input_pdf),
        str(output_pdf),
    ]


def _kill_process_group(pid: int) -> None:
    if _killpg is None:
        return
    with suppress(ProcessLookupError):
        _killpg(pid, signal.SIGKILL)


__all__ = [
    "export_searchable_pdf",
    "export_searchable_pdf_async",
    "ocrmypdf_health",
]
