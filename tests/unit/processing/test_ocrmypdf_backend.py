"""OCRmyPDF searchable-PDF utility tests."""

from __future__ import annotations

import asyncio
import signal
import subprocess
from pathlib import Path

import pytest

from src.processing.parsing.backends import ocrmypdf_backend

pytestmark = pytest.mark.unit


def test_export_searchable_pdf_uses_skip_text_and_tesseract_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_pdf = tmp_path / "input.pdf"
    output_pdf = tmp_path / "output.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    monkeypatch.setattr(
        ocrmypdf_backend.shutil,
        "which",
        lambda command: f"/usr/bin/{command}",
    )

    captured: dict[str, object] = {}

    class _Process:
        pid = 123
        returncode = 0

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            captured["timeout"] = timeout
            output_pdf.write_bytes(b"%PDF-1.4 searchable\n%%EOF\n")
            return "", ""

        def wait(self) -> None:
            raise AssertionError("successful OCR should not need forced reaping")

    def _popen(cmd: list[str], **kwargs: object) -> _Process:
        captured["cmd"] = cmd
        captured.update(kwargs)
        return _Process()

    monkeypatch.setattr(ocrmypdf_backend, "_popen", _popen)

    assert ocrmypdf_backend.export_searchable_pdf(input_pdf, output_pdf) == output_pdf
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--skip-text" in cmd
    assert str(input_pdf) in cmd
    assert str(output_pdf) in cmd
    assert captured["timeout"] == 300.0
    assert captured["start_new_session"] is True


def test_export_searchable_pdf_requires_optional_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_pdf = tmp_path / "input.pdf"
    output_pdf = tmp_path / "output.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setattr(ocrmypdf_backend.shutil, "which", lambda _command: None)

    with pytest.raises(RuntimeError, match="ocrmypdf CLI"):
        ocrmypdf_backend.export_searchable_pdf(input_pdf, output_pdf)


def test_export_timeout_kills_the_entire_ocr_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_pdf = tmp_path / "input.pdf"
    output_pdf = tmp_path / "output.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setattr(
        ocrmypdf_backend.shutil,
        "which",
        lambda command: f"/usr/bin/{command}",
    )
    reaped = False

    class _Process:
        pid = 456
        returncode = None

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            raise subprocess.TimeoutExpired(["ocrmypdf"], timeout)

        def wait(self) -> None:
            nonlocal reaped
            reaped = True

    monkeypatch.setattr(
        ocrmypdf_backend,
        "_popen",
        lambda *_args, **_kwargs: _Process(),
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        ocrmypdf_backend,
        "_killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        ocrmypdf_backend.export_searchable_pdf(input_pdf, output_pdf)

    assert signals == [(456, signal.SIGKILL)]
    assert reaped is True


@pytest.mark.asyncio
async def test_async_export_timeout_path_cancellation_kills_the_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_pdf = tmp_path / "input.pdf"
    output_pdf = tmp_path / "output.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setattr(
        ocrmypdf_backend.shutil,
        "which",
        lambda command: f"/usr/bin/{command}",
    )
    process_started = asyncio.Event()
    reaped = False

    class _Process:
        pid = 789
        returncode = None

        def poll(self) -> None:
            process_started.set()
            return None

        def wait(self) -> int:
            nonlocal reaped
            reaped = True
            return -signal.SIGKILL

    def _popen(*_args: object, **kwargs: object) -> _Process:
        assert kwargs["start_new_session"] is True
        return _Process()

    monkeypatch.setattr(ocrmypdf_backend, "_popen", _popen)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        ocrmypdf_backend,
        "_killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    task = asyncio.create_task(
        ocrmypdf_backend.export_searchable_pdf_async(input_pdf, output_pdf)
    )
    await process_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert signals == [(789, signal.SIGKILL)]
    assert reaped is True
