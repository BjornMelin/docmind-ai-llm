"""Canonical parsing service used by the ingestion API."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import signal
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from threading import Lock, Thread
from types import FrameType
from typing import Any

from src.config.settings import DocMindSettings
from src.persistence.artifacts import ArtifactStore
from src.persistence.hashing import compute_config_hash
from src.processing.parsing.backends.docling_backend import (
    convert_with_docling,
    verify_docling_layout_models,
)
from src.processing.parsing.backends.ocrmypdf_backend import (
    export_searchable_pdf,
    export_searchable_pdf_async,
)
from src.processing.parsing.backends.rapidocr_backend import (
    run_rapidocr,
    verify_rapidocr_models,
)
from src.processing.parsing.canonical_types import (
    DocumentParseResult,
    OcrEngineName,
    PageParseResult,
    ParserFramework,
    ParserProfile,
    ParserVersions,
    ParsingArtifact,
    PdfBackendName,
)
from src.processing.parsing.errors import DocumentParseError
from src.processing.parsing.formats import read_direct_text
from src.processing.parsing.health import parser_health
from src.processing.parsing.model_artifacts import ModelIntegrityError
from src.processing.parsing.pdf_inspector import PdfInspectionResult, inspect_pdf
from src.processing.parsing.router import (
    choose_framework,
    routing_reason,
)
from src.utils.hashing import document_id_from_sha256, sha256_file

_VERSION_PACKAGES = (
    "docling",
    "pypdfium2",
    "rapidocr",
    "ocrmypdf",
)
_killpg: Callable[[int, int], None] | None = getattr(os, "killpg", None)
_ParserWorker = tuple[BaseProcess, Connection]


async def parse_document(
    path: Path,
    *,
    settings: DocMindSettings,
    document_id: str | None = None,
) -> DocumentParseResult:
    """Parse one local document in a killable worker process.

    Args:
        path: Local source document to parse.
        settings: Application settings controlling parsing and OCR behavior.
        document_id: Optional stable document identifier supplied by the caller.

    Returns:
        DocumentParseResult: Canonical parser output for the document.

    Raises:
        DocumentParseError: If the worker fails, times out, or returns invalid
            parser output.
    """
    timeout = float(settings.parsing.parse_timeout_seconds)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    worker: _ParserWorker | None = None
    completed = False
    primary_error: BaseException | None = None
    try:
        worker = await _start_parse_worker_async(
            path,
            settings=settings,
            document_id=document_id,
            timeout_seconds=max(0.0, deadline - loop.time()),
        )
        process, receiver = worker
        while not receiver.poll():
            if not process.is_alive():
                raise DocumentParseError(
                    path,
                    stage="parser_worker",
                    reason="worker_exited_without_result",
                )
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise DocumentParseError(
                    path,
                    stage="parse_timeout",
                    reason="parse_timeout_exceeded",
                )
            await asyncio.sleep(min(0.05, remaining))
        result = await _receive_worker_before_deadline(
            path,
            receiver,
            timeout_seconds=deadline - loop.time(),
        )
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise DocumentParseError(
                path,
                stage="parse_timeout",
                reason="parse_timeout_exceeded",
            )
        if settings.ocr.searchable_pdf_enabled:
            await _attach_searchable_pdf_artifact_async(
                result,
                path,
                settings=settings,
                timeout_seconds=_bounded_ocrmypdf_timeout(
                    settings=settings,
                    remaining_seconds=remaining,
                ),
            )
            if loop.time() >= deadline:
                raise DocumentParseError(
                    path,
                    stage="parse_timeout",
                    reason="parse_timeout_exceeded",
                )
        completed = True
        return result
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if worker is not None:
            await _cleanup_parse_worker_preserving_error(
                path,
                worker=worker,
                force=not completed,
                primary_error=primary_error,
            )


async def _start_parse_worker_async(
    path: Path,
    *,
    settings: DocMindSettings,
    document_id: str | None,
    timeout_seconds: float,
) -> _ParserWorker:
    """Start a worker off-loop and transfer cleanup if the caller stops waiting."""
    owner = _WorkerStartOwner(
        path,
        settings=settings,
        document_id=document_id,
    )
    wrapped_start = asyncio.wrap_future(owner.future)
    try:
        worker = await asyncio.wait_for(
            asyncio.shield(wrapped_start),
            timeout=max(0.0, timeout_seconds),
        )
    except TimeoutError as exc:
        wrapped_start.add_done_callback(_consume_task_result)
        owner.abandon()
        raise DocumentParseError(
            path,
            stage="parse_timeout",
            reason="parse_timeout_exceeded",
            cause=exc,
        ) from exc
    except asyncio.CancelledError:
        wrapped_start.add_done_callback(_consume_task_result)
        owner.abandon()
        raise
    owner.claim()
    return worker


async def _receive_worker_before_deadline(
    path: Path,
    receiver: Connection,
    *,
    timeout_seconds: float,
) -> DocumentParseResult:
    """Receive and decode off-loop without exceeding the parser deadline."""
    if timeout_seconds <= 0:
        raise DocumentParseError(
            path,
            stage="parse_timeout",
            reason="parse_timeout_exceeded",
        )
    receive_task = asyncio.create_task(
        asyncio.to_thread(_receive_and_decode_worker_payload, path, receiver)
    )
    try:
        return await asyncio.wait_for(
            asyncio.shield(receive_task),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        receive_task.add_done_callback(_consume_task_result)
        raise DocumentParseError(
            path,
            stage="parse_timeout",
            reason="parse_timeout_exceeded",
            cause=exc,
        ) from exc
    except asyncio.CancelledError:
        receive_task.add_done_callback(_consume_task_result)
        raise


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    """Observe a detached task result so its exception is not leaked."""
    with suppress(asyncio.CancelledError, Exception):
        task.result()


class _WorkerStartOwner:
    """Own a blocking worker start until the async caller claims or abandons it."""

    def __init__(
        self,
        path: Path,
        *,
        settings: DocMindSettings,
        document_id: str | None,
    ) -> None:
        self._path = path
        self._settings = settings
        self._document_id = document_id
        self._future: Future[_ParserWorker] = Future()
        self._future.set_running_or_notify_cancel()
        self._lock = Lock()
        self._abandoned = False
        self._claimed = False
        self._cleanup_started = False
        self._thread = Thread(
            target=self._run,
            name="docmind-parser-start",
        )
        try:
            self._thread.start()
        except Exception as exc:
            self._future.set_exception(exc)
            raise DocumentParseError(
                path,
                stage="parser_worker",
                reason="worker_start_failed",
                cause=exc,
            ) from exc

    @property
    def future(self) -> Future[_ParserWorker]:
        """Return the loop-independent startup result future."""
        return self._future

    def claim(self) -> None:
        """Transfer worker ownership to the successful async caller."""
        with self._lock:
            self._claimed = True

    def abandon(self) -> None:
        """Ensure an unclaimed worker is reaped even after event-loop shutdown."""
        worker: _ParserWorker | None = None
        with self._lock:
            if self._claimed or self._abandoned:
                return
            self._abandoned = True
            if self._future.done() and not self._cleanup_started:
                with suppress(BaseException):
                    worker = self._future.result()
                if worker is not None:
                    self._cleanup_started = True
        if worker is not None:
            _spawn_worker_cleanup(worker)

    def _run(self) -> None:
        try:
            worker = _start_parse_worker(
                self._path,
                settings=self._settings,
                document_id=self._document_id,
            )
        except BaseException as exc:
            self._future.set_exception(exc)
            return
        cleanup = False
        with self._lock:
            self._future.set_result(worker)
            if self._abandoned and not self._cleanup_started:
                self._cleanup_started = True
                cleanup = True
        if cleanup:
            _close_and_stop_worker(worker)


def _spawn_worker_cleanup(worker: _ParserWorker) -> None:
    """Start loop-independent cleanup for an already completed abandoned start."""
    cleanup_thread = Thread(
        target=_close_and_stop_worker,
        args=(worker,),
        name="docmind-parser-abandoned-cleanup",
    )
    try:
        cleanup_thread.start()
    except RuntimeError:
        _close_and_stop_worker(worker)


def _close_and_stop_worker(worker: _ParserWorker) -> None:
    """Best-effort synchronous cleanup owned independently of the event loop."""
    process, receiver = worker
    with suppress(Exception):
        receiver.close()
    with suppress(Exception):
        _stop_parse_worker(process, force=True)


def _receive_and_decode_worker_payload(
    path: Path,
    receiver: Connection,
) -> DocumentParseResult:
    """Receive and validate a worker result outside the event-loop thread."""
    try:
        payload = receiver.recv()
    except (EOFError, OSError) as exc:
        raise DocumentParseError(
            path,
            stage="parser_worker",
            reason="worker_result_unavailable",
            cause=exc,
        ) from exc
    return _decode_worker_payload(path, payload)


async def _cleanup_parse_worker(
    *,
    process: BaseProcess,
    receiver: Connection,
    force: bool,
) -> None:
    """Close the receive channel and reap the worker without blocking the loop."""
    cleanup_error: Exception | None = None
    try:
        receiver.close()
    except Exception as exc:
        cleanup_error = exc
    stop_task = asyncio.create_task(
        asyncio.to_thread(_stop_parse_worker, process, force=force)
    )
    try:
        await asyncio.shield(stop_task)
    except asyncio.CancelledError:
        stop_task.add_done_callback(_consume_task_result)
        raise
    except Exception as exc:
        if cleanup_error is None:
            cleanup_error = exc
    if cleanup_error is not None:
        raise cleanup_error


async def _cleanup_parse_worker_preserving_error(
    path: Path,
    *,
    worker: _ParserWorker,
    force: bool,
    primary_error: BaseException | None,
) -> None:
    """Clean a worker without replacing an in-flight parse failure."""
    process, receiver = worker
    try:
        await _cleanup_parse_worker(
            process=process,
            receiver=receiver,
            force=force,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if primary_error is None:
            raise DocumentParseError(
                path,
                stage="parser_worker",
                reason="worker_cleanup_failed",
                cause=exc,
            ) from exc


def _start_parse_worker(
    path: Path,
    *,
    settings: DocMindSettings,
    document_id: str | None,
) -> tuple[BaseProcess, Connection]:
    """Start an isolated parser worker and return its receive channel."""
    receiver: Connection | None = None
    sender: Connection | None = None
    process: BaseProcess | None = None
    try:
        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_parse_worker,
            args=(sender, str(path), settings.model_dump(mode="python"), document_id),
            name="docmind-parser",
        )
        process.start()
        sender.close()
        return process, receiver
    except Exception as exc:
        _close_failed_worker_start_resources(
            receiver=receiver,
            sender=sender,
            process=process,
        )
        raise DocumentParseError(
            path,
            stage="parser_worker",
            reason="worker_start_failed",
            cause=exc,
        ) from exc


def _close_failed_worker_start_resources(
    *,
    receiver: Connection | None,
    sender: Connection | None,
    process: BaseProcess | None,
) -> None:
    """Close resources allocated before parser-worker startup failed."""
    for connection in (receiver, sender):
        if connection is not None:
            with suppress(OSError):
                connection.close()
    if process is None:
        return
    with suppress(AssertionError, OSError, ValueError):
        if process.is_alive():
            process.terminate()
    with suppress(AssertionError, OSError, ValueError):
        process.join(timeout=1.0)
    with suppress(AssertionError, OSError, ValueError):
        if process.is_alive():
            process.kill()
    with suppress(AssertionError, OSError, ValueError):
        process.join(timeout=1.0)
    with suppress(OSError, ValueError):
        process.close()


def _parse_worker(
    sender: Connection,
    path_value: str,
    settings_payload: dict[str, Any],
    document_id: str | None,
) -> None:
    """Parse and send a JSON result or metadata-only failure to the parent."""
    path = Path(path_value)
    try:
        _own_parse_worker_process_group()
        worker_settings = DocMindSettings.model_validate(settings_payload)
        result = parse_document_sync(
            path,
            settings=worker_settings,
            document_id=document_id,
            _include_searchable_pdf=False,
        )
        sender.send(("ok", result.model_dump_json()))
    except DocumentParseError as exc:
        sender.send(
            (
                "parse_error",
                {
                    "stage": exc.stage,
                    "reason": exc.reason,
                    "cause_type": exc.cause_type,
                },
            )
        )
    except Exception as exc:
        sender.send(("worker_error", {"cause_type": type(exc).__name__}))
    finally:
        sender.close()


def _decode_worker_payload(path: Path, payload: object) -> DocumentParseResult:
    """Validate the parser worker's small tagged response contract."""
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise DocumentParseError(
            path,
            stage="parser_worker",
            reason="invalid_worker_response",
        )
    status, value = payload
    if status == "ok" and isinstance(value, str):
        try:
            return DocumentParseResult.model_validate_json(value)
        except ValueError as exc:
            raise DocumentParseError(
                path,
                stage="parser_worker",
                reason="invalid_worker_result",
                cause=exc,
            ) from exc
    if status == "parse_error" and isinstance(value, dict):
        stage = value.get("stage")
        reason = value.get("reason")
        cause_type = value.get("cause_type")
        if (
            isinstance(stage, str)
            and isinstance(reason, str)
            and (cause_type is None or isinstance(cause_type, str))
        ):
            raise DocumentParseError(
                path,
                stage=stage,
                reason=reason,
                cause_type=cause_type,
            )
    if status == "worker_error" and isinstance(value, dict):
        cause_type = value.get("cause_type")
        if isinstance(cause_type, str):
            raise DocumentParseError(
                path,
                stage="parser_worker",
                reason="worker_failed",
                cause_type=cause_type,
            )
    raise DocumentParseError(
        path,
        stage="parser_worker",
        reason="invalid_worker_response",
    )


def _stop_parse_worker(process: BaseProcess, *, force: bool) -> None:
    """Reap a parser worker and every process in its private POSIX group."""
    if not force and process.is_alive():
        process.join(timeout=1.0)
    if process.is_alive():
        if not _signal_parse_worker_group(process, signal.SIGTERM):
            process.terminate()
        process.join(timeout=1.0)
    group_killed = _signal_parse_worker_group(process, signal.SIGKILL)
    if process.is_alive() and not group_killed:
        process.kill()
    process.join()


def _own_parse_worker_process_group() -> None:
    if os.name != "posix":
        return
    os.setsid()
    signal.signal(signal.SIGTERM, _kill_worker_group_on_terminate)


def _kill_worker_group_on_terminate(
    _signum: int,
    _frame: FrameType | None,
) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if _killpg is None:
        raise RuntimeError("POSIX process-group termination is unavailable")
    _killpg(os.getpgrp(), signal.SIGKILL)


def _signal_parse_worker_group(process: BaseProcess, signal_number: int) -> bool:
    pid = process.pid
    if os.name != "posix" or pid is None or _killpg is None:
        return False
    try:
        _killpg(pid, signal_number)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def parse_document_sync(
    path: Path,
    *,
    settings: DocMindSettings,
    document_id: str | None = None,
    _include_searchable_pdf: bool = True,
) -> DocumentParseResult:
    """Parse one local document and return canonical parser output.

    Args:
        path: Local source document to parse.
        settings: Application settings controlling parsing and OCR behavior.
        document_id: Optional stable document identifier supplied by the caller.
        _include_searchable_pdf: Whether to attach the optional searchable-PDF
            artifact during synchronous parsing.

    Returns:
        DocumentParseResult: Canonical parser output for the document.

    Raises:
        DocumentParseError: If validation, inspection, conversion, OCR, or required
            parser post-processing fails. Optional searchable-PDF artifact failures
            remain fail-open.
    """
    source_path = Path(path)
    _validate_document_size(source_path, settings=settings)
    if source_path.suffix.lower() == ".pdf":
        _require_pdf_models(source_path, settings=settings)
    inspection = _inspect_pdf_once(source_path, settings=settings)
    try:
        source_hash = sha256_file(source_path)
    except OSError as exc:
        raise DocumentParseError(
            source_path,
            stage="source_hash",
            reason="source_hash_failed",
            cause=exc,
        ) from exc
    doc_id = document_id or document_id_from_sha256(source_hash)
    framework = choose_framework(source_path)
    versions = ParserVersions(packages=_package_versions())
    reason = routing_reason(
        source_path,
        inspection=inspection,
        force_ocr=settings.ocr.force_ocr,
    )

    if framework is ParserFramework.DIRECT_TEXT:
        text = read_direct_text(
            source_path,
            max_bytes=_max_document_bytes(settings),
            probe_bytes=settings.parsing.direct_text_probe_bytes,
        )
        result = _single_page_result(
            path=source_path,
            document_id=doc_id,
            source_hash=source_hash,
            framework=ParserFramework.DIRECT_TEXT,
            text=text,
            reason=reason,
            settings=settings,
            versions=versions,
        )
        _validate_total_text(result, source_path, settings=settings)
        return result

    if framework is ParserFramework.DOCLING:
        try:
            result = convert_with_docling(
                source_path,
                document_id=doc_id,
                source_hash=source_hash,
                model_cache_dir=settings.ocr.model_cache_dir,
                max_pages=settings.parsing.max_pages,
                max_file_size=_max_document_bytes(settings),
                versions=versions,
            )
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError(
                source_path,
                stage="docling_conversion",
                reason="conversion_failed",
                cause=exc,
            ) from exc
        try:
            _apply_parser_metadata(result, settings=settings)
            _apply_pdf_page_routing(
                result,
                source_path,
                inspection=inspection,
                settings=settings,
            )
            _validate_total_text(result, source_path, settings=settings)
            if _include_searchable_pdf:
                _attach_searchable_pdf_artifact(
                    result,
                    source_path,
                    settings=settings,
                )
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError(
                source_path,
                stage="post_conversion",
                reason="parser_post_processing_failed",
                cause=exc,
            ) from exc
        result.routing_decisions.append({"source": source_path.name, "reason": reason})
        return result

    raise DocumentParseError(
        source_path,
        stage="framework_selection",
        reason=f"unsupported_framework_{framework.value}",
    )


def _single_page_result(
    *,
    path: Path,
    document_id: str,
    source_hash: str,
    framework: ParserFramework,
    text: str,
    reason: str,
    settings: DocMindSettings,
    versions: ParserVersions,
) -> DocumentParseResult:
    result = DocumentParseResult(
        document_id=document_id,
        source_filename=path.name,
        source_hash=source_hash,
        profile=ParserProfile.CPU_SAFE,
        parser_framework=framework,
        pdf_backend=PdfBackendName.PYPDFIUM2,
        ocr_engine=OcrEngineName.RAPIDOCR,
        versions=versions,
        page_count=1,
        pages=[
            PageParseResult(
                page_id=f"{document_id}::page::1",
                page_index=0,
                text_markdown=text,
                routing_reason=reason,
                ocr_applied=False,
            )
        ],
        config_hash=_parsing_config_hash(settings),
        health=parser_health(settings),
    )
    result.routing_decisions.append({"source": path.name, "reason": reason})
    return result


def _apply_parser_metadata(
    result: DocumentParseResult, *, settings: DocMindSettings
) -> None:
    result.profile = ParserProfile.CPU_SAFE
    result.ocr_engine = OcrEngineName.RAPIDOCR
    result.config_hash = _parsing_config_hash(settings)
    result.health = parser_health(settings)


def _inspect_pdf_once(
    path: Path, settings: DocMindSettings
) -> PdfInspectionResult | None:
    if path.suffix.lower() != ".pdf":
        return None
    return inspect_pdf(
        path,
        sample_pages=settings.parsing.max_pages,
        max_text_chars=8000,
        max_pages=settings.parsing.max_pages,
        render_dpi=settings.pdf_backend.render_dpi,
        max_render_pixels=settings.parsing.max_render_pixels,
    )


def _apply_pdf_page_routing(
    result: DocumentParseResult,
    path: Path,
    *,
    inspection: PdfInspectionResult | None,
    settings: DocMindSettings,
) -> None:
    if (
        path.suffix.lower() != ".pdf"
        or inspection is None
        or inspection.page_count <= 0
    ):
        return
    if result.page_count != inspection.page_count or len(inspection.pages) != (
        inspection.page_count
    ):
        raise DocumentParseError(
            path,
            stage="page_fidelity",
            reason="physical_page_count_mismatch",
        )
    if not settings.ocr.force_ocr and not any(
        _page_needs_ocr(page_text=page.text, settings=settings)
        for page in inspection.pages
    ):
        return

    ocr_page_indexes = [
        page.page_index
        for page in inspection.pages
        if settings.ocr.force_ocr
        or _page_needs_ocr(page_text=page.text, settings=settings)
    ]
    ocr_text = _ocr_pdf_pages(path, ocr_page_indexes, settings=settings)
    pages: list[PageParseResult] = []
    for page in inspection.pages:
        needs_ocr = settings.ocr.force_ocr or _page_needs_ocr(
            page_text=page.text,
            settings=settings,
        )
        native_page = result.pages[page.page_index]
        text = native_page.text_markdown or page.text
        reason = "pdf_native_text"
        if needs_ocr:
            text = ocr_text[page.page_index]
            if not text.strip():
                raise DocumentParseError(
                    path,
                    stage="rapidocr",
                    reason="empty_ocr_output",
                )
            reason = "pdf_page_rapidocr"
        pages.append(
            PageParseResult(
                page_id=f"{result.document_id}::page::{page.page_index + 1}",
                page_index=page.page_index,
                text_markdown=text,
                ocr_applied=needs_ocr,
                routing_reason=reason,
            )
        )
    if pages:
        result.pages = pages
        result.page_count = len(pages)


def _page_needs_ocr(*, page_text: str, settings: DocMindSettings) -> bool:
    return len(page_text.strip()) < int(settings.pdf_backend.min_text_chars_per_page)


def _ocr_pdf_pages(
    path: Path,
    page_indexes: list[int],
    *,
    settings: DocMindSettings,
) -> dict[int, str]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pypdfium2 is required for PDF OCR rendering") from exc

    if not page_indexes:
        return {}
    output: dict[int, str] = {}
    with (
        tempfile.TemporaryDirectory(prefix="docmind-ocr-") as tmp,
        pdfium.PdfDocument(path) as doc,
    ):
        for page_index in page_indexes:
            image_path = Path(tmp) / f"page-{page_index + 1}.png"
            page = doc[page_index]
            try:
                bitmap = page.render(scale=settings.pdf_backend.render_dpi / 72.0)
                try:
                    bitmap.to_pil().save(image_path)
                finally:
                    bitmap.close()
            finally:
                page.close()
            output[page_index] = run_rapidocr(
                image_path,
                model_cache_dir=settings.ocr.model_cache_dir,
            )
    return output


def _attach_searchable_pdf_artifact(
    result: DocumentParseResult,
    path: Path,
    *,
    settings: DocMindSettings,
) -> None:
    if path.suffix.lower() != ".pdf" or not settings.ocr.searchable_pdf_enabled:
        return
    with tempfile.TemporaryDirectory(prefix="docmind-searchable-pdf-") as tmp:
        output_pdf = Path(tmp) / f"{path.stem}.searchable.pdf"
        try:
            export_searchable_pdf(
                path,
                output_pdf,
                jobs=settings.ocr.ocrmypdf_jobs,
                timeout_seconds=settings.parsing.ocrmypdf_timeout_seconds,
            )
            store = ArtifactStore.from_settings(settings)
            ref = store.put_file(output_pdf)
        except (
            RuntimeError,
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            result.routing_decisions.append(
                {
                    "source": path.name,
                    "reason": "searchable_pdf_unavailable",
                    "error_type": type(exc).__name__,
                }
            )
            return
        result.artifacts.append(
            ParsingArtifact(
                kind="searchable_pdf",
                artifact_id=ref.sha256,
                suffix=ref.suffix,
                metadata={"source_filename": path.name},
            )
        )


async def _attach_searchable_pdf_artifact_async(
    result: DocumentParseResult,
    path: Path,
    *,
    settings: DocMindSettings,
    timeout_seconds: float,
) -> None:
    if path.suffix.lower() != ".pdf" or not settings.ocr.searchable_pdf_enabled:
        return
    with tempfile.TemporaryDirectory(prefix="docmind-searchable-pdf-") as tmp:
        output_pdf = Path(tmp) / f"{path.stem}.searchable.pdf"
        try:
            await export_searchable_pdf_async(
                path,
                output_pdf,
                jobs=settings.ocr.ocrmypdf_jobs,
                timeout_seconds=timeout_seconds,
            )
            store = ArtifactStore.from_settings(settings)
            ref = store.put_file(output_pdf)
        except (
            RuntimeError,
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            result.routing_decisions.append(
                {
                    "source": path.name,
                    "reason": "searchable_pdf_unavailable",
                    "error_type": type(exc).__name__,
                }
            )
            return
        result.artifacts.append(
            ParsingArtifact(
                kind="searchable_pdf",
                artifact_id=ref.sha256,
                suffix=ref.suffix,
                metadata={"source_filename": path.name},
            )
        )


def _bounded_ocrmypdf_timeout(
    *,
    settings: DocMindSettings,
    remaining_seconds: float,
) -> float:
    return min(
        float(settings.parsing.ocrmypdf_timeout_seconds),
        remaining_seconds * 0.9,
    )


def _package_versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for package in _VERSION_PACKAGES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            continue
    return found


def _max_document_bytes(settings: DocMindSettings) -> int:
    return int(settings.processing.max_document_size_mb) * 1024 * 1024


def _validate_document_size(path: Path, *, settings: DocMindSettings) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DocumentParseError(
            path,
            stage="resource_validation",
            reason="document_stat_failed",
            cause=exc,
        ) from exc
    if size > _max_document_bytes(settings):
        raise DocumentParseError(
            path,
            stage="resource_validation",
            reason="document_size_limit_exceeded",
        )


def _require_pdf_models(path: Path, *, settings: DocMindSettings) -> None:
    try:
        verify_docling_layout_models(settings.ocr.model_cache_dir)
        verify_rapidocr_models(settings.ocr.model_cache_dir)
    except (ModelIntegrityError, OSError) as exc:
        raise DocumentParseError(
            path,
            stage="parser_readiness",
            reason="parser_models_not_ready",
            cause=exc,
        ) from exc


def _validate_total_text(
    result: DocumentParseResult,
    path: Path,
    *,
    settings: DocMindSettings,
) -> None:
    if not any(page.text_markdown.strip() for page in result.pages):
        raise DocumentParseError(
            path,
            stage="resource_validation",
            reason="empty_text_output",
        )
    total = sum(len(page.text_markdown) for page in result.pages)
    if total > settings.parsing.max_total_text_chars:
        raise DocumentParseError(
            path,
            stage="resource_validation",
            reason="text_limit_exceeded",
        )


def _parsing_config_hash(settings: DocMindSettings) -> str:
    return compute_config_hash(
        {
            "parsing": settings.parsing.model_dump(mode="json"),
            "ocr": settings.ocr.model_dump(mode="json"),
            "pdf_backend": settings.pdf_backend.model_dump(mode="json"),
        }
    )


__all__ = ["parse_document", "parse_document_sync"]
