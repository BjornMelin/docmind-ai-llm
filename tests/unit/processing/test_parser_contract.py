"""Canonical CPU-safe parser contract tests."""

from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.settings import DocMindSettings
from src.processing.parsing import service
from src.processing.parsing.backends import docling_backend, rapidocr_backend
from src.processing.parsing.backends.docling_backend import (
    docling_layout_model_files,
    missing_docling_layout_models,
)
from src.processing.parsing.backends.rapidocr_backend import rapidocr_model_files
from src.processing.parsing.canonical_types import (
    DocumentParseResult,
    PageParseResult,
    ParserFramework,
    ParserProfile,
    ParserVersions,
)
from src.processing.parsing.errors import DocumentParseError
from src.processing.parsing.model_artifacts import ModelIntegrityError
from src.processing.parsing.router import choose_framework

pytestmark = pytest.mark.unit


def _trust_pdf_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        service,
        "verify_docling_layout_models",
        lambda model_cache_dir: model_cache_dir,
    )
    monkeypatch.setattr(
        service,
        "verify_rapidocr_models",
        lambda model_cache_dir: model_cache_dir,
    )


def test_parser_submodule_import_does_not_load_application_stack() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "import src.processing.parsing.pdf_inspector; "
                "from src.processing.parsing.backends.docling_backend import "
                "docling_layout_model_files; docling_layout_model_files(Path('.')); "
                "forbidden = {'qdrant_client', 'torch', "
                "'src.processing.ingestion_pipeline'}; "
                "loaded = sorted(forbidden & sys.modules.keys()); "
                "assert not loaded, loaded"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_rapidocr_bounds_native_onnxruntime_threads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _rapidocr(*, params: dict[str, object]) -> object:
        captured.update(params)
        return object()

    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=_rapidocr))
    monkeypatch.setattr(rapidocr_backend, "verify_rapidocr_models", lambda _path: _path)
    rapidocr_backend._rapidocr_engine.cache_clear()
    try:
        rapidocr_backend._rapidocr_engine(str(tmp_path))
    finally:
        rapidocr_backend._rapidocr_engine.cache_clear()

    assert captured["EngineConfig.onnxruntime.intra_op_num_threads"] == 4
    assert captured["EngineConfig.onnxruntime.inter_op_num_threads"] == 1


def test_docling_offline_readiness_requires_weights_and_configs(
    tmp_path: Path,
) -> None:
    files = docling_layout_model_files(tmp_path)

    assert set(files) == {
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    }
    assert set(missing_docling_layout_models(tmp_path)) == set(files)

    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    assert missing_docling_layout_models(tmp_path) == []


class _NeverReadyConnection:
    """Minimal receive channel used to exercise timeout cleanup."""

    closed = False

    def poll(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


class _LiveProcess:
    """Minimal live process used to verify forced termination."""

    pid = 987_654
    terminated = False

    def is_alive(self) -> bool:
        return not self.terminated

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def join(self, timeout: float | None = None) -> None:
        del timeout


@pytest.mark.asyncio
async def test_async_parse_timeout_terminates_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "slow.md"
    source.write_text("bounded", encoding="utf-8")
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    process = _LiveProcess()
    receiver = _NeverReadyConnection()
    monkeypatch.setattr(
        service,
        "_start_parse_worker",
        lambda *_args, **_kwargs: (process, receiver),
    )
    times = iter((0.0, 301.0))
    loop = SimpleNamespace(time=lambda: next(times))
    monkeypatch.setattr(
        service,
        "asyncio",
        SimpleNamespace(get_running_loop=lambda: loop),
    )
    cleanup: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        service,
        "_stop_parse_worker",
        lambda stopped, *, force: cleanup.append((stopped, force)),
    )

    with pytest.raises(DocumentParseError) as raised:
        await service.parse_document(source, settings=cfg)

    assert raised.value.stage == "parse_timeout"
    assert receiver.closed is True
    assert cleanup == [(process, True)]


@pytest.mark.asyncio
async def test_async_parse_cancellation_stops_the_worker_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "cancelled.md"
    source.write_text("bounded", encoding="utf-8")
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    process = _LiveProcess()
    receiver = _NeverReadyConnection()
    monkeypatch.setattr(
        service,
        "_start_parse_worker",
        lambda *_args, **_kwargs: (process, receiver),
    )

    async def _cancel(_delay: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        service,
        "asyncio",
        SimpleNamespace(
            get_running_loop=lambda: SimpleNamespace(time=lambda: 0.0),
            sleep=_cancel,
        ),
    )
    cleanup: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        service,
        "_stop_parse_worker",
        lambda stopped, *, force: cleanup.append((stopped, force)),
    )

    with pytest.raises(asyncio.CancelledError):
        await service.parse_document(source, settings=cfg)

    assert receiver.closed is True
    assert cleanup == [(process, True)]


def test_worker_cleanup_kills_residual_process_group_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    class _GroupedProcess:
        pid = 2468
        alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            raise AssertionError("the POSIX process group should own termination")

        def kill(self) -> None:
            raise AssertionError("the POSIX process group should own termination")

        def join(self, timeout: float | None = None) -> None:
            del timeout
            if signals:
                self.alive = False

    process = _GroupedProcess()
    monkeypatch.setattr(
        service,
        "_killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    service._stop_parse_worker(process, force=True)  # type: ignore[arg-type]

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]


def test_parser_worker_excludes_optional_searchable_pdf_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    captured: dict[str, object] = {}

    class _Sender:
        closed = False

        def send(self, payload: object) -> None:
            captured["payload"] = payload

        def close(self) -> None:
            self.closed = True

    def _parse(
        path: Path,
        *,
        settings: DocMindSettings,
        document_id: str | None,
        _include_searchable_pdf: bool,
    ) -> DocumentParseResult:
        del path, settings, document_id
        captured["include_searchable_pdf"] = _include_searchable_pdf
        return DocumentParseResult(
            document_id="doc-1",
            source_filename="source.pdf",
            source_hash="abc",
            profile=ParserProfile.CPU_SAFE,
            parser_framework=ParserFramework.DOCLING,
            page_count=1,
            pages=[
                PageParseResult(
                    page_id="doc-1::page::1",
                    page_index=0,
                    text_markdown="text",
                )
            ],
        )

    sender = _Sender()
    monkeypatch.setattr(service, "_own_parse_worker_process_group", lambda: None)
    monkeypatch.setattr(service, "parse_document_sync", _parse)

    service._parse_worker(
        sender,  # type: ignore[arg-type]
        str(tmp_path / "source.pdf"),
        cfg.model_dump(mode="python"),
        None,
    )

    assert captured["include_searchable_pdf"] is False
    assert sender.closed is True


def test_ocrmypdf_deadline_is_strictly_inside_the_outer_parser_deadline() -> None:
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    timeout = service._bounded_ocrmypdf_timeout(
        settings=cfg,
        remaining_seconds=300.0,
    )

    assert timeout == 270.0
    assert timeout < cfg.parsing.parse_timeout_seconds


@pytest.mark.asyncio
async def test_async_text_parse_round_trips_through_worker(tmp_path: Path) -> None:
    source = tmp_path / "worker.md"
    source.write_text("worker boundary", encoding="utf-8")
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    result = await service.parse_document(source, settings=cfg)

    assert result.pages[0].text_markdown == "worker boundary"


@pytest.mark.asyncio
async def test_async_markdown_skips_searchable_pdf_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.md"
    source.write_text("markdown", encoding="utf-8")
    cfg = DocMindSettings.model_validate({"ocr": {"searchable_pdf_enabled": True}})

    async def _unexpected_export(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("Markdown must not be sent to OCRmyPDF")

    monkeypatch.setattr(service, "export_searchable_pdf_async", _unexpected_export)

    result = await service.parse_document(source, settings=cfg)

    assert result.pages[0].text_markdown == "markdown"
    assert result.artifacts == []


@pytest.mark.asyncio
async def test_async_searchable_pdf_helper_noops_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    result = DocumentParseResult(
        document_id="doc-1",
        source_filename=source.name,
        source_hash="abc",
        profile=ParserProfile.CPU_SAFE,
        parser_framework=ParserFramework.DOCLING,
    )

    async def _unexpected_export(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("disabled searchable-PDF export must be a no-op")

    monkeypatch.setattr(service, "export_searchable_pdf_async", _unexpected_export)

    await service._attach_searchable_pdf_artifact_async(
        result,
        source,
        settings=cfg,
        timeout_seconds=1.0,
    )

    assert result.artifacts == []
    assert result.routing_decisions == []


def test_plain_text_uses_direct_text_framework() -> None:
    assert choose_framework(Path("a.md")) is ParserFramework.DIRECT_TEXT


def test_direct_text_provenance_omits_unused_pdf_and_ocr_backends() -> None:
    result = DocumentParseResult(
        document_id="doc-1",
        source_filename="file.md",
        source_hash="abc",
        profile=ParserProfile.CPU_SAFE,
        parser_framework=ParserFramework.DIRECT_TEXT,
        page_count=1,
    )

    provenance = result.provenance()

    assert "pdf_backend" not in provenance
    assert "ocr_engine" not in provenance


def test_docx_docling_provenance_omits_unused_pdf_and_ocr_backends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.docx"
    source.write_bytes(b"docx fixture")
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    def _convert(*_args: object, **kwargs: object) -> DocumentParseResult:
        document_id = str(kwargs["document_id"])
        return DocumentParseResult(
            document_id=document_id,
            source_filename=source.name,
            source_hash=str(kwargs["source_hash"]),
            profile=ParserProfile.CPU_SAFE,
            parser_framework=ParserFramework.DOCLING,
            page_count=1,
            pages=[
                PageParseResult(
                    page_id=f"{document_id}::page::1",
                    page_index=0,
                    text_markdown="docx text",
                )
            ],
        )

    monkeypatch.setattr(service, "convert_with_docling", _convert)

    provenance = service.parse_document_sync(source, settings=cfg).provenance()

    assert provenance["framework"] == "docling"
    assert "pdf_backend" not in provenance
    assert "ocr_engine" not in provenance


def test_provenance_is_manifest_safe() -> None:
    result = DocumentParseResult(
        document_id="doc-1",
        source_filename="file.pdf",
        source_hash="abc",
        profile=ParserProfile.CPU_SAFE,
        parser_framework=ParserFramework.DOCLING,
        page_count=1,
        pages=[
            PageParseResult(
                page_id="doc-1::page::1",
                page_index=0,
                text_markdown="x",
                ocr_applied=True,
                routing_reason="pdf_low_text_ocr_candidate",
            )
        ],
    )

    provenance = result.provenance()

    assert provenance["profile"] == "cpu_safe"
    assert provenance["pdf_backend"] == "pypdfium2"
    assert provenance["ocr_engine"] == "rapidocr"
    assert provenance["ocr_applied_pages"] == [0]
    assert "file.pdf" not in str(provenance)


def test_pdf_model_preflight_verifies_both_manifests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_cache = tmp_path / "models"
    cfg = DocMindSettings.model_validate({"ocr": {"model_cache_dir": model_cache}})
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        service,
        "verify_docling_layout_models",
        lambda cache: calls.append(("docling", cache)),
    )
    monkeypatch.setattr(
        service,
        "verify_rapidocr_models",
        lambda cache: calls.append(("rapidocr", cache)),
    )

    service._require_pdf_models(tmp_path / "source.pdf", settings=cfg)

    assert calls == [("docling", model_cache), ("rapidocr", model_cache)]


@pytest.mark.parametrize(
    "failing_verifier",
    ["verify_docling_layout_models", "verify_rapidocr_models"],
)
def test_pdf_model_integrity_failure_uses_safe_readiness_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing_verifier: str,
) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    _trust_pdf_models(monkeypatch)

    def _reject(_model_cache_dir: Path) -> None:
        raise ModelIntegrityError("private model path and digest mismatch")

    monkeypatch.setattr(
        service,
        failing_verifier,
        _reject,
    )

    def _convert_unreachable(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Docling conversion should not run")

    monkeypatch.setattr(service, "convert_with_docling", _convert_unreachable)

    with pytest.raises(DocumentParseError) as raised:
        service.parse_document_sync(pdf, settings=cfg)

    assert raised.value.stage == "parser_readiness"
    assert raised.value.reason == "parser_models_not_ready"
    assert raised.value.cause_type == "ModelIntegrityError"
    assert "private model path" not in str(raised.value)


def test_docling_pdf_failure_raises_typed_parse_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Docling failure must not decode the original PDF as plaintext."""
    pdf = tmp_path / "malformed.pdf"
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n")
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]
    inspection = SimpleNamespace(
        has_native_text=True,
        page_count=1,
        pages=[SimpleNamespace(page_index=0, text="native text " * 12)],
    )
    monkeypatch.setattr(
        service,
        "_inspect_pdf_once",
        lambda *_args, **_kwargs: inspection,
    )
    _trust_pdf_models(monkeypatch)

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic Docling failure")

    monkeypatch.setattr(service, "convert_with_docling", _explode)

    with pytest.raises(DocumentParseError) as raised:
        service.parse_document_sync(pdf, settings=cfg)

    assert raised.value.stage == "docling_conversion"
    assert raised.value.reason == "conversion_failed"
    assert raised.value.source_suffix == ".pdf"


def test_docling_partial_success_is_not_publishable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Docling partial output is a parse failure, not a valid document."""
    from docling.datamodel.base_models import ConversionStatus

    pdf = tmp_path / "partial.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    class _PartialConverter:
        def convert(
            self,
            _path: Path,
            *,
            max_file_size: int,
            max_num_pages: int,
        ) -> SimpleNamespace:
            assert max_file_size == 1024
            assert max_num_pages == 10
            document = SimpleNamespace(
                export_to_markdown=lambda: "only the pages that survived"
            )
            return SimpleNamespace(
                status=ConversionStatus.PARTIAL_SUCCESS,
                errors=[RuntimeError("backend detail must not escape")],
                document=document,
            )

    monkeypatch.setattr(
        docling_backend,
        "_docling_converter",
        lambda **_kwargs: _PartialConverter(),
    )

    with pytest.raises(DocumentParseError) as raised:
        docling_backend.convert_with_docling(
            pdf,
            document_id="doc-partial",
            source_hash="abc",
            model_cache_dir=tmp_path / "models",
            max_pages=10,
            max_file_size=1024,
            versions=ParserVersions(),
        )

    assert raised.value.reason == "conversion_status_partial_success"
    assert "backend detail" not in str(raised.value)


def test_mixed_pdf_routes_ocr_only_for_low_text_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf = tmp_path / "mixed.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    monkeypatch.setattr(
        service,
        "_inspect_pdf_once",
        lambda *_args, **_kwargs: SimpleNamespace(
            has_native_text=True,
            page_count=2,
            pages=[
                SimpleNamespace(
                    page_index=0,
                    text="native text " * 12,
                    has_text=True,
                ),
                SimpleNamespace(page_index=1, text="", has_text=False),
            ],
        ),
    )
    _trust_pdf_models(monkeypatch)
    monkeypatch.setattr(
        service,
        "_ocr_pdf_pages",
        lambda *_args, **_kwargs: {1: "ocr text"},
    )

    def _convert(*_args: object, **kwargs: object) -> DocumentParseResult:
        return DocumentParseResult(
            document_id=str(kwargs["document_id"]),
            source_filename=pdf.name,
            source_hash=str(kwargs["source_hash"]),
            profile=ParserProfile.CPU_SAFE,
            parser_framework=ParserFramework.DOCLING,
            page_count=2,
            pages=[
                PageParseResult(
                    page_id=f"{kwargs['document_id']}::page::1",
                    page_index=0,
                    text_markdown="structured native page",
                ),
                PageParseResult(
                    page_id=f"{kwargs['document_id']}::page::2",
                    page_index=1,
                    text_markdown="",
                ),
            ],
        )

    monkeypatch.setattr(service, "convert_with_docling", _convert)

    result = service.parse_document_sync(pdf, settings=cfg)

    assert [page.ocr_applied for page in result.pages] == [False, True]
    assert [page.text_markdown for page in result.pages] == [
        "structured native page",
        "ocr text",
    ]
    assert result.provenance()["ocr_applied_pages"] == [1]


def test_local_rapidocr_cache_allows_page_ocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf = tmp_path / "warm.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    model_cache = tmp_path / "models"
    for model_path in rapidocr_model_files(model_cache).values():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"placeholder")
    cfg = DocMindSettings.model_validate({"ocr": {"model_cache_dir": model_cache}})

    monkeypatch.setattr(
        service,
        "_inspect_pdf_once",
        lambda *_args, **_kwargs: SimpleNamespace(
            has_native_text=False,
            page_count=1,
            pages=[SimpleNamespace(page_index=0, text="", has_text=False)],
        ),
    )
    _trust_pdf_models(monkeypatch)

    def _blocked_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("local parser attempted network access")

    monkeypatch.setattr("socket.create_connection", _blocked_network)
    monkeypatch.setattr(
        service,
        "_ocr_pdf_pages",
        lambda *_args, **_kwargs: {0: "ocr text"},
    )

    def _convert(*_args: object, **kwargs: object) -> DocumentParseResult:
        return DocumentParseResult(
            document_id=str(kwargs["document_id"]),
            source_filename=pdf.name,
            source_hash=str(kwargs["source_hash"]),
            profile=ParserProfile.CPU_SAFE,
            parser_framework=ParserFramework.DOCLING,
            page_count=1,
            pages=[
                PageParseResult(
                    page_id=f"{kwargs['document_id']}::page::1",
                    page_index=0,
                    text_markdown="",
                )
            ],
        )

    monkeypatch.setattr(service, "convert_with_docling", _convert)

    result = service.parse_document_sync(pdf, settings=cfg)

    assert result.pages[0].ocr_applied is True
    assert result.pages[0].text_markdown == "ocr text"


def test_searchable_pdf_artifact_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    cfg = DocMindSettings.model_validate(
        {
            "data_dir": tmp_path / "data",
            "ocr": {"searchable_pdf_enabled": True},
        }
    )

    monkeypatch.setattr(
        service,
        "_inspect_pdf_once",
        lambda *_args, **_kwargs: SimpleNamespace(
            has_native_text=True,
            page_count=1,
            pages=[
                SimpleNamespace(
                    page_index=0,
                    text="native text " * 12,
                    has_text=True,
                )
            ],
        ),
    )
    _trust_pdf_models(monkeypatch)

    def _convert(*_args: object, **kwargs: object) -> DocumentParseResult:
        return DocumentParseResult(
            document_id=str(kwargs["document_id"]),
            source_filename=pdf.name,
            source_hash=str(kwargs["source_hash"]),
            profile=ParserProfile.CPU_SAFE,
            parser_framework=ParserFramework.DOCLING,
            page_count=1,
            pages=[
                PageParseResult(
                    page_id=f"{kwargs['document_id']}::page::1",
                    page_index=0,
                    text_markdown="native text",
                )
            ],
        )

    def _export(_input: Path, output: Path, **_kwargs: object) -> Path:
        output.write_bytes(b"%PDF-1.4 searchable\n%%EOF\n")
        return output

    monkeypatch.setattr(service, "convert_with_docling", _convert)
    monkeypatch.setattr(service, "export_searchable_pdf", _export)

    result = service.parse_document_sync(pdf, settings=cfg)
    artifacts = result.provenance()["searchable_pdf_artifacts"]

    assert artifacts
    assert artifacts[0]["kind"] == "searchable_pdf"
    assert artifacts[0]["suffix"] == ".pdf"
