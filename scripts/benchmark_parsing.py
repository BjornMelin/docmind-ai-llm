"""Local benchmark harness for DocMind's CPU-safe parser contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import DocMindSettings
from src.processing.parsing.service import parse_document_sync

BENCHMARK_SCHEMA_VERSION = 3

_FIXTURE_EXPECTATIONS: dict[str, tuple[int, tuple[str, ...]]] = {
    "adversarial_text.pdf": (1, ("Adversarial punctuation", "alpha")),
    "born_digital.pdf": (1, ("DocMind born digital", "selectable text")),
    "formula_heavy.pdf": (1, ("Formula fixture", "E = m c")),
    "multilingual.pdf": (1, ("English", "Espanol", "Francais", "Deutsch")),
    "multipage_native.pdf": (3, ("Physical page one", "Physical page three")),
    "photo_skewed.pdf": (1, ("Skewed photo", "Low native text")),
    "scanned.pdf": (1, ("Scanned fixture", "OCR should be required")),
    "table_heavy.pdf": (1, ("Quarter", "Q1")),
}


def main() -> None:
    """Run the local parser benchmark harness."""
    args = _parse_args()
    if args.worker_fixture is not None:
        _run_worker(args)
        return
    repository = _repository_identity()
    _require_clean_repository(repository)
    fixtures = _resolve_fixtures(args.fixtures, args.generate_minimal_fixtures)
    settings = DocMindSettings.model_validate(
        {
            "parsing": {"profile": "cpu_safe"},
            "ocr": {
                "searchable_pdf_enabled": args.searchable_pdf,
            },
        }
    )
    results = [
        _benchmark_one(path, settings=settings, repeat=args.repeat) for path in fixtures
    ]
    _validate_benchmark_results(results, expected_repeats=args.repeat)
    payload = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "measurement_scope": "isolated_parser_worker_execution",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "profile": "cpu_safe",
        "repeat_count": args.repeat,
        "runtime": _runtime_identity(),
        "repository": repository,
        "network_egress": "NOT_MEASURED",
        "searchable_pdf_requested": bool(args.searchable_pdf),
        "fixtures": {path.name: _sha256_file(path) for path in fixtures},
        "fixture_count": len(results),
        "results": results,
        "summary": _summary(results),
    }
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{output}\n", encoding="utf-8")
    print(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--fixtures", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--searchable-pdf",
        action="store_true",
        help="Request optional OCRmyPDF searchable-PDF artifact generation.",
    )
    parser.add_argument(
        "--generate-minimal-fixtures",
        action="store_true",
        help="Create tiny license-safe benchmark PDFs under cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON benchmark output.",
    )
    parser.add_argument("--worker-fixture", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve_fixtures(paths: list[Path], generate: bool) -> list[Path]:
    fixtures = [path for path in paths if path.exists() and path.is_file()]
    if generate or not fixtures:
        fixtures.extend(_generate_minimal_fixtures())
    return fixtures


def _generate_minimal_fixtures() -> list[Path]:
    base = Path("cache") / "benchmarks" / "parsing"
    base.mkdir(parents=True, exist_ok=True)
    fixture_specs = {
        "born_digital.pdf": [
            "DocMind born digital fixture.",
            "This page has enough selectable text to avoid OCR routing.",
            "The parser should prefer native text extraction for this document.",
        ],
        "table_heavy.pdf": [
            "Quarter Revenue Cost Margin",
            "Q1 100 71 29",
            "Q2 124 78 46",
            "Q3 132 88 44",
        ],
        "formula_heavy.pdf": [
            "Formula fixture",
            "E = m c^2",
            "a^2 + b^2 = c^2",
            "integral from 0 to 1 of x dx = 1/2",
        ],
    }
    fixtures: list[Path] = []
    for name, lines in fixture_specs.items():
        path = base / name
        _write_text_pdf(path, lines)
        fixtures.append(path)
    multipage = base / "multipage_native.pdf"
    _write_text_pdf_pages(
        multipage,
        [
            ["Physical page one", "Native structured text must remain on page one."],
            ["Physical page two", "Native structured text must remain on page two."],
            ["Physical page three", "Page ordering must remain deterministic."],
        ],
    )
    fixtures.append(multipage)
    adversarial = base / "adversarial_text.pdf"
    _write_text_pdf(
        adversarial,
        [
            "Adversarial punctuation: <>&\\{}[]()",
            "Unicode-safe content: cafe resume naive",
            "Repeated tokens: alpha alpha alpha alpha alpha",
        ],
    )
    fixtures.append(adversarial)
    image_specs = {
        "scanned.pdf": ["Scanned fixture", "OCR should be required."],
        "photo_skewed.pdf": ["Skewed photo fixture", "Low native text."],
        "multilingual.pdf": ["English", "Espanol", "Francais", "Deutsch"],
    }
    for name, lines in image_specs.items():
        path = base / name
        _write_image_pdf(path, lines, skew=name == "photo_skewed.pdf")
        fixtures.append(path)
    return fixtures


def _write_text_pdf(path: Path, lines: list[str]) -> None:
    _write_text_pdf_pages(path, [lines])


def _write_text_pdf_pages(path: Path, pages: list[list[str]]) -> None:
    """Write a deterministic born-digital PDF with one content stream per page."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    font_ref = writer._add_object(font)
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        commands = ["BT", "/F1 12 Tf", "72 720 Td"]
        for index, line in enumerate(lines):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if index:
                commands.append("0 -18 Td")
            commands.append(f"({escaped}) Tj")
        commands.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)


def _write_image_pdf(path: Path, lines: list[str], *, skew: bool = False) -> None:
    image = Image.new("RGB", (900, 1200), "white")
    draw = ImageDraw.Draw(image)
    y = 160
    for line in lines:
        draw.text((120, y), line, fill="black")
        y += 70
    if skew:
        image = image.rotate(5, expand=True, fillcolor="white")
    image.save(path, "PDF", resolution=150.0)


def _benchmark_one(
    path: Path,
    *,
    settings: DocMindSettings,
    repeat: int,
) -> dict[str, Any]:
    if repeat > 1 or settings.ocr.searchable_pdf_enabled:
        return _benchmark_one_isolated(path, settings=settings, repeat=repeat)
    return _benchmark_one_in_process(path, settings=settings, repeat=repeat)


def _benchmark_one_in_process(
    path: Path,
    *,
    settings: DocMindSettings,
    repeat: int,
) -> dict[str, Any]:
    durations: list[float] = []
    last_page_count = 0
    last_framework = ""
    last_output_hash = ""
    output_hashes: list[str] = []
    last_ocr_pages = 0
    last_searchable_artifacts = 0
    last_content_validation: dict[str, Any] = {"passed": False}
    last_text_char_count = 0
    last_error: dict[str, str] | None = None
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        try:
            result = parse_document_sync(path, settings=settings)
        except Exception as exc:  # benchmark records blocked capabilities as data
            durations.append((time.perf_counter() - start) * 1000.0)
            last_error = {"type": type(exc).__name__, "message": str(exc)}
            continue
        durations.append((time.perf_counter() - start) * 1000.0)
        last_page_count = result.page_count
        last_framework = result.parser_framework.value
        provenance = result.provenance()
        last_output_hash = _output_hash(result)
        output_hashes.append(last_output_hash)
        last_content_validation = _content_validation(path.name, result)
        last_text_char_count = sum(len(page.text_markdown) for page in result.pages)
        last_ocr_pages = len(provenance.get("ocr_applied_pages") or [])
        last_searchable_artifacts = len(
            provenance.get("searchable_pdf_artifacts") or []
        )
        last_error = None
    page_denominator = max(1, last_page_count)
    return {
        "source_filename": path.name,
        "framework": last_framework,
        "page_count": last_page_count,
        "latency_ms_median": statistics.median(durations),
        "latency_ms_min": min(durations),
        "latency_ms_max": max(durations),
        "latency_ms_per_page": statistics.median(durations) / page_denominator,
        "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "ocr_fallback_rate": last_ocr_pages / page_denominator,
        "deterministic_output_hash": last_output_hash,
        "repetition_output_hashes": output_hashes,
        "deterministic": len(set(output_hashes)) == 1 and len(output_hashes) == repeat,
        "content_validation": last_content_validation,
        "text_char_count": last_text_char_count,
        "searchable_pdf_artifact_count": last_searchable_artifacts,
        "network_egress": "NOT_MEASURED",
        "error": last_error,
    }


def _benchmark_one_isolated(
    path: Path,
    *,
    settings: DocMindSettings,
    repeat: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    worker_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-fixture",
        str(path),
    ]
    if settings.ocr.searchable_pdf_enabled:
        worker_args.append("--searchable-pdf")
    for _ in range(max(1, repeat)):
        completed = subprocess.run(
            worker_args,
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHONFAULTHANDLER": "1"},
            text=True,
            timeout=float(settings.parsing.parse_timeout_seconds) + 30.0,
        )
        marker = "DOCMIND_BENCHMARK_RESULT_JSON="
        payload_line = next(
            (
                line[len(marker) :]
                for line in completed.stdout.splitlines()
                if line.startswith(marker)
            ),
            None,
        )
        if completed.returncode != 0 or payload_line is None:
            results.append(
                {
                    "source_filename": path.name,
                    "framework": "",
                    "page_count": 0,
                    "latency_ms_median": 0.0,
                    "latency_ms_min": 0.0,
                    "latency_ms_max": 0.0,
                    "latency_ms_per_page": 0.0,
                    "rss_mb": 0.0,
                    "ocr_fallback_rate": 0.0,
                    "deterministic_output_hash": "",
                    "repetition_output_hashes": [],
                    "deterministic": False,
                    "content_validation": {"passed": False},
                    "text_char_count": 0,
                    "searchable_pdf_artifact_count": 0,
                    "network_egress": "NOT_MEASURED",
                    "error": {
                        "type": "BenchmarkWorkerError",
                        "returncode": completed.returncode,
                        "message": (
                            f"returncode={completed.returncode}; "
                            f"stderr={completed.stderr.strip()[:500]}"
                        ),
                    },
                }
            )
            continue
        try:
            payload = json.loads(payload_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Benchmark worker emitted invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Benchmark worker emitted the wrong payload shape")
        results.append(payload)
    return _combine_isolated_results(path.name, results)


def _combine_isolated_results(
    source_filename: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [item for item in results if not item.get("error")]
    if len(successful) != len(results):
        failed_index, failed = next(
            (index, item) for index, item in enumerate(results) if item.get("error")
        )
        combined = dict(failed)
        error = failed.get("error")
        combined["error"] = {
            "failed_repetition": failed_index + 1,
            "returncode": (
                error.get("returncode") if isinstance(error, dict) else None
            ),
            "type": (
                str(error.get("type"))
                if isinstance(error, dict)
                else "BenchmarkWorkerError"
            ),
        }
        combined["repetition_output_hashes"] = [
            str(item["deterministic_output_hash"])
            for item in successful
            if item.get("deterministic_output_hash")
        ]
        combined["deterministic"] = False
        return combined
    durations = [float(item["latency_ms_median"]) for item in successful]
    last = successful[-1]
    combined = dict(last)
    combined["latency_ms_median"] = statistics.median(durations)
    combined["latency_ms_min"] = min(durations)
    combined["latency_ms_max"] = max(durations)
    combined["latency_ms_per_page"] = statistics.median(durations) / max(
        1,
        int(last["page_count"]),
    )
    combined["rss_mb"] = max(float(item["rss_mb"]) for item in successful)
    hashes = [str(item["deterministic_output_hash"]) for item in successful]
    combined["repetition_output_hashes"] = hashes
    combined["deterministic"] = len(set(hashes)) == 1
    combined["error"] = None
    return combined


def _run_worker(args: argparse.Namespace) -> None:
    settings = DocMindSettings.model_validate(
        {
            "parsing": {"profile": "cpu_safe"},
            "ocr": {
                "searchable_pdf_enabled": args.searchable_pdf,
            },
        }
    )
    result = _benchmark_one_in_process(
        args.worker_fixture,
        settings=settings,
        repeat=1,
    )
    print(f"DOCMIND_BENCHMARK_RESULT_JSON={json.dumps(result, sort_keys=True)}")


def _output_hash(result: Any) -> str:
    payload = {
        "pages": [
            {
                "index": page.page_index,
                "text": page.text_markdown,
                "ocr_applied": page.ocr_applied,
                "routing_reason": page.routing_reason,
            }
            for page in result.pages
        ],
        "provenance": result.provenance(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_validation(source_filename: str, result: Any) -> dict[str, Any]:
    """Validate fixture fidelity without persisting document content."""
    text = "\n".join(page.text_markdown for page in result.pages)
    minimum_pages, required_tokens = _FIXTURE_EXPECTATIONS.get(
        source_filename,
        (1, ()),
    )
    normalized = text.casefold()
    matched = sum(token.casefold() in normalized for token in required_tokens)
    passed = (
        result.page_count >= minimum_pages
        and bool(text.strip())
        and matched == len(required_tokens)
    )
    return {
        "minimum_pages": minimum_pages,
        "required_token_count": len(required_tokens),
        "matched_token_count": matched,
        "passed": passed,
    }


def _validate_benchmark_results(
    results: list[dict[str, Any]],
    *,
    expected_repeats: int,
) -> None:
    if expected_repeats < 1:
        raise ValueError("repeat must be at least 1")
    if not results:
        raise RuntimeError("Benchmark produced no fixture results")
    for item in results:
        name = str(item.get("source_filename") or "<unknown>")
        error = item.get("error")
        if error:
            error_type = (
                str(error.get("type")) if isinstance(error, dict) else "unknown_error"
            )
            raise RuntimeError(f"Benchmark fixture failed: {name} ({error_type})")
        if item.get("framework") != "docling" or int(item.get("page_count", 0)) < 1:
            raise RuntimeError(
                f"Benchmark fixture emitted invalid parser output: {name}"
            )
        content_validation = item.get("content_validation")
        if not isinstance(content_validation, dict) or not content_validation.get(
            "passed"
        ):
            raise RuntimeError(f"Benchmark fixture content validation failed: {name}")
        if int(item.get("text_char_count", 0)) < 1:
            raise RuntimeError(f"Benchmark fixture emitted empty text: {name}")
        hashes = item.get("repetition_output_hashes")
        if not isinstance(hashes, list) or len(hashes) != expected_repeats:
            raise RuntimeError(f"Benchmark repetition output missing: {name}")
        if not all(isinstance(value, str) and value for value in hashes):
            raise RuntimeError(f"Benchmark output hash missing: {name}")
        if len(set(hashes)) != 1 or item.get("deterministic") is not True:
            raise RuntimeError(f"Benchmark output is nondeterministic: {name}")


def _runtime_identity() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in (
        "docmind_ai_llm",
        "docling",
        "pypdfium2",
        "rapidocr",
        "onnxruntime",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "packages": packages,
    }


def _repository_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_entry_count": len(status.splitlines()),
        "dirty_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _require_clean_repository(repository: dict[str, Any]) -> None:
    """Reject benchmark evidence that cannot map to one exact source commit."""
    if repository.get("dirty") is True:
        raise RuntimeError(
            "Benchmark evidence requires a clean Git worktree; commit source changes "
            "before generating the release artifact."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["latency_ms_median"]) for item in results]
    if not latencies:
        return {}
    return {
        "content_validated_count": sum(
            1
            for item in results
            if isinstance(item.get("content_validation"), dict)
            and item["content_validation"].get("passed") is True
        ),
        "deterministic_count": sum(
            1 for item in results if item.get("deterministic") is True
        ),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_max": max(latencies),
        "rss_mb_max": max(float(item["rss_mb"]) for item in results),
        "error_count": sum(1 for item in results if item.get("error")),
    }


if __name__ == "__main__":
    main()
