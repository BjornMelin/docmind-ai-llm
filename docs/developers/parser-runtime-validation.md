# Parser runtime validation

This record covers DocMind's supported local parsing path. The canonical configuration lives in `src/config/settings.py`, and the reproducible harness lives in `scripts/benchmark_parsing.py`.

The CPU-safe RapidOCR adapter limits ONNX Runtime to four intra-op threads and one inter-op thread. This bounds native thread-pool parallelism while preserving useful operator concurrency.

## Supported runtime

The parser contract is deliberately narrow:

- Docling for document conversion
- pypdfium2 for PDF inspection and rasterization
- RapidOCR's packaged PP-OCRv6 detector/recognizer and PP-OCRv4 classifier
  defaults for CPU OCR
- OCRmyPDF and Tesseract for optional searchable-PDF artifacts

PaddleOCR, VLM OCR, Tesseract-as-parser, GPU OCR profiles, backend selectors, and automatic remote fallbacks are not supported. A parser failure is surfaced as a typed `DocumentParseError`; DocMind does not decode failed binary inputs as text.

OCRmyPDF export is fail-open and separate from parsing. It runs only on POSIX systems, including Linux, macOS, and WSL2 on Windows. Timeout, cancellation, and failure paths terminate and reap the entire subprocess group so OCR descendants cannot outlive the request.

## Model readiness

Download the parser models into the application cache:

```bash
uv run python tools/models/pull.py \
  --parser-defaults \
  --parser-cache-dir cache/models
```

Then verify the local parser model supply:

```bash
uv run python scripts/parser_health.py --check
```

The health command checks parser dependencies and hashes every Docling layout file against the source-controlled canonical manifest. Any mismatch is reported by relative path in `docling.model_issues`. RapidOCR validates the packaged models against its own upstream checksums during engine initialization; offline initialization and fixture inference are separate test and image gates.

## Benchmark evidence

Regenerate the benchmark artifact:

```bash
uv run python scripts/benchmark_parsing.py \
  --generate-minimal-fixtures \
  --repeat 3 \
  --output docs/benchmarks/parser-runtime-validation.json
```

Schema version 3 refuses a dirty Git worktree and binds results to the exact
source commit, runtime package versions, fixture SHA-256 values, repeat count,
per-run output hashes, and content assertions. It covers native text, tables,
formulas, multiple physical pages, adversarial text, scanned text, skewed text,
and multilingual text with fixture-specific required tokens.

Latency covers isolated parser-worker execution. It does not include application startup, upload persistence, indexing, retrieval, or UI rendering.

`network_egress` is recorded as `NOT_MEASURED`. The harness does not instrument the host network, so the artifact is not evidence of network isolation. Parser model preflight and application endpoint policy are separate controls.

### v2 release-candidate baseline

The checked-in schema 3 artifact records the v2 release-candidate baseline. It
was generated from clean commit
`77c8a62370712cca172392d64c055e30535266c0` on Linux under WSL2 with
CPython 3.12.13. It records Docling 2.112.0, pypdfium2 5.11.0, RapidOCR
3.9.1, and ONNX Runtime 1.27.0.

- 8 of 8 fixtures passed their content assertions.
- All 8 fixtures produced identical output hashes across three isolated runs.
- No parser errors occurred.
- `summary.latency_ms_median` is 4976.170 ms and
  `summary.latency_ms_max` is 5719.963 ms.
- `summary.rss_mb_max` is 1304.359 MiB.

These values are a workstation-specific regression baseline, not a
cross-platform performance promise. The fixture hashes, individual results,
runtime identity, and unrounded values live in
`docs/benchmarks/parser-runtime-validation.json`.
