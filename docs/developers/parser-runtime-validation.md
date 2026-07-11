# Parser runtime validation

This record covers DocMind's supported local parsing path. The canonical configuration lives in `src/config/settings.py`, and the reproducible harness lives in `scripts/benchmark_parsing.py`.

The CPU-safe RapidOCR adapter limits ONNX Runtime to four intra-op threads and one inter-op thread. This bounds native thread-pool parallelism while preserving useful operator concurrency.

## Supported runtime

The parser contract is deliberately narrow:

- Docling for document conversion
- pypdfium2 for PDF inspection and rasterization
- RapidOCR with ONNX Runtime for CPU OCR
- OCRmyPDF and Tesseract for optional searchable-PDF artifacts

PaddleOCR, VLM OCR, Tesseract-as-parser, GPU OCR profiles, backend selectors, and automatic remote fallbacks are not supported. A parser failure is surfaced as a typed `DocumentParseError`; DocMind does not decode failed binary inputs as text.

OCRmyPDF export is fail-open and separate from parsing. It runs only on POSIX systems, including Linux, macOS, and WSL2 on Windows. Timeout, cancellation, and failure paths terminate and reap the entire subprocess group so OCR descendants cannot outlive the request.

## Model readiness

Download the parser models into the application cache:

```bash
uv run python tools/models/pull.py \
  --parser-defaults \
  --rapidocr-cache-dir cache/models
```

Then verify the local parser model supply:

```bash
uv run python scripts/parser_health.py --check
```

The health command checks parser dependencies and hashes every Docling and RapidOCR model file against the source-controlled canonical manifest. Any mismatch is reported by relative path in `docling.model_issues` or `rapidocr.model_issues`. It does not run a fixture parse.

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

### Current release baseline

The checked-in schema 3 artifact was generated from clean commit
`8ce264845eb43bcf02526c37625dcd0358a3f1ba` on Linux under WSL2 with
CPython 3.12.13. It records Docling 2.92.0, pypdfium2 5.7.1, RapidOCR
3.8.1, and ONNX Runtime 1.23.2.

- 8 of 8 fixtures passed their content assertions.
- All 8 fixtures produced identical output hashes across three isolated runs.
- No parser errors occurred.
- `summary.latency_ms_median` is 4217.030 ms and
  `summary.latency_ms_max` is 4776.168 ms.
- `summary.rss_mb_max` is 1245.434 MiB.

These values are a workstation-specific regression baseline, not a
cross-platform performance promise. The fixture hashes, individual results,
runtime identity, and unrounded values live in
`docs/benchmarks/parser-runtime-validation.json`.
