---
spec: SPEC-026
title: Canonical ingestion API
version: 2.5.0
date: 2026-07-11
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-024: Provide one canonical programmatic ingestion API.
  - NFR-MAINT-003: Keep file loading and metadata sanitation under one owner.
  - NFR-SEC-001: Preserve local-first and fail-closed parser behavior.
related_adrs: ["ADR-045"]
---

## Objective

`src/processing/ingestion_api.py` owns local path collection, stable identifier generation, canonical document loading, and metadata sanitation. `src/utils/hashing.py` owns the shared `doc-<full lowercase SHA-256>` formatter used by ingestion callers. Package `__init__.py` modules do not re-export this domain API.

## Public contract

The module exports these functions:

```python
def collect_paths(
    root: Path | str,
    *,
    recursive: bool = True,
    extensions: set[str] | None = None,
) -> list[Path]: ...

async def load_documents(
    paths: Sequence[Path | str],
    *,
    doc_id: str | None = None,
    parsing_overrides: dict[str, Any] | None = None,
) -> list[Document]: ...

async def load_documents_from_inputs(
    inputs: Sequence[IngestionInput],
) -> list[Document]: ...
```

The remaining exports are:

- `generate_stable_id(file_path)`, which returns `doc-<full lowercase SHA-256>`
- `sanitize_document_metadata(meta, *, source_filename)`
- `clear_ingestion_cache()`

## Input behavior

`collect_paths`:

- Returns an empty list when the root does not exist
- Requires an existing directory when the root exists
- Rejects a symlink root
- Rejects symlinks between the root and each file
- Rejects resolved paths outside the root
- Filters by normalized extension and returns a deterministic sort order

`load_documents`:

- Accepts local file paths
- Skips missing paths and non-files
- Uses the parser service for every supported format
- Propagates every parser-service failure as `DocumentParseError`
- Has no direct-text fallback; the parser service alone owns UTF-8 text loading
- Applies `doc_id` only when loading one path
- Applies only the supported `force_ocr` and `export_searchable_pdf` overrides
- Emits one LlamaIndex `Document` per physical parser page

`load_documents_from_inputs`:

- Accepts normalized `IngestionInput` values
- Requires exactly one `source_path` or strict `payload_text` value per input
- Routes path inputs through `load_documents`
- Converts explicit text payloads without a binary coercion path
- Preserves the caller’s stable document identifier
- Rejects parser-owned metadata keys through the input model

## Metadata contract

The ingestion API removes path-like keys before persistence. It stores a safe basename as `source_filename` and excludes parser provenance from embedding and language-model metadata strings.

Parser provenance can include:

- Framework and profile
- Observed PDF and OCR backends (omitted for direct text, where neither runs)
- Package versions
- Page routing and OCR decisions
- Searchable-PDF artifact references
- Configuration hash
- Component health

## Breaking identity migration

`doc-<full lowercase SHA-256>` deliberately replaces the former truncated
document IDs without a compatibility layer. Existing ingestion caches,
docstores, snapshots, and Qdrant collections must be retired or rebuilt before
the upgraded corpus is served. Follow the canonical
[migration runbook](spec-002-ingestion-pipeline.md#breaking-full-sha-identity-migration).

## Package boundary

Callers import the owner directly:

```python
from src.processing.ingestion_api import load_documents
from src.processing.ingestion_api import load_documents_from_inputs
```

Import-light package initializers prevent parser-only processes from loading ingestion, Qdrant, or Torch stacks before they need them.

## Security requirements

- Reject symlink traversal
- Never persist an absolute source path
- Never decode a failed binary source as text
- Never publish a partial binary parse
- Never accept arbitrary in-memory bytes
- Never copy parser-owned provenance from user metadata

## Verification

The canonical tests are:

- `tests/unit/processing/test_ingestion_api.py`
- `tests/unit/processing/test_parser_contract.py`
- `tests/unit/models/test_ingestion_models.py`
- `tests/integration/test_ingestion_pipeline_pdf_images.py`

Run:

```bash
uv run pytest \
  tests/unit/processing/test_ingestion_api.py \
  tests/unit/processing/test_parser_contract.py \
  tests/unit/models/test_ingestion_models.py
```
