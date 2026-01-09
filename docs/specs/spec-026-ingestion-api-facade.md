---
spec: SPEC-026
title: Programmatic Ingestion API + Legacy Facade Cleanup (`src.utils.document`)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - FR-024: Provide a canonical programmatic ingestion API (local-only).
  - NFR-MAINT-003: No placeholder APIs; docs must match code.
  - NFR-SEC-001: Offline-first defaults MUST remain intact.
related_adrs: ["ADR-045", "ADR-009", "ADR-030", "ADR-024"]
---

## Objective

Ship a **single canonical ingestion API** for programmatic callers and remove all placeholder stubs in `src/utils/document.py` while keeping a thin compatibility facade.

This work MUST:

- remove all `TODO(...)` and `NotImplementedError` placeholders from production ingestion helpers
- avoid duplicating ingestion logic across modules
- preserve offline-first and local-file-only behavior

## Non-goals

- Reintroducing a legacy DocumentProcessor or bespoke ingestion abstraction
- Adding network-based ingestion sources (S3/HTTP) for v1
- Implementing spaCy model downloading (offline-first; models are optional and installed separately)

## User stories

1. As a developer, I can ingest local file paths via a stable API without using Streamlit UI adapters.

2. As a maintainer, docs and examples that reference ingestion helpers do not crash due to placeholder stubs.

3. As a security reviewer, ingestion only accepts local filesystem sources and blocks symlink-based path escape.

## Technical design

### Canonical module

Add a small canonical API module under `src/processing/`:

- `src/processing/ingestion_api.py` (new)

Responsibilities:

- convert local file paths into `IngestionInput` records (stable `document_id`, metadata)
- build an `IngestionConfig` aligned with `settings`
- call `src/processing/ingestion_pipeline.ingest_documents` / `ingest_documents_sync`
- return `src.models.processing.IngestionResult` (existing normalized model)

Suggested public functions (typed):

- `collect_paths(root: Path, *, recursive: bool, extensions: set[str] | None) -> list[Path]`
- `build_inputs_from_paths(paths: Sequence[Path], *, encrypt_images: bool) -> list[IngestionInput]`
- `ingest_paths_sync(paths: Sequence[Path], *, encrypt_images: bool | None = None, cfg: IngestionConfig | None = None) -> IngestionResult`
- `async ingest_paths(paths: Sequence[Path], *, encrypt_images: bool | None = None, cfg: IngestionConfig | None = None) -> IngestionResult`

### Path validation (local-only)

Before ingesting each path:

- `path = Path(path).expanduser()`
- `resolved = path.resolve(strict=True)`
- require `resolved.is_file()`
- block symlinks:
  - `path.is_symlink()` OR any parent segment is a symlink (use `Path.resolve()` + explicit checks)

Directory ingestion MUST:

- enumerate files deterministically (sorted)
- ignore symlinked directories and files
- support extension filtering

### Stable document_id policy

Compute a content hash and use it for the stable ID:

- `sha256 = sha256(file_bytes_streaming)`
- `document_id = f"doc-{sha256[:16]}"`
- store `sha256` in `IngestionInput.metadata["sha256"]`

This matches current Streamlit upload ingest behavior (hash-based IDs) and preserves idempotency.

### Config construction

Provide `build_default_ingestion_config(*, encrypt_images: bool | None) -> IngestionConfig` in `src/processing/ingestion_api.py`:

- derive `chunk_size` / `chunk_overlap` from `settings.processing.*`
- use `settings.cache_dir / "ingestion"` for cache/docstore locations
- propagate observability enablement based on `settings.observability` (do not invent new knobs)

### Legacy facade (`src/utils/document.py`)

Convert the placeholder stubs into a **thin forwarding facade**:

- keep function names and signatures where feasible
- forward to canonical module (`src.processing.ingestion_api`) for behavior
- emit `DeprecationWarning` on import or on call (prefer on call)
- remove all placeholder exceptions and TODOs

Behavior mapping:

- `load_documents_unstructured(paths, settings=None)`: load docs using canonical loader (UnstructuredReader when available; fallback to plain text). Return `list[llama_index.core.Document]`.
- `load_documents_from_directory(dir_path, recursive=True, supported_extensions=None)`: collect paths then delegate to `load_documents_unstructured`.
- `load_documents(paths)`: alias to `load_documents_unstructured`.
- `ensure_spacy_model()`: validate that spaCy + requested model are installed (no downloads). Raise a clear `RuntimeError` with offline-first instructions if missing.
- `get_document_info(path)`: return safe file metadata (`name`, `suffix`, `size_bytes`, `mtime_ns`, `sha256` optional).
- `clear_document_cache()`: remove only `settings.cache_dir / "ingestion"` cache artifacts (guarded path, no egress).
- `get_cache_stats()`: return sizes and file counts for the ingestion cache directory.

### `src/processing/__init__.py`

Remove the placeholder TODO and export canonical public functions (minimal `__all__`) so downstream code can prefer:

```python
from src.processing import ingest_paths_sync
```

## Observability

Emit local JSONL events (no content) for:

- `ingestion.api_ingest`: { file_count, total_bytes, duration_ms, exports_count }
- `ingestion.api_cache_clear`: { deleted_files, deleted_bytes }

## Security

- Inputs are local filesystem only; no URL ingestion.
- Block symlink traversal for both file and directory ingestion.
- Never log raw document content or extracted text.

## Testing strategy

### Unit

- `tests/unit/processing/test_ingestion_api_paths.py` (new):

  - path collection deterministic ordering
  - symlink is rejected
  - document_id hashing stable for same content

- `tests/unit/utils/document/test_document_facade.py` (new or extend existing):
  - facade functions no longer raise `NotImplementedError`
  - cache stats/clear are constrained to `settings.cache_dir`

### Integration

- Optional: a lightweight integration test that ingests a small temp text file with embeddings mocked (offline), asserting `IngestionResult.manifest` is populated.

## Rollout / migration

- Backward compatible for internal callers: facade remains.
- Docs SHOULD migrate to canonical API (handled in WP08).

## RTM updates (docs/specs/traceability.md)

Add a planned row:

- FR-024: “Programmatic ingestion API + legacy facade”
  - Code: `src/processing/ingestion_api.py`, `src/utils/document.py`, `src/processing/__init__.py`
  - Tests: new unit tests (and optional integration)
  - Verification: test
  - Status: Planned → Implemented
