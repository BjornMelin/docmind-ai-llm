---
spec: SPEC-026
title: Unified Ingestion API (Canonical)
version: 2.0.0
date: 2026-01-15
owners: ["ai-arch"]
status: Approved
related_requirements:
  - FR-024: Provide a canonical programmatic ingestion API (local-only).
  - NFR-MAINT-003: No split logic; single ownership domain.
  - NFR-SEC-001: Offline-first defaults MUST remain intact.
related_adrs: ["ADR-045"]
---

## Objective

Refactor the ingestion "loading" layer into a single canonical module `src/processing/ingestion_api.py` and remove the legacy `src/utils/document.py`.

## User stories

1. As a developer, I import all ingestion capabilities from `src.processing` (e.g., `from src.processing import collect_paths, load_documents`).
2. As a maintainer, I verify file loading and path sanitization logic in exactly one place.
3. As an operator, I rely on deterministic hashing and secure path validation that is enforced centrally.

## Technical design

### 1. New Module: `src/processing/ingestion_api.py`

This module accepts **Paths** and produces **LlamaIndex Documents** or **IngestionInputs**.

**Core Responsibilities:**

- **Path Enumeration**: `collect_paths(root: Path, ...)`
  - Recursive globbing.
  - **Security**: strict symlink rejection (parents and leaf).
  - **Determinism**: Sort results by path.
- **File Loading**: `load_documents(paths: Sequence[Path], ...)`
  - Detects file type.
  - Uses `UnstructuredReader` (if available and compatible).
  - fallback to `read_text` (UTF-8).
  - **Security**: Sanitize metadata (remove absolute paths).
- **ID Generation**: `generate_stable_id(file_path)` -> `doc-<sha256[:16]>`.
  - Streaming file hash.
- **Helper Exports**:
  - `clear_ingestion_cache()`: targeted cleanup of `settings.cache_dir / "ingestion"`.

### 2. Migration (Refactor)

The implementation agent must:

1. **Extract** the working logic from `src/utils/document.py`.
2. **Adapt** it to `src/processing/ingestion_api.py`.
    - Ensure imports (like `hashing`, `settings`) are aligned.
3. **Delete** `src/utils/document.py`.
4. **Update Call Sources**:
    - `src/processing/ingestion_pipeline.py` -> call `ingestion_api` functions (instead of internal duplicates or utils).
    - `src/ui/_ingest_adapter_impl.py` -> update imports.
    - `tests/` -> update imports.

### 3. API Signature (Target)

```python
# src/processing/ingestion_api.py

def collect_paths(
    root: Path | str,
    *,
    recursive: bool = True,
    extensions: set[str] | None = None
) -> list[Path]:
    ...

async def load_documents(
    paths: Sequence[Path | str],
    *,
    reader: Any | None = None
) -> list[Document]:
    ...
```

### 4. `src/processing/__init__.py`

Update `__all__` to include these new primitives.

### 5. Documentation Updates (Critical)

The implementation is not complete until **ALL** documentation referencing `src.utils.document` is updated.

- **Search**: `rg "src.utils.document" docs/`
- **Update**: Replace with `src.processing.ingestion_api` or `src.processing`.
- **Verify**: `grep` should return zero matches in `docs/` (except changelogs/ADRs).

## Security & Verification

- **Symlinks**: Must be explicitly rejected. `path.resolve(strict=True)` check against `path.absolute()`.
- **Absolute Paths**: Metadata MUST NOT contain absolute paths (PII/Environment leakage). Use `Path(p).name`.

## Testing

- **Move** `tests/unit/utils/document/test_document_helpers.py` to `tests/unit/processing/test_ingestion_api.py`.
- **Verify** that `load_documents` correctly handles:
  - Missing files (skip/warn).
  - Unstructured missing (fallback).
  - Symlinks (raise/skip).
