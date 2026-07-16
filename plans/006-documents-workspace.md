# Plan 006: Build the Documents corpus workspace

> **Executor instructions**: Preserve every ingestion, snapshot, export,
> rebuild, and deletion capability. Use only canonical local metadata.
>
> **Drift check**:
> `git diff --stat 9accab1..HEAD -- src/pages/02_documents.py src/ui/corpus_inventory.py tests/unit/ui/test_corpus_inventory.py tests/unit/pages/test_documents_page_helpers.py tests/integration/ui/test_documents_ingestion_job.py tests/browser/app.spec.ts docs/developers/adrs/ADR-013-user-interface-architecture.md README.md`
> Plan 001 is expected to change mutation-control signatures and tests. Plan 003
> is expected to move heavy Documents imports behind action/resource boundaries.
> Plan 005 may add shared readiness display. Reconcile only those named changes;
> STOP on other drift in Documents ownership.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: Plans 001, 003, 004, and 005
- **Category**: direction
- **Planned at**: commit `9accab1`, 2026-07-16

## Why this matters

Documents has ingestion, maintenance, snapshots, and exports but no corpus
inventory, despite ADR-013 requiring a native sortable/filterable table. Users
cannot inspect what is present before rebuild/delete actions.

## Exact capability and layout mapping

| Current symbol | New location/order | Required behavior |
| --- | --- | --- |
| `_render_latest_snapshot_summary` | top status container | unchanged manifest details |
| `_render_ingest_form` | primary Add documents section | all parser/encryption/GraphRAG controls preserved |
| `_render_ingest_job_panel` | immediately after ingest | Plan 001 progress/cancel/exclusivity preserved |
| new corpus inventory | after active job | native `st.dataframe`, bounded local rows |
| `_render_maintenance_controls` | Advanced maintenance | rebuild/delete, Plan 001 confirmation preserved |
| `_render_export_controls` | Exports section | JSONL/Parquet and image preview preserved |

## Metadata rules

Create `src/ui/corpus_inventory.py`. For direct regular files under canonical
`settings.data_dir / "uploads"`, expose only data available without Qdrant or a
new persistence contract: filename, suffix/type, byte size, and filesystem
modified time. A document ID may be computed only on explicit selected-row
detail using the existing SHA helper; do not hash every large file on each
rerun. Do not invent per-document indexed status or last-snapshot membership.
Show one explanatory caption that active manifests currently track corpus-level
identity. Unknown/unavailable values render as `—`.

## Steps

1. Implement pure bounded inventory collection with path-safety and a maximum
   row count; return typed rows and truncation state. **Verify**: unit tests cover
   empty, files, directories ignored, symlink/path policy, unknown stat, sorting,
   and truncation.
2. Render native `st.dataframe` with explicit column config and single-row
   selection if supported by locked Streamlit. Reuse the selected filename for
   details/maintenance; otherwise retain the safe selectbox inside maintenance.
   **Verify**: AppTest covers empty and populated states without Qdrant.
3. Reorder existing sections exactly as mapped. Do not copy business logic or
   remove advanced controls. **Verify**: a capability inventory test/assertion
   covers every existing label/action before and after.
4. Update ADR/README and browser desktop/mobile table/overflow/keyboard cases.

## Scope

Only files in the drift check plus direct docs. No snapshot schema, Qdrant scan,
parser/retrieval change, new table dependency, or per-document provenance model.

## Git workflow

Use `feat/ui-foundation`; commit `feat(ui): add documents corpus workspace`.
Do not push/open a PR before parent review.

## Verification

```bash
uv run pytest tests/unit/ui/test_corpus_inventory.py tests/unit/pages/test_documents_page_helpers.py tests/integration/ui/test_documents_ingestion_job.py -q --no-cov
bun run test:browser
uv run ruff format --check .
uv run ruff check .
uv run pyright --threads 4
uv run pytest tests/unit tests/integration -q --no-cov
uv run python scripts/check_links.py
npx --yes markdownlint-cli@0.47.0 --disable MD013 MD033 MD041 -- README.md docs/developers/adrs/ADR-013-user-interface-architecture.md plans/006-documents-workspace.md
```

Expected: every command exits 0; both browser projects pass.

## Done criteria

- [ ] Native bounded corpus inventory uses only canonical local file metadata.
- [ ] No UI claims per-document indexed/snapshot status without evidence.
- [ ] Every pre-existing Documents capability remains mapped and tested.
- [ ] Plan 001 mutation safety remains intact.

## STOP conditions

Stop if the requested table appears to require Qdrant scans, hashing all files
per rerun, a new persistence schema, following unsafe paths, or another table
dependency. Render `—` or explain the corpus-level boundary instead.

## Maintenance notes

If a future manifest adds per-document provenance, add it through the typed
inventory owner with versioned tests; do not infer it from collection presence.
