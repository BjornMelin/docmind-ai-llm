# Plan 005: Add the guided workspace shell

> **Executor instructions**: Execute only after Plans 001–004. Change shared
> hierarchy, not page-specific capabilities. Update `plans/README.md`.
>
> **Drift check**:
> `git diff --stat 9accab1..HEAD -- src/app.py src/ui/readiness.py tests/unit/ui/test_readiness.py tests/integration/ui/test_app_smoke_flows.py tests/browser/app.spec.ts docs/developers/adrs/ADR-013-user-interface-architecture.md README.md`
> Earlier plans are expected to change job/chat tests and browser harness files.
> Plan 003 is specifically expected to relocate
> `recover_snapshot_transactions` behind `_recover_persistence_once` while
> preserving fail-closed behavior. Reconcile those named changes; STOP on other
> drift in `src/app.py` or a pre-existing `src/ui/readiness.py` contract.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW
- **Depends on**: Plans 001–004
- **Category**: direction
- **Planned at**: commit `9accab1`, 2026-07-16

## Why this matters

The four pages are presented equally even though the successful journey is
Configure → Ingest → Ask. A compact, local readiness shell can make the next
safe action obvious without blocking direct navigation or adding network work.

## Exact capability mapping

| Existing capability | Destination | Contract |
| --- | --- | --- |
| `st.navigation` Chat/Documents/Analytics/Settings | `src/app.py` | unchanged direct navigation |
| provider/runtime configured state | shared status rail | derive from settings only; no probe |
| current snapshot/staleness | shared status rail | local manifest/files only |
| active corpus/analysis job | shared status rail | Plan 001 process manager, owner-scoped display |
| persistence recovery failure | app shell | unchanged fail-closed stop |

## Scope

Create `src/ui/readiness.py` with frozen typed readiness values and pure next-
step selection. Modify `src/app.py`, direct tests, browser cases, ADR, and README
only. Do not move page widgets or add CSS/JS.

## Steps

1. Implement pure readiness derivation from already-loaded settings, local
   snapshot state, and owner-visible job state. No network calls or model loads.
   **Verify**: unit table covers Configure, Add documents, Rebuild, Ask, and
   active-work states.
2. Render one compact native sidebar/status rail in `src/app.py` with state,
   reason, and page link/next action. Keep all four navigation items directly
   reachable. **Verify**: AppTest asserts titles/navigation and no heavy page
   import is introduced.
3. Add desktop/mobile browser assertions for reachability, focus order, and no
   overflow. Update ADR/README with the workflow and local-only derivation.

## Git workflow

Use `feat/ui-foundation`; commit `feat(ui): add guided workspace status`. Do not
push/open a PR before parent review.

## Verification

```bash
uv run pytest tests/unit/ui/test_readiness.py tests/integration/ui/test_app_smoke_flows.py -q --no-cov
uv run python scripts/check_ui_import_boundary.py
bun run test:browser
uv run ruff format --check .
uv run ruff check .
uv run pyright --threads 4
uv run pytest tests/unit tests/integration -q --no-cov
uv run python scripts/check_links.py
npx --yes markdownlint-cli@0.47.0 --disable MD013 MD033 MD041 -- README.md docs/developers/adrs/ADR-013-user-interface-architecture.md plans/005-workspace-shell.md
```

Expected: every command exits 0; both browser projects pass.

## Done criteria

- [ ] Configure → Ingest → Ask next steps are derived locally and tested.
- [ ] Direct navigation and persistence recovery behavior are unchanged.
- [ ] No readiness render triggers network, model, or retrieval initialization.
- [ ] `src.app` and the audited page modules still pass the fixed import gate.
- [ ] Desktop/mobile browser and all Python/docs gates pass.

## STOP conditions

Stop if readiness requires probing a provider/Qdrant, duplicating snapshot
ownership, or importing page modules into the shell. Stop if direct navigation
would be hidden or gated.

## Maintenance notes

New readiness states belong in the typed owner and its decision table, not in
page-specific conditional copy.
