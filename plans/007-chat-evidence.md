# Plan 007: Polish Chat and current-answer evidence

> **Executor instructions**: Preserve Chat, memory, analysis, visual search,
> sessions, sources, and checkpoints. Do not claim historical source support.
>
> **Drift check**:
> `git diff --stat 4fea380..HEAD -- src/pages/01_chat.py src/ui/components/evidence.py tests/unit/ui/test_chat_page_helpers.py tests/integration/ui/test_app_smoke_flows.py tests/integration/ui/test_analysis_modes.py tests/browser/app.spec.ts docs/specs/spec-008-ui-streamlit.md README.md`
> Plans 002–005 are expected to change truthful status, job calls, imports, and
> the shared shell. Reconcile those named changes; STOP on unexplained Chat
> capability drift.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: Plans 002, 003, 004, and 005
- **Category**: direction
- **Planned at**: commit `4fea380`, 2026-07-16

## Exact mapping

| Capability | Destination | Contract |
| --- | --- | --- |
| session/time-travel sidebars | sidebar, grouped Conversation section | unchanged IDs/checkpoints |
| memory controls | sidebar, collapsed Memories section | all add/search/delete/purge behavior preserved |
| visual search | sidebar, collapsed Visual search | unchanged upload/query/results |
| analysis controls/results | secondary Analyze section | modes, cancellation, citations count preserved |
| composer/history | primary content | Plan 002 truthful status/error behavior |
| current response sources | native Evidence expander/cards | sanitized doc/page/modality/artifacts preserved |

## Steps

1. Capture current desktop/mobile browser states and capability labels.
2. Extract only source rendering into `src/ui/components/evidence.py` with typed
   sanitized view rows. Preserve `ArtifactRef` rendering and source truncation;
   no raw path or blob display. Unit-test text/image/missing/encrypted states.
3. Reorder native containers so conversation/composer are primary and Memories,
   Visual search, and Analyze are progressively disclosed but directly
   reachable. Use native containers/expanders/status only; no DOM CSS/JS.
4. Render current-answer evidence with clear document/page/modality labels and
   accessible image captions. Historical messages continue to render content
   only; add explicit copy only if needed to avoid implying persisted evidence.
5. Browser-test desktop/mobile focus, overflow, empty, history failure,
   provider failure, current sources, analysis, and sidebar access. Align spec/
   README without claiming real streaming or persisted historical evidence.

## Scope

Only drift-check files. No checkpoint/message schema, source persistence,
retrieval result shape, model API, or new component dependency.

## Git workflow

Use `feat/ui-foundation`; commit `feat(ui): polish chat evidence workspace`.
Do not push/open a PR before parent review.

## Verification

```bash
uv run pytest tests/unit/pages/test_chat_page_helpers.py tests/unit/ui/test_chat_page_helpers.py tests/integration/ui/test_app_smoke_flows.py tests/integration/ui/test_analysis_modes.py -q --no-cov
bun run test:browser
uv run ruff format --check .
uv run ruff check .
uv run pyright --threads 4
uv run pytest tests/unit tests/integration -q --no-cov
uv run python scripts/check_links.py
npx --yes markdownlint-cli@0.47.0 --disable MD013 MD033 MD041 -- README.md docs/specs/spec-008-ui-streamlit.md plans/007-chat-evidence.md
```

Expected: every command exits 0; both browser projects pass.

## Done criteria

- [ ] Every mapped Chat capability remains reachable and tested.
- [ ] Current-answer evidence is sanitized, accessible, and artifact-safe.
- [ ] UI does not imply historical evidence is persisted.
- [ ] No raw paths/secrets, custom DOM code, or retrieval/checkpoint changes.

## STOP conditions

Stop if layout requires changing coordinator, checkpoint, memory, retrieval, or
artifact contracts. Stop if historical evidence is requested; specify that
separate versioned persistence plan instead.

## Maintenance notes

When persisted per-message evidence lands, extend the shared evidence component
rather than adding a second historical renderer.
