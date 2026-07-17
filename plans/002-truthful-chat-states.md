# Plan 002: Make Chat loading and generation states truthful

> **Executor instructions**: Follow every step and gate. Update the status row
> in `plans/README.md` when done. Do not expose raw exceptions or identifiers.
>
> **Drift check**:
> `git diff --stat 488f1ab..HEAD -- src/pages/01_chat.py tests/unit/pages/test_chat_page_helpers.py tests/unit/ui/test_chat_page_helpers.py tests/integration/ui/test_app_smoke_flows.py tests/integration/ui/test_chat_persistence_time_travel.py docs/specs/spec-008-ui-streamlit.md docs/specs/requirements.md docs/developers/adrs/ADR-011-agent-orchestration-framework.md docs/developers/adrs/ADR-013-user-interface-architecture.md docs/specs/spec-007-multi-agent-supervisor.md docs/specs/traceability.md CHANGELOG.md`
> Plan 001's foreground runtime leases and current coordinator/session handle
> reacquisition in `_load_chat_messages` and `_handle_chat_prompt` are the
> accepted baseline. Commit `488f1ab` adds the merged test-only AppTest
> module-registry isolation fix; it does not change production behavior.

## Status

- **Status**: DONE
- **Validated implementation**: commit `7a3e69b`, 2026-07-16
- **Priority**: P1
- **Effort**: S–M
- **Risk**: LOW
- **Depends on**: `plans/001-mutation-job-ownership.md`
- **Category**: bug
- **Planned at**: merged main commit `488f1ab`, 2026-07-16

## Why this matters

A checkpoint read failure currently becomes an empty list, so an unreadable
conversation looks like a new chat. Model generation is synchronous, but the
completed answer is then chunked through `st.write_stream`, simulating progress
that did not occur. Users need explicit, sanitized loading/error/status states,
especially in a private document-analysis tool where history and provenance
matter.

## Current state

- `src/pages/01_chat.py` logs any history error and returns `[]`; maintenance
  also renders from inside the loader and returns the same value.
- `main()` renders that value as a normal history, so empty success and failure
  are indistinguishable.
- `tests/unit/ui/test_chat_page_helpers.py` locks in the empty-list behavior.
- `_handle_chat_prompt` waits for `process_query` to finish, then
  animates the finished string with `_chunked_stream`.
- `docs/specs/spec-008-ui-streamlit.md` is only 59 lines; stale streaming and
  performance claims also live in the requirements, SPEC-007, ADR-011, and
  ADR-013 contracts.

Follow the existing `build_pii_log_entry` pattern for diagnostic redaction and
the background analysis panel for status language. Do not add fake token
timers or threads.

## Commands

| Purpose | Command | Expected |
| --- | --- | --- |
| Focused | `uv run pytest tests/unit/pages/test_chat_page_helpers.py tests/unit/ui/test_chat_page_helpers.py tests/integration/ui/test_app_smoke_flows.py tests/integration/ui/test_chat_persistence_time_travel.py -q --no-cov` | all pass |
| Quality | `uv run ruff format --check . && uv run ruff check . && uv run pyright --threads 4` | exit 0 |
| Full | `uv run pytest tests/unit tests/integration -q --no-cov` | all pass |

## Scope

**In scope**: `src/pages/01_chat.py`, direct Chat AppTests/unit and time-travel
tests, `docs/specs/spec-008-ui-streamlit.md`, `docs/specs/requirements.md`,
`docs/developers/adrs/ADR-011-agent-orchestration-framework.md`,
`docs/developers/adrs/ADR-013-user-interface-architecture.md`,
`docs/specs/spec-007-multi-agent-supervisor.md`,
`docs/specs/traceability.md` (including FR-011), and `CHANGELOG.md`.

**Out of scope**: coordinator/LangGraph streaming API design, checkpoint schema,
message-source persistence, provider configuration, or retrieval behavior.

## Git workflow

Use the isolated `feat/truthful-chat-states` branch/worktree. Use a Conventional
Commit such as `fix(ui): make chat states truthful`. Do not push or open a PR
until the parent orchestrator reviews the diff.

## Steps

### 1. Represent history load outcome explicitly

Return a small frozen typed result containing `messages` and a sanitized
failure flag/reference, or move rendering into a typed error boundary. Main
must distinguish success-with-zero-messages from failure. On failure, show an
error that history could not be loaded, offer a normal rerun/retry affordance,
and avoid implying the thread is empty. On maintenance or failure, `main()` must
return before rendering history or accepting a prompt. Keep raw exception text
out of UI.

The loader must not render Streamlit elements. Preserve the Plan 001 foreground
lease while it acquires and reads the current coordinator.

The public coordinator intentionally returns `{}` for no checkpoint, failed
setup, a missing graph, or a closed coordinator. Plan 002 treats that public
empty state as ready because distinguishing those causes requires a coordinator
contract change. It distinguishes raised reads and maintenance without changing
that API.

**Verify**: unit and AppTest cases distinguish empty success, maintenance, and
sanitized failure; retry reruns the page without exposing raw errors or IDs.

### 2. Replace simulated streaming with honest native status

Remove `_chunked_stream` and its tests. No real incremental public coordinator
event API exists at this baseline: `MultiAgentCoordinator.process_query`
returns one completed `AgentResponse`, while private graph consumption is not a
UI contract. Wrap only `process_query` in
`st.spinner("Generating response…", show_time=True)`, then render the completed
answer with `st.markdown`. Preserve the Plan 001 foreground lease,
checkpoint/session-touch ordering, current error IDs, source rendering, and all
response content.

If a genuine incremental coordinator API lands after the `488f1ab` baseline,
STOP and report it rather than designing a second adapter inside this plan.

**Verify**: a unit test monkeypatches `st.spinner` and proves it opens before
`process_query`, closes immediately after the call, and does not enclose
checkpoint/session touch. AppTest proves final answer/error rendering; it does
not claim to observe the intermediate synchronous frame. No code calls
`_chunked_stream` or animates an already-complete string.

### 3. Align the UI contract

Align every governed document in this plan on truthful synchronous status.
Real incremental streaming is future work that requires a public coordinator
event API and end-to-end proof that output arrives before completion. Do not
claim time-to-first-token behavior that is not measured.

## Test plan

Cover zero-message success, sanitized load failure, successful request,
provider failure, source rendering after success, and status cleanup. Existing
checkpoint/session touch assertions must remain.

## Done criteria

- [x] A raised history read failure is visible and cannot be mistaken for the
  coordinator's successful empty state.
- [x] No raw exception, thread ID, user ID, or provider secret reaches UI.
- [x] No post-hoc fake token streaming remains.
- [x] Sources and checkpoint/session touch behavior remain intact.
- [x] Focused, Ruff, Pyright, and full tests pass.

## Validation record

- Focused Chat unit/AppTest/time-travel suite: 75 passed.
- Full unit and integration suite: 1,521 passed, 3 skipped.
- Ruff format/check and Pyright: passed.
- Internal links, structural parity, schema validation, and Markdownlint: passed.

## STOP conditions

Stop if the fix requires changing checkpoint serialization or coordinator
public contracts. Stop if a live incremental API exists; report its exact
signature so a separate real-streaming plan can replace Step 2.

## Maintenance notes

A future streaming implementation must prove output arrives before completion
and preserve cancellation, checkpointing, sources, and sanitized failures. Do
not reintroduce fixed-size chunk animation as a proxy.
