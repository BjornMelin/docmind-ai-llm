# Plan 002: Make Chat loading and generation states truthful

> **Executor instructions**: Follow every step and gate. Update the status row
> in `plans/README.md` when done. Do not expose raw exceptions or identifiers.
>
> **Drift check**:
> `git diff --stat 4fea380..HEAD -- src/pages/01_chat.py tests/unit/pages/test_chat_page_helpers.py tests/unit/ui/test_chat_page_helpers.py tests/integration/ui/test_chat_streaming_timeout_flow.py tests/integration/ui/test_streaming_chunking.py tests/integration/ui/test_app_smoke_flows.py docs/specs/spec-008-ui-streamlit.md`
> Plan 001 is expected to remove the `settings.cache_version` argument from the
> Chat `get_job_manager` caller and update direct job-lifecycle tests. Reconcile
> only that named change; any other drift in Chat/history/streaming symbols is a
> STOP condition.

## Status

- **Priority**: P1
- **Effort**: S–M
- **Risk**: LOW
- **Depends on**: `plans/001-mutation-job-ownership.md`
- **Category**: bug
- **Planned at**: commit `4fea380`, 2026-07-16

## Why this matters

A checkpoint read failure currently becomes an empty list, so an unreadable
conversation looks like a new chat. Model generation is synchronous, but the
completed answer is then chunked through `st.write_stream`, simulating progress
that did not occur. Users need explicit, sanitized loading/error/status states,
especially in a private document-analysis tool where history and provenance
matter.

## Current state

- `src/pages/01_chat.py:676-699` logs any history error and returns `[]`.
- `src/pages/01_chat.py:260-262` renders that value as a normal history.
- `tests/unit/ui/test_chat_page_helpers.py:705-716` locks in the empty-list
  behavior.
- `src/pages/01_chat.py:727-752` waits for `process_query` to finish, then
  animates the finished string with `_chunked_stream`.
- `docs/specs/spec-008-ui-streamlit.md:88-104` describes native streaming and
  performance targets the coordinator does not currently implement.

Follow the existing `build_pii_log_entry` pattern for diagnostic redaction and
the background analysis panel for status language. Do not add fake token
timers or threads.

## Commands

| Purpose | Command | Expected |
| --- | --- | --- |
| Focused | `uv run pytest tests/unit/pages/test_chat_page_helpers.py tests/unit/ui/test_chat_page_helpers.py tests/integration/ui/test_chat_streaming_timeout_flow.py tests/integration/ui/test_streaming_chunking.py tests/integration/ui/test_app_smoke_flows.py -q --no-cov` | all pass |
| Quality | `uv run ruff format --check . && uv run ruff check . && uv run pyright --threads 4` | exit 0 |
| Full | `uv run pytest tests/unit tests/integration -q --no-cov` | all pass |

## Scope

**In scope**: `src/pages/01_chat.py`, direct Chat AppTests/unit tests, and the
governing UI spec if its claimed behavior changes.

**Out of scope**: coordinator/LangGraph streaming API design, checkpoint schema,
message-source persistence, provider configuration, or retrieval behavior.

## Git workflow

Use the existing `feat/ui-foundation` branch/worktree. Use a Conventional
Commit such as `fix(ui): make chat states truthful`. Do not push or open a PR
until the parent orchestrator reviews the diff.

## Steps

### 1. Represent history load outcome explicitly

Return a small frozen typed result containing `messages` and a sanitized
failure flag/reference, or move rendering into a typed error boundary. Main
must distinguish success-with-zero-messages from failure. On failure, show an
error that history could not be loaded, offer a normal rerun/retry affordance,
and avoid implying the thread is empty. Keep raw exception text out of UI.

**Verify**: unit and AppTest cases distinguish empty success from failed load.

### 2. Replace simulated streaming with honest native status

Remove `_chunked_stream` and its tests unless a real incremental coordinator
event API exists at execution time. Wrap the synchronous call in a native
`st.status`/spinner state with truthful copy, then render the completed answer
normally. Preserve current error IDs, source rendering, checkpoint touch, and
all response content.

If a genuine incremental coordinator API has landed since `4fea380`, STOP and
report it rather than designing a second adapter inside this plan.

**Verify**: a unit test monkeypatches `st.status` (or the selected native status
primitive) and proves it opens before `process_query` and completes after the
call. AppTest proves final answer/error rendering. Do not claim AppTest observes
the intermediate synchronous frame. No code calls `_chunked_stream` or
animates an already-complete string.

### 3. Align the UI contract

Update the governing spec to describe synchronous status honestly, or link a
separate future real-streaming requirement. Do not claim time-to-first-token
behavior that is not measured.

## Test plan

Cover zero-message success, sanitized load failure, successful request,
provider failure, source rendering after success, and status cleanup. Existing
checkpoint/session touch assertions must remain.

## Done criteria

- [ ] History read failure is visible and cannot be mistaken for an empty chat.
- [ ] No raw exception, thread ID, user ID, or provider secret reaches UI.
- [ ] No post-hoc fake token streaming remains.
- [ ] Sources and checkpoint/session touch behavior remain intact.
- [ ] Focused, Ruff, Pyright, and full tests pass.

## STOP conditions

Stop if the fix requires changing checkpoint serialization or coordinator
public contracts. Stop if a live incremental API exists; report its exact
signature so a separate real-streaming plan can replace Step 2.

## Maintenance notes

A future streaming implementation must prove output arrives before completion
and preserve cancellation, checkpointing, sources, and sanitized failures. Do
not reintroduce fixed-size chunk animation as a proxy.
