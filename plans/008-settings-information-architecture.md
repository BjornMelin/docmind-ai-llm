# Plan 008: Reframe Settings without changing configuration lifetimes

> **Executor instructions**: Preserve every field, validator, endpoint test,
> rollback, persistence, and cache control. Labels may clarify behavior; behavior
> may not change.
>
> **Drift check**:
> `git diff --stat 9accab1..HEAD -- src/pages/04_settings.py tests/unit/pages/test_settings_page_validation.py tests/integration/test_settings_page.py tests/browser/app.spec.ts docs/specs/spec-008-ui-streamlit.md docs/specs/spec-022-settings-ui-hardening.md README.md`
> Plans 001/005 may change job/readiness display. Reconcile only those named
> changes; STOP on other settings contract drift.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: Plans 001, 004, 005
- **Category**: direction
- **Planned at**: commit `9accab1`, 2026-07-16

## Canonical behavior and labels

- Existing `_apply_validated_runtime` behavior remains: apply validated values
  to the current process, rebind runtime, bump cache generation, do not write
  `.env`. Label: **Apply now**. Help: “Use these values in the current app
  process. They are not saved for the next launch.”
- Existing `_persist_env_from_validated` behavior remains: write validated
  `.env`, do not change active runtime. Label: **Save for next launch**. Help:
  “Save these values for the next app launch. The active runtime is unchanged.”
- Do not add “Apply and save” or change either lifetime in this plan.

## Exact grouping

| Existing symbols | Section |
| --- | --- |
| `_render_provider_section`, `_render_model_section`, provider URLs/OpenAI/Ollama fields | Runtime |
| `_render_document_parsing_section`, `_render_parsing_health` | Documents and parsing |
| `_render_retrieval_section`, `_render_graphrag_section` | Retrieval and GraphRAG |
| `_render_security_section`, llama.cpp guide, endpoint allowlist | Security and advanced |
| `_render_endpoint_test`, `_render_actions` | final Review and actions |
| `_render_cache_controls` | separate Advanced maintenance container |

Provider selection and “Use preset values” remain outside forms because they
must rerun dependent fields immediately. Place the validated dependent fields
inside one native `st.form`; use `st.form_submit_button` for endpoint test,
Apply now, and Save for next launch. If locked Streamlit cannot support the
required multiple-submit behavior, use one form per exact section while keeping
one canonical candidate builder; do not duplicate validation.

## Steps

1. Add a tested active-versus-saved summary using current settings and only
   presence/difference metadata; never display secret values.
2. Recompose existing render functions into the exact sections above. Preserve
   every field and immediate provider/preset rerun.
3. Convert action buttons to exact labels/help while preserving functions and
   rollback/persistence semantics. Align SPEC-008 to the later SPEC-022 owner
   and explain lifetimes in README.
4. Add AppTest/browser cases for provider switch, preset, invalid candidate,
   endpoint test, Apply now, Save for next launch, secret redaction, cache
   controls, keyboard/mobile layout, and no `.env` write during Apply. Update
   Plan 004's browser assertions from the old `Apply runtime`/`Save` labels to
   the exact new labels/help in this plan.

## Scope

Only drift-check files. No settings schema/env-key change, provider API change,
new secret storage, runtime/persistence lifetime change, or removed control.

## Git workflow

Use `feat/ui-foundation`; commit `feat(ui): clarify settings lifetimes`. Do not
push/open a PR before parent review.

## Verification

```bash
uv run pytest tests/unit/pages/test_settings_page_validation.py tests/integration/test_settings_page.py -q --no-cov
bun run test:browser
uv run ruff format --check .
uv run ruff check .
uv run pyright --threads 4
uv run pytest tests/unit tests/integration -q --no-cov
uv run python scripts/check_links.py
npx --yes markdownlint-cli@0.47.0 --disable MD013 MD033 MD041 -- README.md docs/specs/spec-008-ui-streamlit.md docs/specs/spec-022-settings-ui-hardening.md plans/008-settings-information-architecture.md
```

Expected: every command exits 0; both browser projects pass.

## Done criteria

- [ ] Every existing Settings field/action is mapped and tested.
- [ ] Apply now never writes `.env`; Save for next launch never mutates runtime.
- [ ] Secret values never appear in summaries, logs, screenshots, or errors.
- [ ] Immediate provider/preset behavior remains intact.

## STOP conditions

Stop if native forms cannot preserve immediate provider/preset behavior without
duplicating state/validation, if any field has no mapped destination, or if a
requested change alters Apply/Save lifetimes. Report before changing behavior.

## Maintenance notes

Future configuration actions must state current-process versus persisted-next-
launch lifetime explicitly and reuse the canonical candidate validator.
