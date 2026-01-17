---
prompt: PROMPT-031
title: Config Discipline (Remove `os.getenv` Sprawl)
status: Completed
date: 2026-01-16
version: 1.0
related_adrs: ["ADR-050"]
related_specs: ["SPEC-031"]
---

Implements `ADR-050` + `SPEC-031`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-050-config-discipline-env-bridges.md`
- SPEC: `docs/specs/spec-031-config-discipline-env-bridges.md`
- Requirements: `docs/specs/requirements.md` (NFR-MAINT-003, NFR-SEC-001)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.pydantic.dev/latest/concepts/pydantic_settings/> — Pydantic Settings v2 patterns (nested env parsing; precedence).
- <https://docs.pydantic.dev/latest/concepts/types/#secret-types> — Secret handling types (`SecretStr`) and best practices.
- <https://opentelemetry.io/docs/languages/python/> — OpenTelemetry Python docs (resource/env configuration; exporters).
- <https://bbc2.github.io/python-dotenv/reference/> — python-dotenv reference (if `.env` persistence touches quoting semantics).

## Tooling & Skill Strategy (fresh Codex sessions)

This is security-sensitive config work. Prefer repo truth and run structured audits.

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

**Primary tools to leverage:**

- `rg` to inventory every `os.getenv` call and ensure all are removed from core modules.
- Use `multi_tool_use.parallel` for independent inventories (env reads, doc references, tests).
- Context7 for authoritative Pydantic Settings v2 patterns and typing (nested env parsing).
- `opensrc/` for Pydantic internals only when behavior is surprising (prefer docs first).
- `functions.mcp__zen__secaudit` (mandatory) after changes: confirm no secret logging and no new egress surfaces.
- `functions.mcp__zen__codereview` for final correctness gate.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Inventory env reads + config drift:
  - `rg -n \"os\\.getenv\\(\" -S src`
  - `rg -n \"DOCMIND_(TELEMETRY|IMG|ENVIRONMENT)\" -S src docs tests .env.example`
  - `rg -n \"ADR-047\" -S src docs || true`
- Read in parallel:
  - `src/config/settings.py`
  - `src/utils/telemetry.py`
  - `src/processing/pdf_pages.py`
  - `src/telemetry/opentelemetry.py`

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for Pydantic settings resources; read them before web search.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc`
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain`
- OpenAI API docs: `functions.mcp__openaiDeveloperDocs__search_openai_docs` → `functions.mcp__openaiDeveloperDocs__fetch_openai_doc` (only if this work package touches OpenAI API semantics)

**API verification (Context7):**

- `functions.mcp__context7__resolve-library-id` → `pydantic`, `pydantic-settings`
- `functions.mcp__context7__query-docs` → confirm:
  - nested env parsing and precedence rules
  - how to represent `SecretStr` or sensitive fields safely (if used)

**opensrc (only when behavior is surprising):**

- Check first: `cat opensrc/sources.json | rg -n \"pydantic\"`
- Only fetch if missing and you must confirm an edge case; treat `opensrc/` as read-only.

**Security gate (required):**

- Run `functions.mcp__zen__secaudit` after refactor:
  - confirm secrets are not logged
  - confirm any OTLP exporters remain gated/off by default

**Review gate (required if broad changes):**

- Run `functions.mcp__zen__codereview` after tests pass.

**MCP tool sequence (use when it adds signal):**

1. `functions.mcp__zen__planner` → plan settings schema + refactors + tests.
2. Context7:
   - resolve `pydantic` (and `pydantic-settings`) and query docs for env mapping and nested keys.
3. Exa search (official pydantic docs) if a validator/SettingsConfigDict behavior is unclear.

## IMPLEMENTATION EXECUTOR TEMPLATE (DOCMIND / PYTHON)

### YOU ARE

You are an autonomous implementation agent for the **DocMind AI LLM** repository.

You will implement the feature described below end-to-end, including:

- code changes
- tests
- documentation updates (ADR/SPEC/RTM)
- deletion of dead code and removal of legacy/backcompat shims within scope

You must keep changes minimal, library-first, and maintainable.

---

### FEATURE CONTEXT (FILLED)

**Primary Task:** Complete config discipline work by eliminating remaining `os.getenv` drift and formalizing the hashing secret wiring described in ADR-050.

**Why now:** Scattered env reads undermine settings discipline and hide security-sensitive toggles. There is also an `ADR-047` marker and a hashing secret that must be formalized for keyed fingerprints (safe logging correlation) and for the existing HMAC canonicalization utilities in `src/utils/canonicalization.py` (so tests and production share the same policy).

**Definition of Done (DoD):**

- `DOCMIND_TELEMETRY_*` and `DOCMIND_ENVIRONMENT` map through `DocMindSettings` (and/or explicit env bridges in `src/config/*`), so `.env` and env vars both work per Pydantic Settings precedence (env > `.env`).
- `src/utils/telemetry.py` and `src/telemetry/opentelemetry.py` do not call `os.getenv` (env reads, if required, live in `src/config/*`).
- `ADR-047` marker removed (replaced with real ADR reference) and `settings.hashing.hmac_secret` is treated as a real local secret:
  - validator error messages reference `DOCMIND_HASHING__HMAC_SECRET`
  - the secret can be used by `src/utils/log_safety.py` (SPEC-028) for keyed fingerprints and by `src/utils/canonicalization.py` for canonical hashes
- Unit tests validate env→settings mapping and telemetry behavior.
- Quality gates pass.

**In-scope modules/files (initial):**

- `src/config/settings.py`
- `src/utils/log_safety.py`
- `src/utils/canonicalization.py`
- `tests/unit/config/` (new tests)
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Renaming env vars.
- Introducing a new config layer outside settings.

---

### HARD RULES (EXECUTION)

#### 1) Config discipline

- Source of truth is `src/config/settings.py` (Pydantic Settings v2).
- Do not add new `os.getenv` usage outside `src/config/*`.
- Do not add backward-compatible env var aliases unless explicitly required by a spec/ADR.
- Do not weaken offline-first policy or allow remote endpoints by default.

#### 2) Style, Types, and Lint

Must pass:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run pyright`

#### 3) Security posture

- Never log secrets.
- Any telemetry/exporter egress must remain gated by config and default-off.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

1. [ ] Inventory env reads: `rg "os\\.getenv\\(" src/`.
2. [ ] Add new settings groups and mappings in `src/config/settings.py`.
3. [ ] Refactor remaining consumers to use `settings` (no env reads outside `src/config/*`).
4. [ ] Replace `ADR-047` marker with a real ADR reference and formalize hashing secret UX:
   - update validator error message to reference `DOCMIND_HASHING__HMAC_SECRET`
   - add a unit test for the mapping/validation message
5. [ ] Add/update unit tests for settings mappings and telemetry.
6. [ ] Update RTM row and run quality gates.

Commands:

```bash
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run python scripts/run_tests.py --fast
uv run python scripts/run_tests.py
```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. Keeping scattered `os.getenv` reads “because it works” (must centralize in settings).
2. Introducing new env var names (explicitly out-of-scope) instead of mapping existing ones.
3. Logging env values or full base URLs that may contain credentials.
4. Making exporters/remote endpoints enabled by default (violates offline-first).

---

### MCP TOOL STRATEGY (FOR IMPLEMENTATION RUN)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → implementation plan (settings schema + refactors + tests).
2. `functions.mcp__exa__web_search_exa` / `functions.mcp__exa__crawling_exa` → official Pydantic Settings docs if time-sensitive behavior is unclear.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → authoritative API details (`pydantic`, `pydantic-settings`).
4. `functions.mcp__gh_grep__searchGitHub` → optional patterns for nested env parsing and secret handling.
5. `functions.mcp__zen__analyze` → if changes span config + telemetry + processing modules.
6. `functions.mcp__zen__codereview` → post-implementation review.
7. `functions.mcp__zen__secaudit` → mandatory security audit (secrets + egress surfaces).

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel reads.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement     | Status | Proof / Notes                                                                                           |
| --------------- | ------ | ------------------------------------------------------------------------------------------------------- |
| **Packaging**   |        | `uv sync` clean                                                                                         |
| **Formatting**  |        | `uv run ruff format .`                                                                                  |
| **Lint**        |        | `uv run ruff check .` clean                                                                             |
| **Types**       |        | `uv run pyright` clean                                                                                  |
| **Tests**       |        | mapping tests green; `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py` |
| **Docs**        |        | RTM updated                                                                                             |
| **Security**    |        | no secret logs; egress surfaces remain gated/off by default                                             |
| **Tech Debt**   |        | zero work-marker placeholders introduced                                                                |
| **Performance** |        | no new import-time heavy work; settings load remains fast                                               |

**EXECUTE UNTIL COMPLETE.**
