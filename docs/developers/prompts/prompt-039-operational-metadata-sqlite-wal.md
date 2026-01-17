# Implementation Prompt — Operational Metadata Store (SQLite WAL)

Implements `ADR-055` + `SPEC-039`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-055-operational-metadata-sqlite-wal.md`
- SPEC: `docs/specs/spec-039-operational-metadata-sqlite-wal.md`
- Requirements: `docs/specs/requirements.md` (FR-015, FR-025)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.python.org/3/library/sqlite3.html> — stdlib sqlite3 connection/transactions.
- <https://www.sqlite.org/wal.html> — WAL mode semantics, concurrency, durability tradeoffs.
- <https://www.sqlite.org/pragma.html> — PRAGMAs (`journal_mode`, `foreign_keys`, `busy_timeout`, `user_version`).
- <https://duckdb.org/docs/stable/connect/concurrency.html> — Concurrency notes (for separation of ops DB vs. analytics/cache).

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

This is a persistence feature. No special skill is required, but if you touch Streamlit UI pages or AppTest, load `$streamlit-master-architect`.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Locate existing sqlite usage and planned integration points:
  - `rg -n \"sqlite3|sqlite_db_path|enable_wal_mode|PRAGMA\" -S src tests docs`
  - `rg -n \"background_jobs|job\" -S src/ui src/pages tests`
  - `rg -n \"FR-015\\b\" docs/specs/requirements.md docs/specs/traceability.md`
- Read in parallel:
  - `src/config/settings.py` (DatabaseConfig)
  - `src/ui/background_jobs.py`
  - `docs/specs/spec-039-operational-metadata-sqlite-wal.md`

**MCP resources first (when available):**

- `functions.list_mcp_resources` → read any local “sqlite/migrations/persistence” resources.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc` (rare for this package)
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain` (rare for this package)

**API verification (Context7, only when needed):**

- `functions.mcp__context7__resolve-library-id` → `langchain`, `langgraph` only if you end up wiring ops DB into agent graph tooling (prefer not).

**Security gate (required):**

- Run `functions.mcp__zen__secaudit` scoped to:
  - path validation (DB under data_dir by default)
  - metadata-only storage (no raw prompts/doc text)
  - bounded retries on SQLITE_BUSY (no unbounded loops)

**Review gate (recommended):**

- Run `functions.mcp__zen__codereview` after tests pass (persistence changes are easy to get subtly wrong).

**opensrc (optional):**

Use only if sqlite/locking behavior is surprising. Otherwise prefer official sqlite docs.

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

**Primary Task:** Implement a local-only operational metadata store using **SQLite WAL** for job lifecycle tracking and snapshot events, with migrations and offline tests.

**Why now:** Background jobs and restore flows require durable, queryable operational state across restarts. Current code has SQLite config knobs but no implementation, causing spec/requirements drift.

**Definition of Done (DoD):**

- `src/persistence/ops_db.py` exists and provides WAL connections + migrations + minimal job/snapshot APIs.
- Background job orchestration writes best-effort lifecycle updates to ops DB (fail-open, never blocks the UI indefinitely).
- Unit + integration tests exist and run offline.
- `docs/specs/traceability.md` contains an `FR-015` row mapping code/tests (and any touched rows updated).

**In-scope modules/files (initial):**

- `src/persistence/ops_db.py` (new)
- `src/persistence/migrations/ops_db/0001_init.sql` (new)
- `src/ui/background_jobs.py` (integration point)
- `tests/unit/persistence/test_ops_db_migrations.py` (new)
- `tests/unit/persistence/test_ops_db_jobs.py` (new)
- `docs/developers/adrs/ADR-055-operational-metadata-sqlite-wal.md`
- `docs/specs/spec-039-operational-metadata-sqlite-wal.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Replacing the analytics DB (DuckDB).
- Storing raw prompts or document text in SQLite.

---

### HARD RULES (EXECUTION)

#### 1) Python + Packaging

- Python baseline is **3.13.11** (supported: 3.11–3.13; respect `pyproject.toml`).
- Use **uv only**:
  - install/sync: `uv sync`
  - run tools: `uv run <cmd>`

#### 2) SQLite correctness

- WAL must be enabled on every connection (`PRAGMA journal_mode=WAL;`).
- Use `PRAGMA user_version` for migrations.
- Use bounded retries/timeouts for `SQLITE_BUSY` (no unbounded retry loops).
- Do not share a single sqlite3 connection across threads unless you can prove safety; prefer “one connection per operation” for simplicity.

#### 3) Security & privacy

- Metadata-only: no raw prompts/doc text in DB.
- Validate DB path; keep under `settings.data_dir` by default.
- Never log secrets.

#### 4) Quality gates

Run:

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
uv run python scripts/run_tests.py --fast
```

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

You MUST produce a plan and keep exactly one step “in_progress” at a time.

0. [ ] Read ADR/SPEC/requirements/RTM and restate DoD in your plan.
1. [ ] Implement `src/persistence/ops_db.py` + migration loader (user_version).
2. [ ] Add `src/persistence/migrations/ops_db/0001_init.sql`.
3. [ ] Integrate best-effort writes from `src/ui/background_jobs.py`.
4. [ ] Add unit + integration tests (temp data_dir; offline).
5. [ ] Update RTM row(s) and run quality gates.

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. Global sqlite connection reused across threads.
2. Unbounded retry loops on sqlite locks.
3. Writing raw prompt/document text into SQLite.
4. Mixing analytics workloads into the ops DB (keep separation).

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement    | Status | Proof / Notes                            |
| -------------- | ------ | ---------------------------------------- |
| **Packaging**  |        | `uv sync` clean                          |
| **Formatting** |        | `ruff format`                            |
| **Lint**       |        | `ruff check` clean                       |
| **Types**      |        | `pyright` clean                          |
| **Tests**      |        | `scripts/run_tests.py --fast` green      |
| **Docs**       |        | ADR/SPEC/RTM updated                     |
| **Security**   |        | WAL + path validation + metadata-only    |
| **Tech Debt**  |        | zero work-marker placeholders introduced |

**EXECUTE UNTIL COMPLETE.**
