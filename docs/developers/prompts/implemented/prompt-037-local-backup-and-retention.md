---
prompt: PROMPT-037
title: Local Backup & Retention (Snapshots, Cache, Analytics, Qdrant)
status: Completed
date: 2026-01-17
version: 1.0
related_adrs: ["ADR-033"]
related_specs: ["SPEC-037"]
---

Implements `ADR-033` + `SPEC-037`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-033-local-backup-and-retention.md`
- SPEC: `docs/specs/spec-037-local-backup-and-retention.md`
- RTM: `docs/specs/traceability.md`
- Requirements: `docs/specs/requirements.md`

## Official docs (research during implementation)

- <https://docs.python.org/3/library/shutil.html> — Copy primitives (`copy2`, `copytree`) for local-only backups.
- <https://qdrant.tech/documentation/concepts/snapshots/> — Qdrant snapshot API semantics and restore.
- <https://qdrant.tech/documentation/concepts/collections/> — Collection naming and persistence considerations.
- <https://docs.streamlit.io/develop/api-reference/widgets/st.download_button> — Optional UI “download backup manifest” affordance.

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

**Optional skills (only if used):**

- `$streamlit-master-architect` — only if adding a UI “Create backup now” button.

**opensrc guidance:**

- Check `opensrc/sources.json` first; fetch qdrant-client sources only if you need to confirm snapshot endpoints/edge cases.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel` for independent work):**

- Find current data layout usage:
  - `rg -n "data_dir|cache_dir|analytics_db_path|uploads" src/config/settings.py src/pages -S`
- Confirm qdrant-client snapshot APIs (requires `uv sync` first):
  - `uv run python -c "import inspect; from qdrant_client import QdrantClient; c=QdrantClient(location=':memory:'); print([m for m in dir(c) if 'snapshot' in m])"`
  - _Note: Skip if qdrant-client import fails; snapshot APIs are optional for this work._
- Locate existing snapshot retention logic:
  - `rg -n "retention|gc_grace_seconds|recover_snapshots" src/persistence/snapshot.py -S`

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for Qdrant/python resources.
- `functions.read_mcp_resource` → read relevant local docs/indexes before web search.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc` (rare for this package)
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain` (rare for this package)

**API verification (Context7, only when uncertain):**

- `functions.mcp__context7__resolve-library-id` → `qdrant-client`
- `functions.mcp__context7__query-docs` → confirm snapshot API usage and return types.

**Time-sensitive facts (use web tools):**

- Prefer `functions.mcp__exa__web_search_exa` for discovery; use `web.run` if you need citations or dates.

---

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

### FEATURE CONTEXT

**Primary Task:** Implement a manual, local-only backup mechanism with rotation, including:

- cache DB
- snapshots (`data/storage/`)
- optional uploads/analytics/logs
- Qdrant collection snapshot (server-side snapshot + download via REST)

**Why now:** Backups are a core operational feature for a polished offline-first app; users need a predictable way to preserve and restore local state.

**Definition of Done (DoD):**

- `scripts/backup.py` implements `create-backup` and `prune-backups` with clear CLI flags.
- Backups are stored under `data/backups/backup_<timestamp>/` by default, rotated to `settings.backup_keep_last`.
- Qdrant snapshot step:
  - creates snapshot for `settings.database.qdrant_collection`
  - downloads snapshot file into backup dir
  - writes a small manifest JSON describing what was captured
- Unit and integration tests cover rotation, manifests, and Qdrant snapshot behavior (mocked/offline).
- Docs updated and RTM includes `FR-027` mapping.

**ADR-051 Integration Status:**
ADR-051 (Documents Snapshot Service Boundary) is currently **Proposed**. Until it is accepted and `src/persistence/snapshot_service.py` exists, implement Qdrant snapshots using `qdrant_client` directly. Add a `# refactor-marker: migrate to snapshot_service.create_snapshot()` comment in the backup implementation (e.g., `scripts/backup.py`) where Qdrant snapshots are created to signal the future refactor.

**In-scope modules/files (initial):**

- `scripts/backup.py`
- `src/persistence/backup_service.py`
- `src/config/settings.py` (only if new knobs required)
- `tests/unit/scripts/test_backup_rotation.py` (new)
- `tests/unit/scripts/test_backup_manifest.py` (new)
- `docs/specs/spec-037-local-backup-and-retention.md`
- `docs/specs/requirements.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Remote backup targets (S3, Google Drive).
- Background schedulers/cron integration.

---

### HARD RULES (EXECUTION)

#### 1) Python + Packaging

- Python baseline is **3.13.11** (Python 3.13-only; respect `pyproject.toml`).
- Use **uv only**:
  - install/sync: `uv sync`
  - run tools: `uv run <cmd>`

#### 2) Style, Types, and Lint

Your code must pass:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run pyright`

#### 3) Security & Offline-first

- Validate filesystem paths (no traversal, no symlink escape).
- Never log secrets (including `.env` contents).
- Keep remote endpoints blocked by default; backup operates locally.

---

### STEP-BY-STEP EXECUTION PLAN

You MUST produce a plan and keep exactly one step “in_progress” at a time.

1. [ ] Read ADR/SPEC/RTM and confirm data locations and existing snapshot retention code.
2. [ ] Implement `scripts/backup.py`:
   - create backup dir
   - copy local artifacts
   - create + download Qdrant snapshot via REST
   - rotate backups
3. [ ] Add tests (rotation, manifest, Qdrant snapshot mocked).
4. [ ] Update docs + RTM (`FR-027`).
5. [ ] Run quality gates.

**Commands (required):**

```bash
uv sync
uv run ruff format .
uv run ruff check .
uv run pyright
uv run python -m pytest -q
```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. Writing backups to unsafe/unvalidated paths.
2. Copying raw Qdrant storage directories while Qdrant is running (prefer server snapshot).
3. Silent exception swallowing (backup must fail loudly and not prune on partial failures).
4. Background threads calling Streamlit APIs (if UI button is added).

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement       | Status | Proof / Notes                      |
| ----------------- | ------ | ---------------------------------- |
| **Formatting**    |        | `uv run ruff format .`             |
| **Lint**          |        | `uv run ruff check .`              |
| **Types**         |        | `uv run pyright`                   |
| **Tests**         |        | `uv run python -m pytest -q`       |
| **Docs**          |        | ADR/SPEC/RTM updated               |
| **Security**      |        | path validation; no secret logs    |
| **Offline-first** |        | no new network surfaces introduced |

**EXECUTE UNTIL COMPLETE.**
