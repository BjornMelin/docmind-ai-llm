# Implementation Prompt — Containerization Hardening (Dockerfile + Compose)

Implements `ADR-042` + `SPEC-023`.

Use the `$docker-architect` workflow patterns (multi-stage, non-root, .dockerignore).

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

**Use skill:** `$docker-architect`

Mandatory workflow steps from the skill:

```bash
uv sync
python3 /home/bjorn/.codex/skills/docker-architect/scripts/docker_inventory.py --root .
python3 /home/bjorn/.codex/skills/docker-architect/scripts/docker_audit.py --root .
docker buildx version
```

Skill references to consult (as needed):
- `/home/bjorn/.codex/skills/docker-architect/references/security_hardening.md`
- `/home/bjorn/.codex/skills/docker-architect/references/compose_patterns.md`
- `/home/bjorn/.codex/skills/docker-architect/references/dockerfile_patterns.md`

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Repo truth scan:
  - `ls -la Dockerfile docker-compose.yml || true`
  - `rg -n \"(FROM|USER|EXPOSE|HEALTHCHECK|CMD|ENTRYPOINT)\" Dockerfile docker-compose.yml || true`
  - `rg -n \"DOCMIND_|OLLAMA_|LMSTUDIO_|VLLM_\" docker-compose.yml .env.example docs -S || true`
- Run skill scripts in parallel with reading docker artifacts:
  - `/home/bjorn/.codex/skills/docker-architect/scripts/docker_inventory.py`
  - `/home/bjorn/.codex/skills/docker-architect/scripts/docker_audit.py`

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for Docker/Compose/Streamlit resources (rare but cheap to check).
- `functions.read_mcp_resource` → prefer local docs if present.

**Time-sensitive facts (prefer web tools):**

- Use `functions.mcp__exa__web_search_exa` (or `web.run` when you need citations/dates) for:
  - current `python:3.11-*` tag conventions
  - Streamlit container best practices (ports, healthchecks)

**Long-running docker flows (use `functions.write_stdin`):**

- For `docker compose up` / `docker compose logs -f`, keep sessions alive and stream output via `functions.write_stdin` instead of restarting.

**Security gate (required):**

- Run `functions.mcp__zen__secaudit` scoped to container hardening (secrets, non-root, `.dockerignore`, exposed ports).

**Review gate (recommended):**

- Run `functions.mcp__zen__codereview` for Dockerfile/compose changes (blast radius is large).

**Python-side verification still matters:**

- After container changes, run repo quality gates on host:
  - `uv run ruff format .`
  - `uv run ruff check . --fix`
  - `uv run pyright`
  - `uv run pylint --fail-under=9.5 src/ tests/ scripts/`
  - `uv run python scripts/run_tests.py --fast`

### MCP tool sequence (use when it adds signal)

1. `functions.mcp__zen__planner` → plan Dockerfile/compose changes + smoke checks.
2. Exa search (official sources) for:
   - Python 3.11 base image tags
   - Streamlit container best practices (ports, health checks)
3. Context7 for `uv` and Streamlit entrypoint behavior if uncertain.
4. `functions.mcp__zen__secaudit` → container hardening checklist (non-root, secrets, .dockerignore).

**opensrc (optional):**

Use only if container behavior depends on packaging internals (rare). Prefer repo scripts and `pyproject.toml` truth.

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

**Primary Task:** Fix Dockerfile + docker-compose so containers run correctly and match repo constraints (Python 3.11, `DOCMIND_*` env vars, Streamlit entrypoint `src/app.py`).

**Why now:** Current Docker artifacts are not runnable and violate Python constraint (`>=3.11,<3.12`), blocking any “ship-ready” claim.

**Definition of Done (DoD):**

- `Dockerfile` uses Python 3.11 and launches `streamlit run src/app.py` correctly.
- Docker CMD/ENTRYPOINT is valid (no shell in JSON array bug).
- Container runs as non-root user.
- `.dockerignore` exists and excludes `.env` and large dev artifacts.
- `docker-compose.yml` uses canonical `DOCMIND_*` variables (no legacy names like `OLLAMA_BASE_URL`).
- `docs/specs/traceability.md` includes NFR-PORT-003 row (Planned → Implemented).

**In-scope modules/files (initial):**

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore` (new)
- `docs/developers/adrs/ADR-042-containerization-hardening.md`
- `docs/specs/spec-023-containerization-hardening.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Implementing GPU compose profiles unless already trivial.
- Registry publish workflow.

---

### HARD RULES (EXECUTION)

#### 1) Python + Packaging

- Container runtime must be Python **3.11.x**.
- Use `uv sync --frozen` in container builds.

#### 2) Security

- No secrets baked into the image.
- Non-root runtime user.
- `.env` excluded by default.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Run docker inventory/audit and capture baseline.

   - Run:

     ```bash
     python3 /home/bjorn/.codex/skills/docker-architect/scripts/docker_inventory.py --root .
     python3 /home/bjorn/.codex/skills/docker-architect/scripts/docker_audit.py --root .
     ```

2. [ ] Add `.dockerignore` to exclude secrets and build junk.

3. [ ] Replace Dockerfile with a Python 3.11 multi-stage uv build (non-root runtime).

4. [ ] Update `docker-compose.yml` to canonical `DOCMIND_*` env contract and validate config.

   - Run: `docker compose config`

5. [ ] Build and run container locally (smoke).

   - Run:

     ```bash
     docker build -t docmind:dev .
     docker compose up --build -d
     docker compose logs -f --tail=120 app
     ```

6. [ ] Update RTM row NFR-PORT-003 and verify docs are consistent.

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. Python 3.12+ base images (violates pyproject).
2. JSON `CMD` with embedded shell strings (`[". venv/bin/activate && ..."]`).
3. Missing `.dockerignore`.
4. Root runtime user without justification.
5. Compose env vars that bypass `DOCMIND_*` settings model.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement    | Status | Proof / Notes                    |
| -------------- | ------ | -------------------------------- |
| Docker build   |        | `docker build` succeeds          |
| Compose config |        | `docker compose config` clean    |
| Runtime        |        | app logs show Streamlit started  |
| Security       |        | non-root user; `.env` not copied |
| Docs           |        | ADR/SPEC/RTM updated             |

**EXECUTE UNTIL COMPLETE.**
