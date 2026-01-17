---
prompt: PROMPT-023
title: Containerization Hardening (Dockerfile + Compose)
status: Completed
date: 2026-01-11
version: 1.0
related_adrs: ["ADR-042"]
related_specs: ["SPEC-023"]
---

## Implementation Prompt — Containerization Hardening (Dockerfile + Compose)

Implements `ADR-042` + `SPEC-023`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-042-containerization-hardening.md`
- SPEC: `docs/specs/spec-023-containerization-hardening.md`
- Requirements: `docs/specs/requirements.md` (NFR-PORT-003, NFR-SEC-001)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.docker.com/build/building/best-practices/> — Dockerfile best practices (multi-stage, caching, secrets).
- <https://docs.docker.com/compose/> — Docker Compose overview and workflows.
- <https://docs.docker.com/compose/compose-file/> — Compose file reference (ports, env, healthchecks).
- <https://docs.docker.com/reference/cli/docker/compose/config/> — `docker compose config` validation.
- <https://docs.docker.com/compose/how-tos/gpu-support/> — Compose GPU support (device reservations / prerequisites).
- <https://docs.docker.com/engine/containers/resource_constraints/#gpu> — Docker Engine GPU prerequisites.
- <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html> — NVIDIA Container Toolkit install guide.
- <https://hub.docker.com/_/python> — Official Python image tags and supported variants.
- <https://docs.astral.sh/uv/guides/integration/docker/> — `uv` Docker install guidance (`uv sync --frozen`).
- <https://docs.streamlit.io/deploy/tutorials/docker> — Streamlit Docker deployment tutorial (ports, command).

Use the `$docker-architect` workflow patterns (multi-stage, non-root, .dockerignore).

## Tooling & Skill Strategy (fresh Codex sessions)

**Audience note:** This prompt targets the Codex/MCP execution environment (functions._ tools and `${CODEX_*}` paths). If you are running without Codex, replace those calls with direct shell commands and standard tooling; the skill paths are external to the DocMind repo.

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

**Use skill:** `$docker-architect`

Mandatory workflow steps from the skill:

```bash
uv sync
python3 ${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/docker-architect/scripts/docker_inventory.py --root .
python3 ${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/docker-architect/scripts/docker_audit.py --root .
docker buildx version
```

Skill references to consult (as needed):

- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/docker-architect/references/security_hardening.md`
- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/docker-architect/references/compose_patterns.md`
- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/docker-architect/references/dockerfile_patterns.md`

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Repo truth scan:
  - `ls -la Dockerfile docker-compose.yml || true`
  - `rg -n \"(FROM|USER|EXPOSE|HEALTHCHECK|CMD|ENTRYPOINT)\" Dockerfile docker-compose.yml || true`
  - `rg -n \"DOCMIND_|OLLAMA_|LMSTUDIO_|VLLM_\" docker-compose.yml .env.example docs -S || true`
- Run skill scripts in parallel with reading docker artifacts:
  - `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/docker-architect/scripts/docker_inventory.py`
  - `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/docker-architect/scripts/docker_audit.py`

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for Docker/Compose/Streamlit resources (rare but cheap to check).
- `functions.read_mcp_resource` → prefer local docs if present.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc`
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain`

**Time-sensitive facts (prefer web tools):**

- Use `functions.mcp__exa__web_search_exa` (or `web.run` when you need citations/dates) for:
  - current `python:3.13.11-*` tag conventions
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
  - `uv run python scripts/run_tests.py --fast`

### MCP tool sequence (use when it adds signal)

1. `functions.mcp__zen__planner` → plan Dockerfile/compose changes + smoke checks.
2. Exa search (official sources) for:
   - Python 3.13.11 base image tags
   - Streamlit container best practices (ports, health checks)
3. Context7 for `uv` and Streamlit entrypoint behavior if uncertain.
4. `functions.mcp__zen__secaudit` → container hardening checklist (non-root, secrets, .dockerignore).

**opensrc (optional):**

Use only if container behavior depends on packaging internals (rare). Prefer repo scripts and `pyproject.toml` truth.

## Implementation Executor Template (DocMind / Python)

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

**Primary Task:** Fix Dockerfile + docker-compose so containers run correctly and match repo constraints (Python 3.13.11 baseline, `DOCMIND_*` env vars, Streamlit entrypoint `src/app.py`).

**Why now:** Current Docker artifacts are not runnable and violate Python constraint (`>=3.13,<3.14`), blocking any “ship-ready” claim.

**Definition of Done (DoD):**

- `Dockerfile` uses Python 3.13.11 and launches `streamlit run src/app.py` correctly.
- Docker CMD/ENTRYPOINT is valid (no shell in JSON array bug).
- Container runs as non-root user.
- `.dockerignore` exists and excludes `.env` and large dev artifacts.
- `docker-compose.yml` uses canonical `DOCMIND_*` variables (no provider/legacy env var names).
- `docker-compose.yml` provides a `gpu` profile that runs **Ollama** with GPU access on an internal network (no host port publish by default). Note: since `ollama` resolves to a private RFC1918 address on the compose network, DocMind must either set `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true` for this deployment or use a strict localhost architecture (shared network namespace) so DocMind connects to `http://localhost:11434` with remote endpoints still disabled.
- `docker-compose.prod.yml` provides `read_only: true` and `tmpfs` for `/tmp`.
- `docs/specs/traceability.md` includes NFR-PORT-003 row (Planned → Implemented).

**In-scope modules/files (initial):**

- `Dockerfile`
- `docker-compose.yml`
- `docker-compose.prod.yml`
- `.dockerignore` (new)
- `docs/developers/adrs/ADR-042-containerization-hardening.md`
- `docs/specs/spec-023-containerization-hardening.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Bundling multiple GPU inference backends in compose (vLLM/LM Studio). DocMind must support them via configuration, but only Ollama is shipped as a compose GPU service.
- Registry publish workflow.

---

### Hard Rules (Execution)

#### 1) Python + Packaging

- Container runtime baseline is Python **3.13.11**.
- Use `uv sync --frozen` in container builds.

#### 2) Style, Types, and Lint (host-side)

After container changes, the repo must still pass:

- `uv run ruff format .`
- `uv run ruff check . --fix`
- `uv run pyright`
- `uv run python scripts/run_tests.py --fast` (then `uv run python scripts/run_tests.py` before marking complete)

#### 2) Security

- No secrets baked into the image.
- Non-root runtime user.
- `.env` excluded by default.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

1. [ ] Run docker inventory/audit and capture baseline.

   - Run:

     ```bash
     python3 ${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/docker-architect/scripts/docker_inventory.py --root .
     python3 ${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/docker-architect/scripts/docker_audit.py --root .
     ```

2. [ ] Add `.dockerignore` to exclude secrets and build junk.

3. [ ] Replace Dockerfile with a Python 3.13.11 multi-stage uv build (non-root runtime).

4. [ ] Update `docker-compose.yml` to canonical `DOCMIND_*` env contract and validate config.

   - Run: `docker compose config`
   - Run: `docker compose -f docker-compose.yml -f docker-compose.prod.yml config`

5. [ ] Build and run container locally (smoke).

   - Run:

     ```bash
     docker build -t docmind:dev .
     docker compose up --build -d
     docker compose logs -f --tail=120 app
     ```

6. [ ] Update RTM row NFR-PORT-003 and verify docs are consistent.
7. [ ] Add Docker static validation in CI (hadolint + CMD/ENTRYPOINT checks + `.dockerignore` + non-root/user checks + `DOCMIND_*` prefix scan), or file a tracking ADR/issue if out of scope for this WP.

---

### Anti-Pattern Kill List (Immediate Deletion/Rewrite)

1. Python base images that drift from the repo’s pinned runtime baseline (must be Python 3.13.11).
2. JSON `CMD` with embedded shell strings (`[". venv/bin/activate && …"]`).
3. Missing `.dockerignore`.
4. Root runtime user without justification.
5. Compose env vars that bypass `DOCMIND_*` settings model.

**Automation note:** Add a lightweight static check in CI to enforce these rules (Dockerfile CMD/ENTRYPOINT validation, `.dockerignore` presence, non-root runtime user, and env var prefix scanning). Track this as a follow-up ADR or GitHub issue if it cannot ship with this work package.

---

### MCP TOOL STRATEGY (FOR IMPLEMENTATION RUN)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → implementation plan (Dockerfile/compose + smoke checks).
2. `functions.mcp__exa__web_search_exa` / `functions.mcp__exa__crawling_exa` → official base-image/tag guidance and Streamlit container docs.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → API details (only if you need `uv`/Streamlit invocation specifics).
4. `functions.mcp__gh_grep__searchGitHub` → production patterns for multi-stage Python+uv images (optional).
5. `functions.mcp__zen__analyze` → only if you discover complex compose/service topology.
6. `functions.mcp__zen__codereview` → post-implementation review (blast radius is large).
7. `functions.mcp__zen__secaudit` → required security audit (non-root, secrets, `.dockerignore`, ports).

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel reads.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement           | Status | Proof / Notes                                                                      |
| --------------------- | ------ | ---------------------------------------------------------------------------------- |
| **Docker Build**      |        | `docker build -t docmind:dev .` succeeds                                           |
| **Compose Config**    |        | `docker compose config` clean                                                      |
| **Container Runtime** |        | Streamlit starts; app accessible on expected port                                  |
| **Packaging**         |        | `uv sync` clean (host)                                                             |
| **Formatting**        |        | `uv run ruff format .`                                                             |
| **Lint**              |        | `uv run ruff check .` clean                                                        |
| **Types**             |        | `uv run pyright` clean                                                             |
| **Tests**             |        | `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py` |
| **Docs**              |        | ADR/SPEC/RTM updated                                                               |
| **Security**          |        | non-root user; `.env` not copied; no secrets baked                                 |
| **Tech Debt**         |        | zero work-marker placeholders introduced                                           |
| **Performance**       |        | image build is reproducible; no unnecessary layers                                 |

**EXECUTE UNTIL COMPLETE.**
