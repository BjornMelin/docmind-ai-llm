# Implementation Prompt — Containerization Hardening (Dockerfile + Compose)

Implements `ADR-042` + `SPEC-023`.

Use the `$docker-architect` workflow patterns (multi-stage, non-root, .dockerignore).

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
