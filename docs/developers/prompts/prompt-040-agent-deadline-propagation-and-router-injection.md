# Implementation Prompt — Agent Deadline Propagation + Router Injection

Implements `ADR-056` + `SPEC-040`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-056-agent-deadline-propagation-and-router-injection.md`
- SPEC: `docs/specs/spec-040-agent-deadline-propagation-and-router-injection.md`
- Requirements: `docs/specs/requirements.md` (FR-014, FR-029)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://langchain-ai.github.io/langgraph/how-tos/streaming/> — LangGraph streaming patterns (sync vs. async).
- <https://langchain-5e9cc07a.mintlify.app/oss/python/langgraph/interrupts> — Human-in-the-loop / interrupts (control-flow, not hard cancellation).
- <https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig> — RunnableConfig / execution config surface.
- <https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/> — `ChatOpenAI(timeout=...)` reference.

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

This is agent/orchestration work. No dedicated skill exists for LangGraph, but:

- Use `$streamlit-master-architect` if you touch Streamlit pages/AppTest to validate UX changes.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Confirm current wiring and config drift:
  - `rg -n \"enable_deadline_propagation|enable_router_injection\" -S src`
  - `rg -n \"get_retrieval_tools\\(\" -S src/agents/registry/tool_registry.py`
  - `rg -n \"router_tool\\b|retrieve_documents\\b\" -S src/agents`
  - `rg -n \"compiled_graph\\.stream\\(\" -S src/agents/coordinator.py`
- Read in parallel:
  - `src/agents/coordinator.py`
  - `src/agents/models.py`
  - `src/agents/registry/tool_registry.py`
  - `src/agents/tools/retrieval.py`
  - `src/config/langchain_factory.py`
  - `src/config/llm_factory.py`

**MCP resources first (when available):**

- `functions.list_mcp_resources` → read any LangGraph/LangChain/timeout resources if present.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain` (interrupts, persistence/config, runnable timeouts)
- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc` (only if router injection uses LlamaIndex engines directly)
- OpenAI API docs: `functions.mcp__openaiDeveloperDocs__search_openai_docs` → `functions.mcp__openaiDeveloperDocs__fetch_openai_doc` (only if touching OpenAI client semantics)

**API verification (Context7, only when uncertain):**

- `functions.mcp__context7__resolve-library-id` → `langgraph`, `langchain`
- `functions.mcp__context7__query-docs` → confirm how tools access config/state and any timeout-related fields.

**Real-world patterns (only if blocked):**

- `functions.mcp__gh_grep__searchGitHub` for “deadline” or “timeout” patterns in LangGraph tool wrappers.

**Security gate (required):**

- Run `functions.mcp__zen__secaudit` with scope:
  - no new unsafe logging of raw prompts/content
  - timeouts are bounded; no unbounded retry loops
  - router injection does not bypass allowlist policy (still local-only)

**Review gate (recommended):**

- Run `functions.mcp__zen__codereview` after tests pass (agent wiring regressions are subtle).

**opensrc (optional):**

Use only if you must confirm LangGraph/LangChain internals for timeout behavior. Prefer official docs + runtime introspection.

## Implementation Executor Template (DocMind / Python)

### You Are

You are an autonomous implementation agent for the **DocMind AI LLM** repository.

You will implement the feature described below end-to-end, including:

- code changes
- tests
- documentation updates (ADR/SPEC/RTM)
- deletion of dead code and removal of legacy/backcompat shims within scope

You must keep changes minimal, library-first, and maintainable.

---

### Feature Context (Filled)

**Primary Task:** Implement cooperative agent deadline propagation and router injection:

- make `enable_deadline_propagation` and `enable_router_injection` functional
- restore a consistent retrieval tool contract so sources/documents are available to synthesis/validation

**Why now:** Current agent workflow enforces timeouts only between streamed states and has retrieval tool wiring drift, which can produce missing sources and make timeouts ineffective. These are release-blocking correctness/perf issues.

**Definition of Done (DoD):**

- `MultiAgentState` includes deadline fields and coordinator seeds them when enabled.
- LLM/Qdrant timeouts are capped to `agents.decision_timeout` when deadline propagation is enabled.
- DefaultToolRegistry wires retrieval to `retrieve_documents` (not a router-only response tool).
- `retrieve_documents` supports router injection when enabled and router_engine is present, returning structured documents/sources.
- Tests added/updated; RTM includes `FR-029` mapping.

**In-scope modules/files (initial):**

- `src/agents/models.py`
- `src/agents/coordinator.py`
- `src/agents/registry/tool_registry.py`
- `src/agents/tools/retrieval.py`
- `src/config/langchain_factory.py`
- `src/config/llm_factory.py`
- `tests/unit/agents/*` (new/updated)
- `tests/unit/config/*` (new/updated)
- `docs/developers/adrs/ADR-056-agent-deadline-propagation-and-router-injection.md`
- `docs/specs/spec-040-agent-deadline-propagation-and-router-injection.md`
- `docs/specs/requirements.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Full async `astream` cancellation refactor for Streamlit.
- Changing Qdrant schema or retrieval algorithms.

---

### Hard Rules (Execution)

#### 1) Python + Packaging

- Python version must remain **3.11.x** (respect `pyproject.toml`).
- Use **uv only**:
  - install/sync: `uv sync`
  - run tools: `uv run <cmd>`

#### 2) Deadline semantics

- Use an **absolute monotonic deadline** (e.g., `time.monotonic()` seconds).
- Never introduce global mutable deadline state (must be per-run/state-scoped).
- Fail open: if deadline data is missing/corrupt, default to existing behavior.

#### 3) Security & privacy

- Never log raw prompts or document text while debugging timeouts.
- Do not add new network endpoints; keep allowlist enforcement intact.

#### 4) Quality gates

Run:

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
uv run python scripts/run_tests.py --fast
```

---

### Step-by-Step Execution Plan (Filled)

You MUST produce a plan and keep exactly one step “in_progress” at a time.

0. [ ] Read ADR/SPEC/requirements/RTM and restate DoD in your plan.
1. [ ] Fix retrieval tool wiring in `DefaultToolRegistry` (structured retrieval payload contract).
2. [ ] Implement router injection path inside `retrieve_documents` when enabled and router_engine present.
3. [ ] Add deadline fields to state + seed deadline_ts when enabled.
4. [ ] Cap timeouts (ChatOpenAI + LlamaIndex LLM + Qdrant) when deadline propagation enabled.
5. [ ] Add unit/integration tests and update RTM (FR-029).
6. [ ] Run quality gates and security+review gates.

---

### Anti-Pattern Kill List (Immediate Deletion/Rewrite)

1. Global flags for cancellation shared across users/sessions.
2. “Hard-killing” threads/futures to cancel work.
3. Unbounded retries when the deadline is exceeded.
4. Tool outputs that drop source nodes/documents (breaks RAG provenance).

---

### Final Verification Checklist (Must Complete)

| Requirement    | Status | Proof / Notes                                              |
| -------------- | ------ | ---------------------------------------------------------- |
| **Packaging**  |        | `uv sync` clean                                            |
| **Formatting** |        | `ruff format`                                              |
| **Lint**       |        | `ruff check` clean                                         |
| **Types**      |        | `pyright` clean                                            |
| **Tests**      |        | `scripts/run_tests.py --fast` green                        |
| **Docs**       |        | ADR/SPEC/RTM updated                                       |
| **Security**   |        | no raw-content logs; bounded timeouts; allowlist preserved |
| **Tech Debt**  |        | zero TODO/FIXME introduced                                 |

**Execute Until Complete.**
