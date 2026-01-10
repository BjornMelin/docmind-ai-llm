# Implementation Prompt — Semantic Response Cache (Qdrant-backed, Guardrailed)

Implements `ADR-035` + `SPEC-038`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-035-semantic-cache-qdrant.md`
- SPEC: `docs/specs/spec-038-semantic-cache-qdrant.md`
- RTM: `docs/specs/traceability.md`
- Requirements: `docs/specs/requirements.md`

## Official docs (research during implementation)

- <https://qdrant.tech/documentation/concepts/collections/> — Create collection schema (vectors + payload indexes).
- <https://qdrant.tech/documentation/concepts/payload/> — Metadata filtering; strict filters are required for correctness.
- <https://qdrant.tech/documentation/concepts/snapshots/> — Cache persistence and backup implications.
- <https://docs.llamaindex.ai/en/stable/examples/llm/anthropic_prompt_caching/> — Provider-managed prompt caching (reference only).

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

**opensrc guidance:**

- Check `opensrc/sources.json` before fetching.
- Fetch dependency source only if needed for edge cases:
  - `npx opensrc pypi:qdrant-client@<installed-version> --modify=false`
  - `npx opensrc pypi:llama-index@<pinned-version> --modify=false`

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel` for independent work):**

- Locate the LLM call boundary where caching can be inserted:
  - `rg -n "Settings\\.llm|\\.complete\\(|\\.achat\\(|\\.chat\\(|llm\\." src/agents src/pages src/config -S`
- Confirm semantic cache config fields and current provider enum:
  - `rg -n "class SemanticCacheConfig" src/config/settings.py`
- Confirm Qdrant client usage patterns already in repo:
  - `rg -n "QdrantClient\\(" src -S`
- Verify qdrant-client is available and inspect snapshot methods:
  - `uv run python -c \"import inspect; from qdrant_client import QdrantClient; c=QdrantClient(location=':memory:'); print([m for m in dir(c) if 'snapshot' in m])\"`

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for Qdrant/LlamaIndex resources.
- `functions.read_mcp_resource` → read relevant local docs/indexes before web search.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc`
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain` (only if cache integration touches LangGraph tools/state)

**API verification (Context7, only when uncertain):**

- `functions.mcp__context7__resolve-library-id` → `qdrant-client`, `llama-index`
- `functions.mcp__context7__query-docs` → confirm the exact Qdrant search/upsert APIs and any LlamaIndex LLM wrapper interfaces you touch.

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

**Primary Task:** Implement a Qdrant-backed semantic response cache with:

- exact-match fast path (`prompt_key`)
- optional semantic similarity search (threshold + strict filters)
- invalidation via `corpus_hash` + `config_hash`
- TTL and bounded size controls
- fail-open behavior

**Why now:** The config surface for semantic caching exists, and performance optimizations are part of a “finished” offline-first app. We want caching without adding new services or heavy native deps.

**Definition of Done (DoD):**

- `SemanticCacheConfig.provider` supports `qdrant` and is documented.
- New cache module provides a small protocol + a Qdrant implementation.
- Cache is integrated at a single, well-defined boundary in the answer generation path.
- Cache never stores raw prompts; it may store response payload (bounded) or an encrypted blob reference.
- Unit/integration tests run offline using `QdrantClient(location=':memory:')`.
- RTM updated with `FR-026` mapping to code/tests.

**In-scope modules/files (initial):**

- `src/utils/semantic_cache.py` (new)
- `src/config/settings.py` (extend `SemanticCacheConfig` for qdrant provider + optional knobs)
- One integration point (choose the narrowest viable boundary):
  - `src/agents/coordinator.py` (preferred) or
  - `src/config/llm_factory.py` (if implementing an LLM wrapper)
- `tests/unit/utils/test_semantic_cache.py` (new)
- `tests/integration/test_semantic_cache_integration.py` (new; uses MockLLM)
- `docs/specs/spec-038-semantic-cache-qdrant.md`
- `docs/specs/requirements.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- LiteLLM proxy caching.
- Adding GPTCache/FAISS dependencies as required runtime deps (may be a follow-up provider).

---

### HARD RULES (EXECUTION)

#### 1) Python + Packaging

- Python version must remain **3.11.x** (respect `pyproject.toml`).
- Use **uv only**:
  - install/sync: `uv sync`
  - run tools: `uv run <cmd>`

#### 2) Style, Types, and Lint

Your code must pass:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run pyright`

#### 3) Security & Privacy

- Never store raw prompts in any persisted cache.
- Never log secrets.
- Invalidate cache on corpus/config change; do not cross-hit across corpora.

#### 4) Offline-first posture

- Cache must be fully local and default-off.
- No new remote endpoints are introduced by this feature.

---

### STEP-BY-STEP EXECUTION PLAN

You MUST produce a plan and keep exactly one step “in_progress” at a time.

1. [ ] Read ADR/SPEC and locate the best cache insertion point in the LLM call path.
2. [ ] Implement `src/utils/semantic_cache.py`:
   - protocol + Qdrant implementation
   - canonicalization + prompt_key hashing
   - strict filter construction
3. [ ] Extend `SemanticCacheConfig`:
   - provider `qdrant`
   - optional `collection_name`, `allow_semantic_for_templates`
4. [ ] Integrate cache at the chosen boundary (single place).
5. [ ] Add tests:
   - unit tests for keying + filters + TTL + threshold behavior
   - integration test proving cache hit bypasses MockLLM generation
6. [ ] Update docs + RTM (`FR-026`).
7. [ ] Run quality gates.

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

1. Storing raw prompts or full retrieved context text in the cache payload.
2. Semantic hits without strict filters (model/template/params/corpus hash).
3. Cache returning answers across different corpora/configs.
4. Broad `except Exception` swallowing without telemetry/logging (cache must fail-open but be observable).
5. Unbounded cache growth without TTL/pruning.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement       | Status | Proof / Notes                              |
| ----------------- | ------ | ------------------------------------------ |
| **Formatting**    |        | `uv run ruff format .`                     |
| **Lint**          |        | `uv run ruff check .`                      |
| **Types**         |        | `uv run pyright`                           |
| **Tests**         |        | `uv run python -m pytest -q`               |
| **Docs**          |        | ADR/SPEC/RTM updated                       |
| **Security**      |        | no raw prompts stored; strict invalidation |
| **Offline-first** |        | default-off; no new egress                 |

**EXECUTE UNTIL COMPLETE.**
