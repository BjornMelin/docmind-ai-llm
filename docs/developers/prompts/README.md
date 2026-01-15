# DocMind — Implementation Prompts

This directory contains **fully-atomic, copy/paste implementation prompts** intended for a coding agent to execute work packages end-to-end (code + tests + docs + RTM).

Each prompt is self-contained and embeds DocMind’s “Implementation Executor Template” filled with feature-specific details and quality gates.

## How to use

1. Pick the next work package prompt below.
2. Paste the entire prompt into your coding agent.
3. Run the required quality gates in the prompt before marking the package complete.
4. Repeat until the release readiness prompt is satisfied.

## Prompts (recommended order)

1. `docs/developers/prompts/prompt-021-release-readiness-v1.md`
2. `docs/developers/prompts/prompt-031-config-discipline-env-bridges.md`
3. `docs/developers/prompts/prompt-040-agent-deadline-propagation-and-router-injection.md`
4. `docs/developers/prompts/prompt-032-documents-snapshot-service-boundary.md`
5. `docs/developers/prompts/prompt-039-operational-metadata-sqlite-wal.md`
6. `docs/developers/prompts/prompt-033-background-ingestion-jobs.md`
7. `docs/developers/prompts/prompt-036-document-analysis-modes.md`
8. `docs/developers/prompts/prompt-038-semantic-cache-qdrant.md`
9. `docs/developers/prompts/prompt-037-local-backup-and-retention.md`
10. `docs/developers/prompts/prompt-028-safe-logging-no-pii-redactor.md`
11. `docs/developers/prompts/prompt-034-analytics-page-hardening.md`
12. `docs/developers/prompts/prompt-030-multimodal-helper-cleanup.md`
13. `docs/developers/prompts/prompt-029-docs-consistency-pass.md`

## Implemented prompts

- `docs/developers/prompts/implemented/prompt-022-settings-ui-hardening.md`
- `docs/developers/prompts/implemented/prompt-023-containerization-hardening.md`
- `docs/developers/prompts/implemented/prompt-025-keyword-tool-sparse-only.md`
- `docs/developers/prompts/implemented/prompt-026-ingestion-api-facade.md`
- `docs/developers/prompts/implemented/prompt-027-remove-legacy-main-entrypoint.md`
- `docs/developers/prompts/implemented/prompt-041-chat-persistence-langgraph-sqlite-hybrid-memory.md`
- `docs/developers/prompts/implemented/prompt-042-multimodal-ingestion.md`
- `docs/developers/prompts/implemented/prompt-043-hybrid-retrieval-logic.md`
- `docs/developers/prompts/implemented/prompt-044-ui-and-persistence.md`

## Superseded prompts (do not implement)

- `docs/developers/prompts/superseded/prompt-024-chat-persistence-simplechatstore.md`
- `docs/developers/prompts/superseded/prompt-035-config-surface-pruning-unused-knobs.md`

## Codex tool + skill usage (fresh sessions)

Before starting any prompt in a fresh Codex TUI session:

1. Read the tool inventory: `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.
2. Use repo-truth first:
   - Local search: `rg`
   - Dependency internals: `opensrc/` (see `AGENTS.md`)
3. For uncertain APIs/best-practices:
   - Context7: `functions.mcp__context7__resolve-library-id` → `functions.mcp__context7__query-docs`
   - Exa: `functions.mcp__exa__web_search_exa` → `functions.mcp__exa__crawling_exa`
   - GitHub patterns: `functions.mcp__gh_grep__searchGitHub`
4. For non-trivial changes, run planning/review gates:
   - `functions.mcp__zen__planner` (plan)
   - `functions.mcp__zen__secaudit` (security)
   - `functions.mcp__zen__codereview` (quality gate)

## Setup (portable paths)

Set environment variables so prompts remain portable across machines and CI:

```bash
export CODEX_PROMPT_LIBRARY="${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}"
export CODEX_SKILLS_HOME="${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}"
```

Fallback: If you do not have the Codex skill library installed, copy required skill files into a repo-local directory (e.g., `docs/developers/skills/`) and update the prompt paths accordingly before running.

## Tool Inventory (required + usage guidance)

This section is the *minimum* expected tool usage guidance for any prompt execution.

### Shell/files/editing

- `functions.exec_command` — run discovery/build/test commands; prefer `rg` for search and keep commands readable.
- `functions.write_stdin` — interact with long-running commands (dev server, `docker compose up`, REPL) without restarting them.
- `functions.apply_patch` — small, precise edits with clear diffs; avoid for lockfiles/generated outputs or bulk rewrites.
- `multi_tool_use.parallel` — run independent tool calls concurrently (multiple searches, docs lookups, and shell reads).
- `functions.view_image` — attach screenshots/diagrams for UI verification or debugging (when available).

### Planning/progress

- `functions.update_plan` — track multi-step changes; keep exactly one step `in_progress`.

### Zen (analysis/review)

- `functions.mcp__zen__analyze` — architecture/performance/system boundary analysis before non-trivial refactors.
- `functions.mcp__zen__codereview` — structured code review as a quality gate on significant changes.
- `functions.mcp__zen__secaudit` — security audit for auth, path handling, secrets, logging/PII, new egress surfaces.
- `functions.mcp__zen__consensus` — major design/library selection only (must hit ≥9.0/10.0).
- `functions.mcp__zen__listmodels` — discover available models before running consensus.
- `functions.mcp__zen__version` — diagnose tool availability/config mismatches.

### Docs/API reference

- `functions.mcp__context7__resolve-library-id` — map library name → Context7 id.
- `functions.mcp__context7__query-docs` — authoritative API references/snippets once you have the id.

### LlamaIndex docs (MCP)

- `functions.mcp__llama_index_docs__search_docs` — ranked search across LlamaIndex docs (best for concepts).
- `functions.mcp__llama_index_docs__grep_docs` — exact/regex search (best for class/function names).
- `functions.mcp__llama_index_docs__read_doc` — read a full doc page once you know its path.

### LangChain/LangGraph docs (MCP)

- `functions.mcp__langchain-docs__SearchDocsByLangChain` — search LangChain/LangGraph docs (best for LangGraph persistence, interrupts, state APIs).

### OpenAI developer docs (MCP)

Use only when the work package touches the OpenAI API/client semantics:

- `functions.mcp__openaiDeveloperDocs__search_openai_docs` — search OpenAI platform docs.
- `functions.mcp__openaiDeveloperDocs__fetch_openai_doc` — fetch exact markdown for a page/section.
- `functions.mcp__openaiDeveloperDocs__get_openapi_spec` — inspect endpoint schemas/code samples.

### Web

- `web.run` — general browsing/search with citations when info is time-sensitive or uncertain.
- `functions.mcp__exa__web_search_exa` — fast discovery search.
- `functions.mcp__exa__deep_search_exa` — targeted deep research for niche/precise questions.
- `functions.mcp__exa__crawling_exa` — extract full content from a known URL.

### Real-world code examples

- `functions.mcp__gh_grep__searchGitHub` — grep across GitHub repos for production patterns of an API/tool.

### MCP resources

- `functions.list_mcp_resources` — discover locally exposed resources (docs, indexes, schemas).
- `functions.list_mcp_resource_templates` — discover parameterized resources (when templates exist).
- `functions.read_mcp_resource` — read a specific resource by URI.

### opensrc usage (Codex CLI)

- Treat `opensrc/` as read-only dependency source for internals/edge cases; check `opensrc/sources.json` first and cite exact paths + versions when used.
- Fetch npm sources with `npx opensrc <package>` or `npx opensrc <package>@<version>`; GitHub sources with `npx opensrc <owner>/<repo>[@tag]`.
- List/remove sources via `npx opensrc list` and `npx opensrc remove <name>`.
- Prefer non-interactive flags when available (for example `--modify=false`).
- Refresh sources after dependency upgrades.
