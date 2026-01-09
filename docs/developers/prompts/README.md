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
2. `docs/developers/prompts/prompt-022-settings-ui-hardening.md`
3. `docs/developers/prompts/prompt-026-ingestion-api-facade.md`
4. `docs/developers/prompts/prompt-025-keyword-tool-sparse-only.md`
5. `docs/developers/prompts/prompt-032-documents-snapshot-service-boundary.md`
6. `docs/developers/prompts/prompt-033-background-ingestion-jobs.md`
7. `docs/developers/prompts/prompt-024-chat-persistence-simplechatstore.md`
8. `docs/developers/prompts/prompt-027-remove-legacy-main-entrypoint.md`
9. `docs/developers/prompts/prompt-028-safe-logging-no-pii-redactor.md`
10. `docs/developers/prompts/prompt-031-config-discipline-env-bridges.md`
11. `docs/developers/prompts/prompt-035-config-surface-pruning-unused-knobs.md`
12. `docs/developers/prompts/prompt-034-analytics-page-hardening.md`
13. `docs/developers/prompts/prompt-030-multimodal-helper-cleanup.md`
14. `docs/developers/prompts/prompt-029-docs-consistency-pass.md`
15. `docs/developers/prompts/prompt-023-containerization-hardening.md`

## Codex tool + skill usage (fresh sessions)

Before starting any prompt in a fresh Codex TUI session:

1. Read the tool inventory: `~/prompt_library/assistant/codex-inventory.md`.
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
