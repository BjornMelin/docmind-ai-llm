---
prompt: PROMPT-041
title: Chat Persistence + Hybrid Agentic Memory (LangGraph SQLite)
status: Completed
date: 2026-01-13
version: 1.0
related_adrs: ["ADR-058", "ADR-057"]
related_specs: ["SPEC-041", "SPEC-042"]
---

## Summary

Implements SPEC-041 as integrated by ADR-058:

- Durable chat sessions via LangGraph SQLite checkpointer + DocMind session registry.
- Time travel (checkpoint list + fork/resume) with spec-compliant “fork immediately on resume”.
- Long-term memory store + review/purge UI, with consolidation policy.

## Notes

- Multi-agent orchestration uses a repo-local, graph-native supervisor (`src/agents/supervisor_graph.py`),
  eliminating reliance on deprecated prebuilt agent helpers and any warning suppression.

## Verification

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
uv run python scripts/run_tests.py --fast
```
