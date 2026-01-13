---
prompt: PROMPT-044
title: UI + Persistence (Sessions, Time Travel, Memory, Multimodal Rendering)
status: Completed
date: 2026-01-13
version: 1.0
related_adrs: ["ADR-057", "ADR-058"]
related_specs: ["SPEC-041", "SPEC-042"]
---

## Implementation Prompt — 0044: UI + Persistence (Sessions, Time Travel, Memory, Multimodal Rendering)

**Purpose:** Ship final-release Streamlit UX and durability for multimodal chat.  
**Source of truth:** ADR-058 + ADR-057 + SPEC-042 + SPEC-041.

## Scope

- Chat page:
  - sessions (create/rename/delete/select)
  - time travel (resume from checkpoint id)
  - memory review UI (store/search/delete)
  - multimodal sources rendering (thumbnails via artifact refs)
  - query-by-image “Visual search”
- Documents page:
  - preview page-image exports via artifact refs
  - snapshot utilities always visible when indices exist

## Key files

- Chat: `src/pages/01_chat.py`
- Documents: `src/pages/02_documents.py`
- Sessions UI: `src/ui/chat_sessions.py`
- Chat DB: `src/persistence/chat_db.py`
- Memory store: `src/persistence/memory_store.py`

## Streamlit requirements (SMA)

- Avoid `unsafe_allow_html=True` for user-provided content.
- Use `st.cache_resource` for long-lived objects (DB connections, checkpointers).
- Keep UI responsive by limiting rendered images and using expanders.

## Step-by-step

1. Anchor the chat DB path under `settings.data_dir` in tests and prod.
2. Update the chat sessions sidebar to set `st.query_params["chat"]` and rerun safely.
3. Trigger time travel forks immediately from a checkpoint and rerun.
4. Render multimodal source thumbnails:
   - resolve `ArtifactRef`
   - decrypt `.enc` images at render-time using `open_image_encrypted`
5. Run “Visual search” sidebar SigLIP image→image and render results.
6. Verify AppTest smoke:
   - `uv run python scripts/run_tests.py --fast`

## Acceptance criteria

- Chat page loads without external services (fail-open).
- Sessions are durable across reruns (SQLite WAL).
- Images render without persisting blobs.
