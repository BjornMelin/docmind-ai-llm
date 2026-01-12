# Implementation Prompt — 0044: UI + Persistence (Sessions, Time Travel, Memory, Multimodal Rendering)

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

1. Ensure chat DB path is anchored under `settings.data_dir` in tests and prod.
2. Ensure chat sessions sidebar updates `st.query_params["chat"]` and reruns safely.
3. Ensure time travel sets `chat_resume_checkpoint_id` and triggers a rerun.
4. Ensure multimodal sources render thumbnails:
   - resolve `ArtifactRef`
   - decrypt `.enc` images at render-time using `open_image_encrypted`
5. Ensure “Visual search” sidebar runs SigLIP image→image and renders results.
6. Verify AppTest smoke:
   - `uv run python scripts/run_tests.py --fast`

## Acceptance criteria

- Chat page loads without external services (fail-open).
- Sessions are durable across reruns (SQLite WAL).
- Images render without persisting blobs.

