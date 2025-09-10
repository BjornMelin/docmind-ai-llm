# UI Wiring Plan

## Documents Page (`src/pages/02_documents.py`)

- Add checkbox: Build GraphRAG (beta) [default off; configurable]
- On successful ingestion:
  - Always: create VectorStoreIndex, hybrid retriever as needed; save to st.session_state
  - If GraphRAG enabled: build PropertyGraphIndex; save to st.session_state as pg_index
  - Expose export buttons: JSONL (always), Parquet (if pyarrow); cap seeds to 32 by default
  - Persist snapshot via SnapshotManager; display snapshot path
  - Snapshot Utilities: button “Rebuild GraphRAG Snapshot” to re-run snapshot on demand (no reindex)
  - Expander “About snapshots” describing atomic rename, manifest hashes, and staleness

## Chat Page (`src/pages/01_chat.py`)

- settings_override: forward router_engine, vector_index, pg_index
- On load: attempt to read latest manifest; compute staleness and render badge with explicit copy: “Snapshot is stale (content/config changed). Open Documents → ‘Rebuild GraphRAG Snapshot’ to refresh.”
- Default chat strategy: router when pg_index present; else vector
- Settings panel: toggle GraphRAG usage; show traversal depth (read-only default 1)

## Telemetry (optional)

- Log: router selection, traversal depth, staleness, export actions
