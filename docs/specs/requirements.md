# DocMind AI — Software Requirements Specification (SRS)

Version: 1.2.0 • Date: 2025-09-07 • Owner: Eng/Arch
Scope: Local-first, multimodal Agentic RAG app with hybrid retrieval, reranking, GraphRAG, and multi-provider LLM runtimes.

## 0. Front-matter

### Glossary

- RAG: Retrieval-Augmented Generation
- GraphRAG: Graph-aware retrieval via property graphs
- LLM: Large Language Model
- VLM: Vision-Language Model
- RRF: Reciprocal Rank Fusion

### Context & Stakeholders

- Users: Analysts, researchers, engineers running local RAG.
- Operators: Individual users on mixed hardware (CPU-only, Nvidia CUDA, AMD ROCm, Apple Silicon).
- Maintainers: DocMind AI dev team.

## 1. Business goals and constraints

- G1: Operate fully offline by default; no paid APIs required.
- G2: Support heterogeneous hardware and OS with selectable local LLM providers in-UI.
- G3: Provide accurate multimodal retrieval with tunable hybrid search and reranking.
- C1: Library-first; minimize custom code (KISS/DRY/YAGNI).
- C2: One definitive architecture (no prod/local split).

## 2. Functional Requirements (FR-###)

FR-001 The system **shall** ingest documents using Unstructured with `strategy=auto` and apply OCR fallback when needed. Source: ADR‑002; Accept: see AC‑FR‑001.  
FR-002 The system **shall** build a LlamaIndex `IngestionPipeline` with per-node+transform caching (DuckDBKV). Source: ADR‑010; Accept: AC‑FR‑002.  
FR-003 The system **shall** create canonical nodes with deterministic IDs and include `pdf_page_image` nodes for pages. Source: ADR‑002; Accept: AC‑FR‑003.  
FR-004 The system **shall** embed text with BGE‑M3 and images with SigLIP by default (OpenCLIP MAY be selected explicitly). Source: ADR‑002; Accept: AC‑FR‑004.  
FR-005 The system **shall** persist vectors in Qdrant with named vectors `text-dense` and `text-sparse` and perform server‑side hybrid queries via the Query API. Default fusion **SHALL** be RRF; DBSF MAY be enabled experimentally via environment when supported by Qdrant. The sparse index **SHALL** prefer FastEmbed BM42 with IDF modifier; fallback to BM25 when BM42 is unavailable. Source: SPEC‑004/ADR‑005/006; Accept: AC‑FR‑005.  
FR-006 The system **shall not** implement client-side fusion as default; all hybrid fusion **SHALL** occur server‑side in Qdrant. Source: SPEC‑004; Accept: AC‑FR‑006.  
FR-007 The system **shall** rerank text with BGE‑reranker‑v2‑m3 and visual/page-image nodes with SigLIP text–image similarity by default; ColPali MAY be enabled when thresholds are met (visual‑heavy corpora, small K, sufficient GPU). Source: SPEC‑005/ADR‑037; Accept: AC‑FR‑007.  
FR-008 The system **shall** run hybrid and reranking **always‑on** with internal caps/timeouts; no UI toggles. Ops overrides MAY be provided via environment variables. Source: ADR‑024; Accept: AC‑FR‑008.  
FR-009 The system **shall** support optional GraphRAG via LlamaIndex PropertyGraphIndex using documented APIs only (e.g., `as_retriever`, `get_rel_map`) and a UI toggle. No custom synonym retriever. Compose RouterQueryEngine with vector+graph tools (fallback to vector when graph missing). Persist via SnapshotManager with manifest hashing and a lock. Graph exports SHALL be produced from `get_rel_map` to JSONL and Parquet (when PyArrow is available). Source: ADR‑019/ADR‑038/SPEC‑006/SPEC‑014; Accept: AC‑FR‑009. (Status: Implemented)  
FR-009.1 The system **shall** display a staleness badge in Chat when manifest hashes differ, with tooltip copy exactly: “Snapshot is stale (content/config changed). Rebuild in Documents → Rebuild GraphRAG Snapshot.” The check MUST be local‑only (no network). Source: SPEC‑014; Accept: AC‑FR‑009.  
FR-009.2 The system **shall** implement SnapshotManager single‑writer lock semantics with bounded timeout and atomic `_tmp → <timestamp>` rename, and include required manifest fields. Source: SPEC‑014; Accept: AC‑FR‑009‑LOCK.  
FR-009.3 The system **shall** provide exports with JSONL required and Parquet optional (guarded by pyarrow). Source: SPEC‑006; Accept: AC‑FR‑009.  
FR-009.4 The system **shall** select export seeds deterministically, de‑duplicate, and cap at 32 items. Source: SPEC‑006; Accept: AC‑FR‑009‑SEEDS.  
FR-009.5 The system **shall** validate export paths as non‑egress, sanitize file names, and block symlink targets. Source: SPEC‑011; Accept: AC‑FR‑009‑SEC.  
FR-009.6 The system **shall** emit telemetry events for router selection, staleness detection, export actions, and traversal depth (where applicable). Source: SPEC‑012; Accept: AC‑FR‑OBS‑001.  
FR-010 The system **shall** provide a multipage Streamlit UI using `st.Page`/`st.navigation` with Chat, Documents, Analytics, Settings. Source: ADR‑012; Accept: AC‑FR‑010.  
FR-011 The system **shall** implement native chat streaming via `st.chat_message` + `st.chat_input` + `st.write_stream`. Source: ADR‑012; Accept: AC‑FR‑011.  
FR-012 The system **shall** allow users to select an LLM provider among llama.cpp, vLLM, Ollama, LM Studio, and choose the model at runtime in UI and settings. Source: ADR‑009; Accept: AC‑FR‑012.  
FR-013 The system **shall** provide OpenAI‑compatible client wiring for vLLM, Ollama, LM Studio, and llama.cpp server modes. Source: ADR‑009; Accept: AC‑FR‑013.  
FR-014 The system **shall** run a LangGraph‑supervised multi‑agent flow with deterministic JSON‑schema outputs when available. Source: ADR‑001; Accept: AC‑FR‑014.  
FR-015 The system **shall** persist ingestion cache via DuckDBKV and operational metadata via SQLite WAL. Source: ADR‑010; Accept: AC‑FR‑015.  
FR-016 The system **shall** provide an evaluation harness for IR (BEIR/M‑BEIR) and E2E (RAGAS) runnable offline. Source: ADR‑011; Accept: AC‑FR‑016.  
FR-017 The system **shall** collect minimal observability (latency, memory, top‑k, fusion mode, reranker hits) locally.
FR-020 The system **shall** provide a file‑based prompt template system built on LlamaIndex `RichPromptTemplate` with YAML front matter metadata and presets (tones/roles/lengths). It SHALL expose minimal APIs to list templates, get metadata, render text prompts, and format chat messages, and SHALL fully replace legacy `src/prompts.py` constants without back‑compat shims. Source: ADR‑020/SPEC‑020; Accept: AC‑FR‑020.

## 3. Non‑Functional Requirements (NFR‑###) — ISO/IEC 25010

### Functional suitability

NFR‑FS‑001 The app **shall** achieve ≥0.7 nDCG@10 on a small bundled eval set (text RAG). Verification: test+analysis.  
NFR‑FS‑002 With visual reranker on mixed corpora, Recall@20 **shall** improve ≥5% over no‑rerank baseline. Verification: test.

### Reliability

NFR‑REL‑001 The app **shall** recover from vector store restarts without re‑ingestion (idempotent upsert). Verification: demo.  
NFR‑REL‑002 Cache hits **shall** be deterministic across runs given same inputs and config. Verification: test.

### Performance efficiency

NFR‑PERF‑001 Chat p50 end‑to‑end latency ≤2.0 s on mid‑GPU profile; ≤6.0 s on CPU‑only profile. Verification: test.  
NFR‑PERF‑002 Rerank P95 for text top‑40 ≤150 ms on 4070‑class GPU; visual SigLIP top‑10 ≤150 ms; ColPali top‑10 ≤400 ms when enabled. Verification: test.  
NFR‑PERF‑003 Qdrant hybrid query p50 ≤120–200 ms for fused_top_k=60 on local machine. Verification: test.

### Usability

NFR‑USE‑001 Streamlit UI navigable with keyboard; forms avoid unnecessary reruns. Verification: inspection.  

### Security

NFR‑SEC‑001 Default egress disabled; only local endpoints allowed unless explicitly configured. Verification: inspection.  
NFR‑SEC‑002 Local data **shall** remain on device; logging excludes sensitive content.
NFR‑SEC‑003 Optional AES‑GCM encryption‑at‑rest available; off by default. Verification: test.

### Compatibility

NFR‑COMP‑001 Windows/macOS/Linux supported; Apple Metal via llama.cpp; AMD ROCm via vLLM. Verification: demo.  

### Maintainability

NFR‑MAINT‑001 Library‑first: app code **shall not** re‑implement features available in LlamaIndex/Qdrant/Streamlit. Verification: inspection.  
NFR‑MAINT‑002 Pylint score ≥9.5; Ruff passes.

### Portability

NFR‑PORT‑001 Single definitive architecture; no prod/local forks; configuration via settings/UI only. Verification: inspection.

## 4. Data and Interface Requirements

- Vector store: Qdrant collection with named vectors `text-dense` (float32) and `text-sparse` (CSR), deterministic point IDs = SHA‑256(content). Hybrid queries enabled with server‑side fusion.  
- Ingestion cache: DuckDBKV; pipeline caches node+transform hashes.  
- LLM API: OpenAI‑compatible for vLLM/Ollama/LM Studio/llama.cpp server.  
- UI contracts: `st.Page` navigation, chat stream, status blocks, fragments.  

## 5. Compliance, privacy, and security controls

- Local‑first default, opt‑in remote endpoints; CORS disabled in Streamlit.  
- License inventory: Apache‑2.0 (LlamaIndex, Qdrant client), MIT (BGE‑M3), OpenCLIP (MIT), SigLIP (Apache‑2.0).  

## 6. Assumptions, dependencies, out-of-scope

- Assumption: Qdrant running locally via docker or embedded (qdrant-local acceptable).  
- Out‑of‑scope: proprietary cloud LLMs.

## 7. Verification methods per requirement

- FR‑001..017,020: Test, analysis, or demonstration; see RTM.

## 8. Traceability seeds

- Linked ADRs: 001..012 consolidated set.
- Code refs: e.g., src/config/llm_factory.py (provider wiring) and Streamlit UI (pages).

## Acceptance Criteria (Gherkin excerpts)

### AC‑FR‑001

```gherkin
Scenario: PDF with images ingests with page images
  Given a PDF with tables and figures
  When I ingest with strategy=auto
  Then nodes shall include at least one "pdf_page_image" node
```

### AC‑FR‑012

```gherkin
Scenario: Switch provider from vLLM to Ollama
  Given Settings provider=vllm and model=A
  When I change provider to "ollama" and model=B and press Save
  Then subsequent chats shall hit the Ollama base_url and model=B
FR‑SEC‑IMG‑ENC The system **shall** support optional encryption‑at‑rest for page images using AES‑GCM; metadata SHALL record `encrypted=true`, `alg`, and `kid`. Keys SHALL be provisioned via env/keystore and never logged in plaintext. Source: SPEC‑011; Accept: AC‑FR‑SEC‑IMG‑ENC.  
```

### AC‑FR‑005‑B

```gherkin
Scenario: Unique pages in final results
  Given fused_top_k=60 and top_k=10
  When hybrid retrieval returns results
  Then the final 10 SHALL be unique by page_id (fallback to source_file:page_number)
```

### AC‑FR‑SEC‑IMG‑ENC

```gherkin
Scenario: Encrypt page image at rest
  Given AES‑GCM keys and a page image
  When I enable encryption and ingest
  Then the stored object SHALL be .enc, metadata SHALL include {encrypted: true, kid}
  And a decrypt step SHALL restore the original bytes
```

### AC‑FR‑020

```gherkin
Scenario: Template catalog and rendering
  Given default templates and presets on disk
  When the UI lists templates and a user selects one
  And the system renders with default context
  Then a non‑empty prompt/message is produced without errors
And no references to src/prompts.py remain in the repository
```

### AC‑FR‑009

```gherkin
Scenario: Router composition and fallback
  Given GraphRAG is enabled and a graph exists
  When I query
  Then RouterQueryEngine SHALL include vector and graph tools
  And route to vector only if the graph is missing or unhealthy

Scenario: Snapshot manifest and staleness
  Given SnapshotManager created storage/<timestamp> with manifest.json
  And current corpus/config hashes differ
  When I open Chat
  Then a staleness badge SHALL be visible

Scenario: Exports
  Given a graph store and seeds
  When I export
  Then JSONL SHALL be written (one relation per line)
  And Parquet SHALL be written when pyarrow is available
```

FR‑SEC‑NET‑001 The system **shall** default to offline‑first behavior; remote endpoints are disabled unless explicitly allowlisted. LM Studio endpoints MUST end with `/v1`. Source: SPEC‑011/ADR‑024; Accept: AC‑FR‑SEC‑NET‑001.  

### AC‑FR‑009‑LOCK

```gherkin
Scenario: Single‑writer snapshot rebuild with timeout
  Given a running rebuild task holds the SnapshotManager lock
  When I attempt a second rebuild within the lock timeout
  Then the UI SHALL present a locked state and the second rebuild SHALL not start
```

### AC‑FR‑009‑SEEDS

```gherkin
Scenario: Deterministic seed selection with cap
  Given a set of candidate seeds derived from ingested documents or retriever top‑K
  When I compute export seeds
  Then the resulting list SHALL be deterministic, de‑duplicated, and contain at most 32 items
```

### AC‑FR‑009‑SEC

```gherkin
Scenario: Non‑egress export path validation
  Given a base data directory
  When I attempt to export to a path outside the base or to a symlink
  Then the export SHALL be blocked with a clear error
```

### AC‑FR‑OBS‑001

```gherkin
Scenario: Telemetry events emitted
  When a query is processed and/or a snapshot/export action occurs
  Then the system SHALL emit events: router_selected, snapshot_stale_detected, export_performed
  And traversal_depth SHALL be recorded when a knowledge_graph route is taken
```

### AC‑FR‑SEC‑NET‑001

```gherkin
Scenario: Reject non‑allowlisted remote endpoint
  Given offline‑first defaults and a strict allowlist
  When I configure a remote LLM URL that is not allowlisted
  Then the Settings page SHALL reject the value with a helpful error
```
