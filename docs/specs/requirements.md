# DocMind AI — Software Requirements Specification (SRS)

Version: 2.0.0 • Date: 2026-01-14 • Owner: Eng/Arch
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
- C3: Primary runtime is CPython **3.13.11** (`requires-python = ">=3.13,<3.14"`).
- C4: High-performance LLM serving (vLLM) is supported **out-of-process** via OpenAI-compatible HTTP; the app must not require `vllm` as an in-env dependency.

## 2. Functional Requirements (FR-###)

| ID | Requirement | Source | Acceptance |
| :--- | :--- | :--- | :--- |
| **FR-001** | The system **shall** ingest documents using Unstructured with `strategy=auto` and apply OCR fallback when needed. | ADR‑002 | AC‑FR‑001 |
| **FR-002** | The system **shall** build a LlamaIndex `IngestionPipeline` with per-node+transform caching (DuckDBKV). | ADR‑010 | AC‑FR‑002 |
| **FR-003** | The system **shall** create canonical nodes with deterministic IDs and include `pdf_page_image` nodes for pages. | ADR‑002 | AC‑FR‑003 |
| **FR-036** | The system **shall** optionally enrich ingested text nodes with sentence spans and entity spans using the centralized spaCy NLP subsystem, controlled by settings and supporting cross-platform device selection (`cpu\|cuda\|apple\|auto`). | SPEC‑015/ADR‑061 | AC‑FR‑036 |
| **FR-004** | The system **shall** embed text with BGE‑M3 and images with SigLIP by default (OpenCLIP MAY be selected explicitly). | ADR‑002 | AC‑FR‑004 |
| **FR-005** | The system **shall** persist vectors in Qdrant with named vectors `text-dense` and `text-sparse` and perform server‑side hybrid queries via the Query API. Default fusion **SHALL** be RRF; DBSF MAY be enabled experimentally via environment when supported by Qdrant. The sparse index **SHALL** prefer FastEmbed BM42 with IDF modifier; fallback to BM25 when BM42 is unavailable. During hybrid queries, the system **shall** emit telemetry fields: `retrieval.fusion_mode`, `retrieval.prefetch_*_limit`, `retrieval.fused_limit`, `retrieval.return_count`, `retrieval.latency_ms`, `retrieval.sparse_fallback`, and `dedup.*`. | SPEC‑004/ADR‑005/006 | AC‑FR‑005 |
| **FR-006** | The system **shall not** implement client-side fusion as default; all hybrid fusion **SHALL** occur server‑side in Qdrant. | SPEC‑004 | AC‑FR‑006 |
| **FR-007** | The system **shall** rerank text with BGE‑reranker‑v2‑m3 and visual/page-image nodes with SigLIP text–image similarity by default; ColPali MAY be enabled when thresholds are met (visual‑heavy corpora, small K, sufficient GPU). | SPEC‑005/ADR‑037 | AC‑FR‑007 |
| **FR-008** | The system **shall** run hybrid and reranking **always‑on** with internal caps/timeouts; no UI toggles. Ops overrides MAY be provided via environment variables (canonical: `DOCMIND_RETRIEVAL__USE_RERANKING`). | ADR‑024 | AC‑FR‑008 |
| **FR-009** | The system **shall** support optional GraphRAG via LlamaIndex PropertyGraphIndex using documented APIs only (e.g., `as_retriever`, `get_rel_map`) and a UI toggle. Compose RouterQueryEngine with vector+graph tools (fallback to vector when graph missing). Persist via SnapshotManager with manifest hashing, `graph_exports` metadata, and a single-writer lock. | ADR‑019/ADR038/SPEC-014 | AC‑FR‑009 |
| **FR-009.1** | The system **shall** display a staleness badge in Chat when manifest hashes differ, with tooltip copy exactly: “Snapshot is stale (content/config changed). Rebuild in Documents → Rebuild GraphRAG Snapshot.” The check MUST be local‑only (no network). | SPEC‑014 | AC‑FR‑009 |
| **FR-009.2** | The system **shall** implement SnapshotManager single‑writer lock semantics with bounded timeout, `portalocker`-backed metadata, atomic `_tmp → <timestamp>` rename, and the tri-file manifest. | SPEC‑014 | AC‑FR‑009‑LOCK |
| **FR-009.3** | The system **shall** provide exports with JSONL required and Parquet optional (guarded by PyArrow), using timestamped filenames stored under `graph/` and recording telemetry. | SPEC‑006 | AC‑FR‑009 |
| **FR-009.4** | The system **shall** select export seeds deterministically, de‑duplicate, and cap at 32 items. | SPEC‑006 | AC‑FR‑009‑SEEDS |
| **FR-009.5** | The system **shall** validate export paths as non‑egress, sanitize file names, and block symlink targets. | SPEC‑011 | AC‑FR‑009‑SEC |
| **FR-009.6** | The system **shall** emit telemetry events for router selection, staleness detection, export actions, and traversal depth (where applicable). | SPEC‑012 | AC‑FR‑OBS‑001 |
| **FR-010** | The system **shall** provide a multipage Streamlit UI using `st.Page`/`st.navigation` with Chat, Documents, Analytics, Settings. | ADR‑013/SPEC‑008 | AC‑FR‑010 |
| **FR-011** | The system **shall** implement native chat streaming via `st.chat_message` + `st.chat_input` + `st.write_stream`. | ADR‑013/SPEC‑008 | AC‑FR‑011 |
| **FR-012** | The system **shall** allow users to select an LLM provider among llama.cpp, vLLM, Ollama, LM Studio, and choose the model at runtime in UI and settings. | ADR‑009 | AC‑FR‑012 |
| **FR-013** | The system **shall** provide OpenAI‑compatible client wiring for vLLM, Ollama, LM Studio, and llama.cpp server modes. | ADR‑009 | AC‑FR‑013 |
| **FR-014** | The system **shall** run a LangGraph‑supervised multi‑agent flow with deterministic JSON‑schema outputs when available. | ADR‑001 | AC‑FR‑014 |
| **FR-015** | The system **shall** persist ingestion cache via DuckDBKV and operational metadata via SQLite WAL. | ADR‑010 | AC‑FR‑015 |
| **FR-016** | The system **shall** provide an evaluation harness for IR (BEIR/M‑BEIR) and E2E (RAGAS) runnable offline. | ADR‑011 | AC‑FR‑016 |
| **FR-017** | The system **shall** collect minimal observability (latency, memory, top‑k, fusion mode, reranker hits) locally. | Status: Implemented | - |
| **FR-020** | The system **shall** provide a file‑based prompt template system built on LlamaIndex `RichPromptTemplate` with YAML front matter metadata and presets. | ADR‑020/SPEC‑020 | AC‑FR‑020 |
| **FR-021** | The system **shall** pre-validate Settings UI configuration before persisting changes and must not use unsafe HTML rendering in UI elements. | SPEC‑022/ADR‑041 | AC‑FR‑021 |
| **FR-022** | The system **shall** persist Chat history locally across refresh/restart and provide per-session “clear/purge” actions via LangGraph `SqliteSaver` and `ArtifactRef` resolution. | SPEC‑041/ADR‑058 | AC‑FR‑022 |
| **FR-023** | The system **shall** provide a keyword/lexical retrieval tool for exact term lookups (gated by config; disabled by default). | SPEC‑025/ADR‑044 | AC‑FR‑023 |
| **FR-024** | The system **shall** provide a canonical programmatic ingestion API for local filesystem inputs. | SPEC‑026/ADR‑045 | AC‑FR‑024 |
| **FR-025** | The system **shall** support background ingestion and snapshot rebuild jobs in the Documents UI with progress reporting and best-effort cancellation. | SPEC‑033/ADR‑052 | AC‑FR‑025 |
| **FR-026** | The system **shall** support an optional semantic response cache with strict invalidation by corpus/config/model/template/params. | SPEC‑038/ADR‑035 | AC‑FR‑026 |
| **FR-027** | The system **shall** support manual local backups (snapshots + cache + optional uploads/analytics) with retention/rotation and documented restore steps. | SPEC‑037/ADR‑033 | AC‑FR‑027 |
| **FR-028** | The system **shall** support document analysis modes `auto \| separate \| combined` with deterministic, offline-safe execution. | SPEC‑036/ADR‑023 | AC‑FR‑028 |
| **FR-029** | The system **shall** enforce and propagate an agent decision timeout budget so nested tool/LLM calls cannot exceed it. | SPEC‑040/ADR‑056 | AC‑FR‑029 |
| **FR-030** | The system **shall** provide multi-session Chat management (create/rename/delete/select) and store session metadata locally in the DocMind registry. | SPEC‑041/ADR‑058 | AC‑FR‑030 |
| **FR-031** | The system **shall** support branching/time travel for Chat sessions: list checkpoints, fork from a checkpoint, and resume execution. | SPEC‑041/ADR‑058 | AC‑FR‑031 |
| **FR-032** | The system **shall** support long-term memory (facts/preferences) with metadata-filtered recall and user-visible review/purge controls. | SPEC‑041/ADR‑058 | AC‑FR‑032 |
| **FR‑SEC‑IMG‑ENC** | The system **shall** support optional encryption‑at‑rest for page images using AES‑GCM; metadata SHALL record `encrypted=true`, `alg`, and `kid`. | SPEC‑011 | AC‑FR‑SEC‑IMG‑ENC |
| **FR‑SEC‑NET‑001** | The system **shall** default to offline‑first behavior; remote endpoints are disabled unless explicitly allowlisted. | SPEC‑011/ADR‑024 | AC‑FR‑SEC‑NET‑001 |

## 3. Non‑Functional Requirements (NFR‑###) — ISO/IEC 25010

| Category | ID | Requirement | Verification |
| :--- | :--- | :--- | :--- |
| **Functional suitability** | **NFR‑FS‑001** | The app **shall** achieve ≥0.7 nDCG@10 on a small bundled eval set (text RAG). | test+analysis |
| **Functional suitability** | **NFR‑FS‑002** | With visual reranker on mixed corpora, Recall@20 **shall** improve ≥5% over no‑rerank baseline. | test |
| **Reliability** | **NFR‑REL‑001** | The app **shall** recover from vector store restarts without re‑ingestion (idempotent upsert). | demo |
| **Reliability** | **NFR‑REL‑002** | Cache hits **shall** be deterministic across runs given same inputs and config. | test |
| **Performance efficiency** | **NFR‑PERF‑001** | Chat p50 end‑to‑end latency ≤2.0 s on mid‑GPU profile; ≤6.0 s on CPU‑only profile. | test |
| **Performance efficiency** | **NFR‑PERF‑002** | Rerank P95 for text top‑40 ≤150 ms on 4070‑class GPU; visual SigLIP top‑10 ≤150 ms; ColPali top‑10 ≤400 ms when enabled. | test |
| **Performance efficiency** | **NFR‑PERF‑003** | Qdrant hybrid query p50 ≤120–200 ms for fused_top_k=60 on local machine. | test |
| **Usability** | **NFR‑USE‑001** | Streamlit UI navigable with keyboard; forms avoid unnecessary reruns. | inspection |
| **Observability** | **NFR‑OBS‑001** | The app **shall** emit structured, local-first telemetry events (JSONL) for key actions (router selection, staleness detection, exports, job lifecycle) with sampling/rotation controls. | test |
| **Observability** | **NFR‑OBS‑002** | The app **shall** support optional OpenTelemetry tracing and metrics export when explicitly enabled; disabled by default and safe for offline operation. | test |
| **Security** | **NFR‑SEC‑001** | Default egress disabled; only local endpoints allowed unless explicitly configured. | inspection |
| **Security** | **NFR‑SEC‑002** | Local data **shall** remain on device; logging excludes sensitive content. | inspection |
| **Security** | **NFR‑SEC‑003** | Optional AES‑GCM encryption‑at‑rest available; off by default. | test |
| **Security** | **NFR-SEC-004** | Streamlit UI **shall not** execute unsafe HTML/JS; `unsafe_allow_html=True` is prohibited in production UI. | inspection+test |
| **Compatibility** | **NFR‑COMP‑001** | Windows/macOS/Linux supported; Apple Metal via llama.cpp; AMD ROCm via vLLM. | demo |
| **Maintainability** | **NFR‑MAINT‑001** | Library‑first: app code **shall not** re‑implement features available in LlamaIndex/Qdrant/Streamlit. | inspection |
| **Maintainability** | **NFR‑MAINT‑002** | Pylint score ≥9.5; Ruff passes. | inspection |
| **Maintainability** | **NFR-MAINT-003** | No placeholder APIs (work-marker comments / NotImplementedError) in shipped production modules; docs/specs/RTM must match code. | inspection+quality gates |
| **Portability** | **NFR‑PORT‑001** | Single definitive architecture; no prod/local forks; configuration via settings/UI only. | inspection |
| **Portability** | **NFR-PORT-002** | Cross-platform paths and cache env overrides supported for local packaging and model predownload workflows. | inspection |
| **Portability** | **NFR-PORT-003** | Docker/compose artifacts **shall** run out-of-the-box for local deployments and be reproducible from `uv.lock`. | manual run+inspection |

## 4. Data and Interface Requirements

- Vector store: Qdrant collection with named vectors `text-dense` (float32) and `text-sparse` (CSR), deterministic point IDs = SHA‑256(content). Hybrid queries enabled with server‑side fusion.

- Ingestion cache: DuckDBKV; pipeline caches node+transform hashes.

- LLM API: OpenAI‑compatible for vLLM/Ollama/LM Studio/llama.cpp server, plus optional Ollama-native `/api/*` support for advanced capabilities (SPEC-043).

- UI contracts: `st.Page` navigation, chat stream, status blocks, fragments.

- Configuration schema: nested groups expose canonical policy surfaces.

  - `openai.*` fields normalize `base_url` to a single `/v1` suffix and accept placeholder `api_key` values for loopback servers, matching the OpenAI-compatible contracts for LM Studio and vLLM. Concrete examples live in the [Configuration Guide](../developers/configuration.md#llm-backend-selection).

  - `security.*` enforces local-first defaults. `allow_remote_endpoints` is false by default, remote hosts **must** appear in the `endpoint_allowlist`, and effective policy state is surfaced read-only in the UI.

  - `retrieval.hybrid.*` unifies server-side hybrid gating: `retrieval.enable_server_hybrid` (bool) plus `retrieval.fusion_mode` ∈ {`rrf`, `dbsf`}. No legacy client-side fusion flags remain.

## 5. Compliance, privacy, and security controls

- Local‑first default, opt‑in remote endpoints; CORS disabled in Streamlit.

- License inventory: Apache‑2.0 (LlamaIndex, Qdrant client), MIT (BGE‑M3), OpenCLIP (MIT), SigLIP (Apache‑2.0).

## 6. Assumptions, dependencies, out-of-scope

- Assumption: Qdrant running locally via docker or embedded (qdrant-local acceptable).

- Out‑of‑scope: general third-party proprietary cloud LLM providers.
- In scope (optional): Ollama Cloud features (including web search/fetch) when explicitly enabled and allowlisted (SPEC-011 + SPEC-043).

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

### AC‑FR‑036

```gherkin
Scenario: NLP enrichment produces a stable schema
  Given NLP enrichment is enabled
  When I enrich a sample text during ingestion
  Then each enriched node SHALL include node.metadata["docmind_nlp"] with:
    | provider   | "spacy" |
    | sentences  | list of {start_char, end_char, text} |
    | entities   | list of {label, text, start_char, end_char} |
  And the schema SHALL remain valid even if entity outputs differ by model/version
```

### AC‑FR‑012

```gherkin
Scenario: Switch provider from vLLM to Ollama
  Given Settings provider=vllm and model=A
  When I change provider to "ollama" and model=B and press Save
  Then subsequent chats shall hit the Ollama base_url and model=B
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
And no references to src/prompts.py remain in production code under src/
```

### AC‑FR‑009

```gherkin
Scenario: Router composition and fallback
  Given GraphRAG is enabled and a graph exists
  When I query
  Then RouterQueryEngine SHALL include vector and graph tools
  And route to vector only if the graph is missing or unhealthy

Scenario: Snapshot manifest and staleness
  Given SnapshotManager created storage/<timestamp> with manifest.meta.json and manifest.jsonl
  And current corpus/config hashes differ
  When I open Chat
  Then a staleness badge SHALL be visible

Scenario: Exports
  Given a graph store and seeds
  When I export
  Then JSONL SHALL be written (one relation per line)
  And Parquet SHALL be written when pyarrow is available
```

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

### AC‑FR‑026

```gherkin
Scenario: Exact cache hit
  Given semantic cache is enabled and a prior request was stored
  When I repeat the same request (same prompt_key, template, model, params)
  Then the system SHALL return the cached response without calling the LLM

Scenario: Semantic cache hit is invalidated by corpus/config hash
  Given semantic cache is enabled and a prior response exists
  And the corpus_hash or config_hash changed
  When I send a near-duplicate request
  Then the system SHALL NOT return the cached response
```

### AC‑FR‑027

```gherkin
Scenario: Backup creates a timestamped directory and prunes old backups
  Given keep_last=2 and there are 2 existing backups
  When I run the backup command successfully
  Then a new backup directory SHALL be created under data/backups/
  And older backups beyond keep_last SHALL be pruned
```

### AC‑FR‑028

```gherkin
Scenario: Separate mode returns per-document outputs
  Given analysis mode is "separate" and multiple documents are selected
  When I run analysis
  Then the UI SHALL render one result per document (tabs) and report timing metadata

Scenario: Combined mode returns a single output
  Given analysis mode is "combined"
  When I run analysis
  Then the UI SHALL render a single combined answer across the corpus
```

### AC‑FR‑029

```gherkin
Scenario: Agent deadline propagation caps per-call timeouts
  Given agent deadline propagation is enabled and decision_timeout=10s
  When the multi-agent workflow runs
  Then individual LLM/tool call timeouts SHALL be capped to <=10s

Scenario: Deadline exceeded returns a graceful timeout response
  Given agent deadline propagation is enabled and decision_timeout is small
  When a call exceeds the remaining budget
  Then the coordinator SHALL return a timeout response (or fallback) without crashing
```

### AC‑FR‑030

```gherkin
Scenario: Create and select chat sessions
  Given the Chat interface with multi-session support
  When I create a new session with name "Research Q1"
  Then the session metadata (name, created_at, selected_at) SHALL be persisted locally
  And I can switch between sessions using the session selector
```

### AC‑FR‑031

```gherkin
Scenario: Branch from checkpoint and resume
  Given a session with chat history and multiple checkpoints
  When I list checkpoints and fork from checkpoint #N
  Then a new branched session SHALL be created with history up to checkpoint #N
  And resuming the fork SHALL allow new queries without losing the branch point
```
