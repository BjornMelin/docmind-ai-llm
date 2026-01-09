# DocMind AI — Software Requirements Specification (SRS)

Version: 1.5.0 • Date: 2026-01-09 • Owner: Eng/Arch
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

FR-001 The system **shall** ingest documents using Unstructured with `strategy=auto` and apply OCR fallback when needed. Source: ADR‑002; Accept: see AC‑FR‑001.<br>
FR-002 The system **shall** build a LlamaIndex `IngestionPipeline` with per-node+transform caching (DuckDBKV). Source: ADR‑010; Accept: AC‑FR‑002.<br>
FR-003 The system **shall** create canonical nodes with deterministic IDs and include `pdf_page_image` nodes for pages. Source: ADR‑002; Accept: AC‑FR‑003.<br>
FR-004 The system **shall** embed text with BGE‑M3 and images with SigLIP by default (OpenCLIP MAY be selected explicitly). Source: ADR‑002; Accept: AC‑FR‑004.<br>
FR-005 The system **shall** persist vectors in Qdrant with named vectors `text-dense` and `text-sparse` and perform server‑side hybrid queries via the Query API. Default fusion **SHALL** be RRF; DBSF MAY be enabled experimentally via environment when supported by Qdrant. The sparse index **SHALL** prefer FastEmbed BM42 with IDF modifier; fallback to BM25 when BM42 is unavailable. During hybrid queries, the system **shall** emit telemetry fields: `retrieval.fusion_mode`, `retrieval.prefetch_*_limit`, `retrieval.fused_limit`, `retrieval.return_count`, `retrieval.latency_ms`, `retrieval.sparse_fallback`, and `dedup.*`. Source: SPEC‑004/ADR‑005/006; Accept: AC‑FR‑005.<br>
FR-006 The system **shall not** implement client-side fusion as default; all hybrid fusion **SHALL** occur server‑side in Qdrant. Source: SPEC‑004; Accept: AC‑FR‑006.<br>
FR-007 The system **shall** rerank text with BGE‑reranker‑v2‑m3 and visual/page-image nodes with SigLIP text–image similarity by default; ColPali MAY be enabled when thresholds are met (visual‑heavy corpora, small K, sufficient GPU). Source: SPEC‑005/ADR‑037; Accept: AC‑FR‑007.<br>
FR-008 The system **shall** run hybrid and reranking **always‑on** with internal caps/timeouts; no UI toggles. Ops overrides MAY be provided via environment variables. (canonical: `DOCMIND_RETRIEVAL__USE_RERANKING`, mapping to `settings.retrieval.use_reranking`). Source: ADR‑024; Accept: AC‑FR‑008.<br>
FR-009 The system **shall** support optional GraphRAG via LlamaIndex PropertyGraphIndex using documented APIs only (e.g., `as_retriever`, `get_rel_map`) and a UI toggle. No custom synonym retriever. Compose RouterQueryEngine with vector+graph tools (fallback to vector when graph missing). Persist via SnapshotManager with manifest hashing, `graph_exports` metadata, and a single-writer lock. Graph exports SHALL be produced from `get_rel_map` to JSONL (required) and Parquet (optional) with telemetry recorded in manifest metadata. Source: ADR‑019/ADR‑038/SPEC‑006/SPEC‑014; Accept: AC‑FR‑009. (Status: Completed)<br>
FR-009.1 The system **shall** display a staleness badge in Chat when manifest hashes differ, with tooltip copy exactly: “Snapshot is stale (content/config changed). Rebuild in Documents → Rebuild GraphRAG Snapshot.” The check MUST be local‑only (no network). Source: SPEC‑014; Accept: AC‑FR‑009.<br>
FR-009.2 The system **shall** implement SnapshotManager single‑writer lock semantics with bounded timeout, `portalocker`-backed metadata (owner, heartbeat, ttl, takeover count), atomic `_tmp → <timestamp>` rename, and the tri-file manifest (`manifest.jsonl`, `manifest.meta.json`, `manifest.checksum`) plus `graph_exports` metadata. Source: SPEC‑014; Accept: AC‑FR‑009‑LOCK.<br>
FR-009.3 The system **shall** provide exports with JSONL required and Parquet optional (guarded by PyArrow), using timestamped filenames (`graph_export-YYYYMMDDTHHMMSSZ.*`) stored under `graph/` and recording telemetry (`dest_path`, `seed_count`, `size_bytes`, `duration_ms`). Source: SPEC‑006; Accept: AC‑FR‑009.<br>
FR-009.4 The system **shall** select export seeds deterministically, de‑duplicate, and cap at 32 items. Source: SPEC‑006; Accept: AC‑FR‑009‑SEEDS.<br>
FR-009.5 The system **shall** validate export paths as non‑egress, sanitize file names, and block symlink targets. Source: SPEC‑011; Accept: AC‑FR‑009‑SEC.<br>
FR-009.6 The system **shall** emit telemetry events for router selection, staleness detection, export actions, and traversal depth (where applicable). Source: SPEC‑012; Accept: AC‑FR‑OBS‑001.<br>
FR-010 The system **shall** provide a multipage Streamlit UI using `st.Page`/`st.navigation` with Chat, Documents, Analytics, Settings. Source: ADR‑013/SPEC‑008; Accept: AC‑FR‑010.<br>
FR-011 The system **shall** implement native chat streaming via `st.chat_message` + `st.chat_input` + `st.write_stream`. Source: ADR‑013/SPEC‑008; Accept: AC‑FR‑011.<br>
FR-012 The system **shall** allow users to select an LLM provider among llama.cpp, vLLM, Ollama, LM Studio, and choose the model at runtime in UI and settings. Source: ADR‑009; Accept: AC‑FR‑012.<br>
FR-013 The system **shall** provide OpenAI‑compatible client wiring for vLLM, Ollama, LM Studio, and llama.cpp server modes. Source: ADR‑009; Accept: AC‑FR‑013.<br>
FR-014 The system **shall** run a LangGraph‑supervised multi‑agent flow with deterministic JSON‑schema outputs when available. Source: ADR‑001; Accept: AC‑FR‑014.<br>
FR-015 The system **shall** persist ingestion cache via DuckDBKV and operational metadata via SQLite WAL. Source: ADR‑010; Accept: AC‑FR‑015.<br>
FR-016 The system **shall** provide an evaluation harness for IR (BEIR/M‑BEIR) and E2E (RAGAS) runnable offline. Source: ADR‑011; Accept: AC‑FR‑016.<br>
FR-017 The system **shall** collect minimal observability (latency, memory, top‑k, fusion mode, reranker hits) locally. (Status: Implemented)<br>
FR-020 The system **shall** provide a file‑based prompt template system built on LlamaIndex `RichPromptTemplate` with YAML front matter metadata and presets (tones/roles/lengths). It SHALL expose minimal APIs to list templates, get metadata, render text prompts, and format chat messages, and SHALL fully replace legacy `src/prompts.py` constants without back‑compat shims. Source: ADR‑020/SPEC‑020; Accept: AC‑FR‑020.<br>
FR-021 The system **shall** pre-validate Settings UI configuration before persisting changes and must not use unsafe HTML rendering in UI elements. Source: SPEC‑022/ADR‑041; Accept: AC‑FR‑021.<br>
FR-022 The system **shall** persist Chat history locally across refresh/restart and provide per-session “clear/purge” actions. Source: SPEC‑041/ADR‑057; Accept: AC‑FR‑022.<br>
FR-023 The system **shall** provide a keyword/lexical retrieval tool for exact term lookups (gated by config; disabled by default). Source: SPEC‑025/ADR‑044; Accept: AC‑FR‑023.<br>
FR-024 The system **shall** provide a canonical programmatic ingestion API for local filesystem inputs and maintain a thin legacy facade for documentation compatibility. Source: SPEC‑026/ADR‑045; Accept: AC‑FR‑024.<br>
FR-025 The system **shall** support background ingestion and snapshot rebuild jobs in the Documents UI with progress reporting and best-effort cancellation (no partial snapshot publication). Source: SPEC‑033/ADR‑052; Accept: AC‑FR‑025.<br>
FR-026 The system **shall** support an optional semantic response cache with strict invalidation by corpus/config/model/template/params, and must not store raw prompts. Source: SPEC‑038/ADR‑035; Accept: AC‑FR‑026.<br>
FR-027 The system **shall** support manual local backups (snapshots + cache + optional uploads/analytics) with retention/rotation and documented restore steps. Source: SPEC‑037/ADR‑033; Accept: AC‑FR‑027.<br>
FR-028 The system **shall** support document analysis modes `auto | separate | combined` with deterministic, offline-safe execution and best-effort cancellation. Source: SPEC‑036/ADR‑023; Accept: AC‑FR‑028.<br>
FR-029 The system **shall** enforce and propagate an agent decision timeout budget so nested tool/LLM calls cannot exceed it, and it SHALL fail gracefully when the deadline is exceeded. Source: SPEC‑040/ADR‑056; Accept: AC‑FR‑029.<br>
FR-030 The system **shall** provide multi-session Chat management (create/rename/delete/select) and store session metadata locally. Source: SPEC‑041/ADR‑057; Accept: AC‑FR‑030.<br>
FR-031 The system **shall** support branching/time travel for Chat sessions: list checkpoints, fork from a checkpoint, and resume execution from the fork while preserving history. Source: SPEC‑041/ADR‑057; Accept: AC‑FR‑031.<br>
FR-032 The system **shall** support long-term memory (facts/preferences) with metadata-filtered recall and user-visible review/purge controls. Source: SPEC‑041/ADR‑057; Accept: AC‑FR‑032.<br>

## 3. Non‑Functional Requirements (NFR‑###) — ISO/IEC 25010

### Functional suitability

NFR‑FS‑001 The app **shall** achieve ≥0.7 nDCG@10 on a small bundled eval set (text RAG). Verification: test+analysis.<br>
NFR‑FS‑002 With visual reranker on mixed corpora, Recall@20 **shall** improve ≥5% over no‑rerank baseline. Verification: test.<br>

### Reliability

NFR‑REL‑001 The app **shall** recover from vector store restarts without re‑ingestion (idempotent upsert). Verification: demo.<br>
NFR‑REL‑002 Cache hits **shall** be deterministic across runs given same inputs and config. Verification: test.<br>

### Performance efficiency

NFR‑PERF‑001 Chat p50 end‑to‑end latency ≤2.0 s on mid‑GPU profile; ≤6.0 s on CPU‑only profile. Verification: test.<br>
NFR‑PERF‑002 Rerank P95 for text top‑40 ≤150 ms on 4070‑class GPU; visual SigLIP top‑10 ≤150 ms; ColPali top‑10 ≤400 ms when enabled. Verification: test.<br>
NFR‑PERF‑003 Qdrant hybrid query p50 ≤120–200 ms for fused_top_k=60 on local machine. Verification: test.<br>

### Usability

NFR‑USE‑001 Streamlit UI navigable with keyboard; forms avoid unnecessary reruns. Verification: inspection.<br>

### Observability

NFR‑OBS‑001 The app **shall** emit structured, local-first telemetry events (JSONL) for key actions (router selection, staleness detection, exports, job lifecycle) with sampling/rotation controls. Verification: test.<br>
NFR‑OBS‑002 The app **shall** support optional OpenTelemetry tracing and metrics export when explicitly enabled; disabled by default and safe for offline operation. Verification: test.<br>

### Security

NFR‑SEC‑001 Default egress disabled; only local endpoints allowed unless explicitly configured. Verification: inspection.<br>
NFR‑SEC‑002 Local data **shall** remain on device; logging excludes sensitive content.<br>
NFR‑SEC‑003 Optional AES‑GCM encryption‑at‑rest available; off by default. Verification: test.<br>
NFR-SEC-004 Streamlit UI **shall not** execute unsafe HTML/JS; `unsafe_allow_html=True` is prohibited in production UI. Verification: inspection+test.<br>

### Compatibility

NFR‑COMP‑001 Windows/macOS/Linux supported; Apple Metal via llama.cpp; AMD ROCm via vLLM. Verification: demo.<br>

### Maintainability

NFR‑MAINT‑001 Library‑first: app code **shall not** re‑implement features available in LlamaIndex/Qdrant/Streamlit. Verification: inspection.<br>
NFR‑MAINT‑002 Pylint score ≥9.5; Ruff passes.<br>
NFR-MAINT-003 No placeholder APIs (TODO/NotImplemented) in shipped production modules; docs/specs/RTM must match code. Verification: inspection+quality gates.<br>

### Portability

NFR‑PORT‑001 Single definitive architecture; no prod/local forks; configuration via settings/UI only. Verification: inspection.<br>
NFR-PORT-002 Cross-platform paths and cache env overrides supported for local packaging and model predownload workflows. Verification: inspection.<br>
NFR-PORT-003 Docker/compose artifacts **shall** run out-of-the-box for local deployments and be reproducible from `uv.lock`. Verification: manual run+inspection.<br>

## 4. Data and Interface Requirements

- Vector store: Qdrant collection with named vectors `text-dense` (float32) and `text-sparse` (CSR), deterministic point IDs = SHA‑256(content). Hybrid queries enabled with server‑side fusion.<br>
- Ingestion cache: DuckDBKV; pipeline caches node+transform hashes.<br>
- LLM API: OpenAI‑compatible for vLLM/Ollama/LM Studio/llama.cpp server.<br>
- UI contracts: `st.Page` navigation, chat stream, status blocks, fragments.<br>
- Configuration schema: nested groups expose canonical policy surfaces.<br>
  - `openai.*` fields normalize `base_url` to a single `/v1` suffix and accept placeholder `api_key` values for loopback servers, matching the OpenAI-compatible contracts for LM Studio and vLLM. Concrete examples live in the [configuration reference](../developers/configuration-reference.md#openai-compatible-local-servers-lm-studio-vllm-llamacpp).
  - `security.*` enforces local-first defaults. `allow_remote_endpoints` is false by default, remote hosts **must** appear in the `endpoint_allowlist`, and effective policy state is surfaced read-only in the UI.
  - `retrieval.hybrid.*` unifies server-side hybrid gating: `retrieval.enable_server_hybrid` (bool) plus `retrieval.fusion_mode` ∈ {`rrf`, `dbsf`}. No legacy client-side fusion flags remain.

## 5. Compliance, privacy, and security controls

- Local‑first default, opt‑in remote endpoints; CORS disabled in Streamlit.<br>
- License inventory: Apache‑2.0 (LlamaIndex, Qdrant client), MIT (BGE‑M3), OpenCLIP (MIT), SigLIP (Apache‑2.0).<br>

## 6. Assumptions, dependencies, out-of-scope

- Assumption: Qdrant running locally via docker or embedded (qdrant-local acceptable).<br>
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
FR‑SEC‑IMG‑ENC The system **shall** support optional encryption‑at‑rest for page images using AES‑GCM; metadata SHALL record `encrypted=true`, `alg`, and `kid`. Keys SHALL be provisioned via env/keystore and never logged in plaintext. Source: SPEC‑011; Accept: AC‑FR‑SEC‑IMG‑ENC.<br>
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

FR‑SEC‑NET‑001 The system **shall** default to offline‑first behavior; remote endpoints are disabled unless explicitly allowlisted. LM Studio endpoints MUST end with `/v1`. Source: SPEC‑011/ADR‑024; Accept: AC‑FR‑SEC‑NET‑001.<br>

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
