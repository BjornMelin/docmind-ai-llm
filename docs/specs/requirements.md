# DocMind AI — Software Requirements Specification (SRS)

Version: 1.0.0 • Date: 2025-09-05 • Owner: Eng/Arch
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
FR-004 The system **shall** embed text with BGE‑M3 and images with OpenCLIP/SigLIP (selectable by hardware). Source: ADR‑004; Accept: AC‑FR‑004.  
FR-005 The system **shall** persist vectors in Qdrant with named vectors `dense` and `sparse` and enable server-side hybrid queries. Source: ADR‑005/006; Accept: AC‑FR‑005.  
FR-006 The system **shall** support client-side RRF fusion as a fallback mechanism configurable in settings. Source: ADR‑006; Accept: AC‑FR‑006.  
FR-007 The system **shall** rerank text with BGE‑reranker‑v2‑m3 and visual/page-image nodes with ColPali. Source: ADR‑007; Accept: AC‑FR‑007.  
FR-008 The system **shall** expose reranker controls (normalize scores, top‑N, mode auto|text|multimodal). Source: ADR‑036; Accept: AC‑FR‑008.  
FR-009 The system **shall** support optional GraphRAG via LlamaIndex PropertyGraphIndex with default LLMSynonymRetriever and a UI toggle. Source: ADR‑008; Accept: AC‑FR‑009.  
FR-010 The system **shall** provide a multipage Streamlit UI using `st.Page`/`st.navigation` with Chat, Documents, Analytics, Settings. Source: ADR‑012; Accept: AC‑FR‑010.  
FR-011 The system **shall** implement native chat streaming via `st.chat_message` + `st.chat_input` + `st.write_stream`. Source: ADR‑012; Accept: AC‑FR‑011.  
FR-012 The system **shall** allow users to select an LLM provider among llama.cpp, vLLM, Ollama, LM Studio, and choose the model at runtime in UI and settings. Source: ADR‑009; Accept: AC‑FR‑012.  
FR-013 The system **shall** provide OpenAI‑compatible client wiring for vLLM, Ollama, LM Studio, and llama.cpp server modes. Source: ADR‑009; Accept: AC‑FR‑013.  
FR-014 The system **shall** run a LangGraph‑supervised multi‑agent flow with deterministic JSON‑schema outputs when available. Source: ADR‑001; Accept: AC‑FR‑014.  
FR-015 The system **shall** persist ingestion cache via DuckDBKV and operational metadata via SQLite WAL. Source: ADR‑010; Accept: AC‑FR‑015.  
FR-016 The system **shall** provide an evaluation harness for IR (BEIR/M‑BEIR) and E2E (RAGAS) runnable offline. Source: ADR‑011; Accept: AC‑FR‑016.  
FR-017 The system **shall** collect minimal observability (latency, memory, top‑k, fusion mode, reranker hits) locally.

## 3. Non‑Functional Requirements (NFR‑###) — ISO/IEC 25010

### Functional suitability

NFR‑FS‑001 The app **shall** achieve ≥0.7 nDCG@10 on a small bundled eval set (text RAG). Verification: test+analysis.  
NFR‑FS‑002 With visual reranker on mixed corpora, Recall@20 **shall** improve ≥5% over no‑rerank baseline. Verification: test.

### Reliability

NFR‑REL‑001 The app **shall** recover from vector store restarts without re‑ingestion (idempotent upsert). Verification: demo.  
NFR‑REL‑002 Cache hits **shall** be deterministic across runs given same inputs and config. Verification: test.

### Performance efficiency

NFR‑PERF‑001 Chat p50 end‑to‑end latency ≤2.0 s on mid‑GPU profile; ≤6.0 s on CPU‑only profile. Verification: test.  
NFR‑PERF‑002 Rerank P95 for top‑20 ≤150 ms on 4070‑class GPU. Verification: test.  
NFR‑PERF‑003 Qdrant hybrid query p50 ≤80 ms for 10k corpus on local machine. Verification: test.

### Usability

NFR‑USE‑001 Streamlit UI navigable with keyboard; forms avoid unnecessary reruns. Verification: inspection.  

### Security

NFR‑SEC‑001 Default egress disabled; only local endpoints allowed unless explicitly configured. Verification: inspection.  
NFR‑SEC‑002 Local data **shall** remain on device; logging excludes sensitive content.

### Compatibility

NFR‑COMP‑001 Windows/macOS/Linux supported; Apple Metal via llama.cpp; AMD ROCm via vLLM. Verification: demo.  

### Maintainability

NFR‑MAINT‑001 Library‑first: app code **shall not** re‑implement features available in LlamaIndex/Qdrant/Streamlit. Verification: inspection.  
NFR‑MAINT‑002 Pylint score ≥9.5; Ruff passes.

### Portability

NFR‑PORT‑001 Single definitive architecture; no prod/local forks; configuration via settings/UI only. Verification: inspection.

## 4. Data and Interface Requirements

- Vector store: Qdrant collection with named vectors `dense` (float32) and `sparse` (CSR), deterministic point IDs = SHA‑256(content). Hybrid queries enabled.  
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

- FR‑001..017: Test, analysis, or demonstration; see RTM.

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
```
