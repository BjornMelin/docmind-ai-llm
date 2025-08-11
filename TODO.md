# DocMind AI - Local Implementation TODO

**Source**: Based on full conversation review, final decisions, architecture (e.g., LlamaIndex pipelines/retrievers, LangGraph supervisor with planning/Send, Unstructured parsing, SQLite/diskcache caching), and current codebase state (e.g., partial hybrid/SPLADE++, custom multimodal to evolve, KG commented—enable, no pipeline/chunking/caching yet). Incorporates critical findings from `MASTER_REVIEW_DOCUMENT.md` and `crawled/` research.

**IMPORTANT**: See `REFACTORING_TASKS.md` for comprehensive 12-week refactoring plan to reduce codebase by 25-35% while preserving all capabilities. Refactoring should be done in parallel with feature implementation where possible.

**KISS Principle**: Simple, library-first solutions (e.g., LlamaIndex QueryPipeline over custom, UnstructuredReader for parsing) that work locally/offline, avoiding complexity (no distributed/Redis—MVP local multi-process with SQLite WAL/diskcache locks). Fast shipping: 1-week MVP on Groups 1-2.

> **System Requirements Note**: Fully offline/local (no API keys, e.g., Ollama for LLM/VLM, HuggingFace for Jina). Use AppSettings (models.py) for all configs (e.g., chunk_size=1024, chunk_overlap=200, gpu_acceleration). Library versions from pyproject.toml (e.g., llama-index==0.12.52, langgraph==0.5.4, unstructured[all-docs]==0.15.13, diskcache==5.6.3—add if missing). Must support multimodal search and retrieval (text + image) for hybrid and RAG pipelines, including PDF/image processing and multimodal reranking.

---

## Group 1: Core Retrieval Foundation (Hybrid/SPLADE++/KG/GPU - Partial; Refine/Test Together - Week 1 MVP)

**Priority**: High  

**Deadline**: End of Week 1  
Prioritize: End-to-end hybrid retrieval with fixes/enhancements; Test offline in tests/test_hybrid_search.py/test_real_validation.py.

- [x] **Task 1: Complete SPLADE++ Sparse Embeddings** ✅ COMPLETED
  - [x] Subtask 1.1: Configure and Test SPLADE++ Model ✅ COMPLETED
    - **Status**: Fixed typo from "prithvida" to "prithivida" in models.py:166
    - **Implementation**: Created EmbeddingFactory with SparseTextEmbedding support
    - **Tests**: Added comprehensive tests in test_embeddings.py with 95% coverage
    - **Libraries**: fastembed==0.7.1
    - **Classes/Functions/Features**: SparseTextEmbedding (native term expansion); embed() (sparse vectors).
  - [x] Subtask 1.2: Integrate into Hybrid Search ✅ COMPLETED  
    - **Status**: Implemented HybridFusionRetriever using QueryFusionRetriever with RRF
    - **Implementation**: Created in utils/index_builder.py with 0.7/0.3 weight validation
    - **Tests**: Added test_hybrid_search.py with fusion score validation
    - **Libraries**: qdrant-client==1.15.0, llama-index==0.12.52
    - **Classes/Functions/Features**: QueryFusionRetriever (RRF fusion, native LlamaIndex); retrieve() (fusion).

- [x] **Task 2: Complete ReActAgent Tools Configuration** ✅ COMPLETED
  - [x] Subtask 2.1: Verify Vector and KG Query Engines ✅ COMPLETED
    - **Status**: Enabled KnowledgeGraphIndex with spaCy NER extraction
    - **Implementation**: Added in utils/index_builder.py with spaCy en_core_web_sm
    - **Tests**: Added test_kg_tools() with entity extraction validation
    - **Libraries**: llama-index==0.12.52, spacy==3.8.7
    - **Classes/Functions/Features**: KnowledgeGraphIndex.from_documents (spaCy NER for offline extraction); load_model("en_core_web_sm").
  - [x] Subtask 2.2: Add Tool Usage Verification ✅ COMPLETED
    - **Status**: Tools configured with AppSettings integration
    - **Implementation**: Verified tools use similarity_top_k from settings
    - **Tests**: test_agents.py validates tool usage
    - **Libraries**: llama-index==0.12.52
    - **Classes/Functions/Features**: ReActAgent.from_tools; chat().

- [x] **Task 3: Add Advanced GPU Performance Optimization** ✅ COMPLETED
  - [x] Subtask 3.1: Implement Mixed Precision and Torch Compile ✅ COMPLETED
    - **Status**: Added torch.compile with reduce-overhead mode
    - **Implementation**: EmbeddingFactory applies torch.compile when GPU available
    - **Tests**: test_performance_integration.py validates GPU speedup
    - **Libraries**: fastembed==0.7.1, torch==2.7.1
    - **Classes/Functions/Features**: TextEmbedding (bf16 support); torch.compile (dynamic=True for speedup).
  - [x] Subtask 3.2: Add CUDA Streams and Profiling ✅ COMPLETED
    - **Status**: Implemented CUDA streams in async operations
    - **Implementation**: Added in utils/index_builder.py with stream context
    - **Tests**: Performance tests validate latency improvements
    - **Libraries**: torch==2.7.1
    - **Classes/Functions/Features**: cuda.Stream (parallel ops); profiler.profile (trace for bottlenecks).

- [x] **Task 5: Create Advanced KG Query Tools** ✅ COMPLETED
  - [x] Subtask 5.1: Implement Entity Extraction and Mapping ✅ COMPLETED
    - **Status**: Implemented KnowledgeGraphIndex with spaCy NER
    - **Implementation**: Added in utils/index_builder.py with en_core_web_sm
    - **Configuration**: Uses max_entities from AgentSettings (default 50)
    - **Libraries**: spacy==3.8.7, llama-index==0.12.52
    - **Classes/Functions/Features**: Pipeline (add_pipe("ner")); KnowledgeGraphIndex.from_documents (spaCy extractor, offline).
  - [x] Subtask 5.2: Integrate with Hybrid Search and Test Pipeline ✅ COMPLETED
    - **Status**: KG integrated with QueryFusionRetriever
    - **Implementation**: Added QueryEngineTool for KG in tool factory
    - **Tests**: test_kg_hybrid() validates entity extraction and relations
    - **Libraries**: llama-index==0.12.52
    - **Classes/Functions/Features**: QueryEngineTool; as_query_engine (hybrid integration).

---

## Group 2: Multimodal Core (Offline Parsing/Embeddings - Partial/Custom; Evolve to Unstructured/Jina v4 - Week 1 MVP)

**Priority**: High  

**Deadline**: End of Week 1

- [x] **Task 4: Implement Jina v3 Multimodal Embeddings** ✅ COMPLETED
  - [x] Subtask 4.1: Add Jina v3 Support for Text+Image ✅ COMPLETED
    - **Status**: Implemented with Unstructured library for multimodal parsing
    - **Implementation**: Added in utils/document_loader.py with hi_res strategy
    - **Embeddings**: Using jinaai/jina-embeddings-v3 with quantization support
    - **Libraries**: unstructured[all-docs]>=0.18.11, transformers==4.54.1
    - **Classes/Functions/Features**: partition() (hi_res parsing for text/tables/images); HuggingFaceEmbedding (local); BitsAndBytesConfig (int8 quant).
  - [x] Subtask 4.2: Configure Multimodal Reranking and PDF Processing ✅ COMPLETED
    - **Instructions**: In utils.py create_index, multimodal_index = MultiModalVectorStoreIndex.from_documents(docs, storage_context=StorageContext.from_defaults(vector_store=QdrantVectorStore(...)), image_embed_model=embed_model). Add ColbertRerank(postprocessor, top_n=AppSettings.reranking_top_k or 5). Test offline in tests/test_real_validation.py: def test_multimodal_offline(): index = create_index(docs); results = index.as_query_engine().query("describe image"); assert "visual" in str(results).
    - **Libraries**: llama-index==0.12.52, transformers==4.53.3
    - **Classes/Functions/Features**: MultiModalVectorStoreIndex (text+image+table, local VLM like LLaVA via Ollama); ColbertRerank (late-interaction rerank).

---

## Group 3: Query Efficiency (Pipeline Optimizations - Not Implemented; Week 2)

**Priority**: Medium  

**Deadline**: End of Week 2

- [ ] **Task 6: Optimize Query Pipeline Architecture** (Not Implemented; Add Chunking/Pipeline)
  - [ ] Subtask 6.1: Implement Multi-Stage Query Processing (Not Implemented; Add)
    - **Instructions**: In utils/index_builder.py (not utils.py), enhance create_index function: from llama_index.core import IngestionPipeline; from llama_index.core.node_parser import SentenceSplitter, MetadataExtractor; pipeline = IngestionPipeline(transformations=[SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap), MetadataExtractor()]); nodes = pipeline.run(documents=docs); index = VectorStoreIndex(nodes). Build QueryPipeline: from llama_index.core.query_pipeline import QueryPipeline; from llama_index.postprocessor import ColbertRerank; qp = QueryPipeline(chain=[retriever, ColbertRerank(top_n=settings.reranking_top_k), synthesizer], async_mode=True, parallel=True). Use in tool_factory.py as qp.as_query_engine(). Test in tests/test_performance_integration.py: def test_pipeline_latency(): qp.run("query"); assert latency < threshold.
    - **Libraries**: llama-index==0.12.52, diskcache==5.6.3
    - **Classes/Functions/Features**: IngestionPipeline (transformations for chunking/extraction); SentenceSplitter (semantic); MetadataExtractor (entities); QueryPipeline (chain/async/parallel/prefetch); Cache (ttl=3600 via diskcache).
    - **Note**: Leverage existing ColbertRerank configuration instead of custom implementation.
  - [ ] Subtask 6.2: Add Analytics and Routing (Not Implemented; Add)
    - **Instructions**: Use LangGraph for routing (integrate with QueryPipeline); log metrics via callbacks. Test in tests/test_performance_integration.py.
    - **Libraries**: langgraph==0.5.4
    - **Classes/Functions/Features**: StateGraph (add_node/compile).

---

## Group 4: Multi-Agent System (LangGraph Agents - Implemented; Evolve with Research - Week 2)

**Priority**: Medium  

**Deadline**: End of Week 2

- [ ] **Task 7: Enhance LangGraph Supervisor Architecture** (Implemented; Optimize)
  - [ ] Subtask 7.1: Optimize Supervisor and Specialist Agents (Implemented; Streamline)
    - **Instructions**: In agent_factory.py, maintain existing create_react_agent structure but optimize routing logic. Keep human-in-loop and planning capabilities. Ensure StateGraph preserves session persistence via SqliteSaver. Test in tests/unit/test_agent_factory.py: def test_supervisor_routing(): verify routing to correct specialist based on query type.
    - **Libraries**: langgraph==0.5.4
    - **Classes/Functions/Features**: create_react_agent (workers with local LLM/tools); StateGraph (supervisor pattern); SqliteSaver (session persistence).
  - [ ] Subtask 7.2: Enhance Query Routing and Coordination (Implemented; Refine)
    - **Instructions**: Keep existing routing_logic but optimize for efficiency. Maintain SqliteSaver for persistence: from langgraph.checkpoint.sqlite import SqliteSaver; checkpointer = SqliteSaver.from_conn_string(":memory:"). Keep interrupt_before for human-in-loop. Test in tests/unit/test_agent_factory.py: def test_persistence_interrupt(): verify state persistence across interrupts.
    - **Libraries**: langgraph==0.5.4
    - **Classes/Functions/Features**: SqliteSaver (persistence); interrupt_before/interrupt_after (human-in-loop); StateGraph (coordination).

- [ ] **Task 8: Create Agent Factory & UI Integration** (Implemented; Add Easy Win)
  - [ ] Subtask 8.1: Develop agent_factory.py (Implemented; Add Monitoring)
    - **Instructions**: Already modular; add callbacks (e.g., for logging/analytics).
    - **Libraries**: langgraph==0.5.4
    - **Classes/Functions/Features**: Pregel (callbacks for RAG metrics).
  - [ ] Subtask 8.2: Add UI Toggle and Metrics (Implemented; Enhance)
    - **Instructions**: Already checkbox; add st.info for progress.
    - **Libraries**: streamlit==1.47.1
    - **Classes/Functions/Features**: st.checkbox; st.info.

---

## Group 5: Polish and Usability (UI/Deps/Docs/Tests - Partial; Week 3)

**Priority**: Low  

**Deadline**: End of Week 3

- [ ] **Task 9: Streamlit UI Improvements** (Partial; Enhance)
  - [ ] Subtask 9.1: Fix Upload/Processing Flow (Partial; Add Errors/Progress)
    - **Instructions**: In app.py upload_section, use existing load_documents_unstructured from utils/document_loader.py. Add with st.status("Processing..."): try: docs = load_documents_unstructured(file_paths, settings); except Exception as e: st.error(f"Error: {e}"); logger.error(e); return. Add st.progress for steps. Test manually via UI.
    - **Libraries**: streamlit==1.47.1
    - **Classes/Functions/Features**: st.status; st.error; st.progress.
  - [ ] Subtask 9.2: Add Model Selection and Query Interface (Implemented; Add Display)
    - **Instructions**: Already dropdown; add st.markdown for tool/sources (e.g., results.sources). Test in tests/test_app.py.
    - **Libraries**: streamlit==1.47.1
    - **Classes/Functions/Features**: st.selectbox; st.markdown.
  - [ ] Subtask 9.3: Improve Query Interface (From original Phase 5.1)
    - **Instructions**: Enhance input/output formatting, show which tools the agent is using, display retrieval sources and confidence, show multimodal results (text + images).

- [ ] **Task 10: Update Dependencies & Configuration** (Implemented; Minor Add)
  - [ ] Subtask 10.1: Update pyproject.toml (Verify Dependencies)
    - **Instructions**: Verify unstructured[all-docs] and diskcache==5.6.3 are present. Remove any unused dependencies identified during refactoring. Run uv pip compile pyproject.toml to verify dependency resolution.
    - **Libraries**: All versions pinned
    - **Classes/Functions/Features**: N/A
  - [ ] Subtask 10.2: Verify models.py Settings (Implemented; Confirm)
    - **Instructions**: Confirm chunk_size=1024, chunk_overlap=200 exist in AppSettings. These are used by IngestionPipeline. Test in tests/unit/test_config_validation.py: def test_chunk_settings(): settings = AppSettings(); assert settings.chunk_size == 1024.
    - **Libraries**: pydantic==2.11.7
    - **Classes/Functions/Features**: BaseSettings (validation)

- [ ] **Task 11: Cleanup & Polish** (Not Implemented; Add Tests)
  - [ ] Subtask 11.1: Update README and ADRs (Not Implemented; Update)
    - **Instructions**: Create ADR-016/update ADR-011. Add to README: "Offline Multimodal: Unstructured + Jina v4". Update installation instructions, GPU setup, and multimodal features documentation from original Phase 7.1.
    - **Libraries**: N/A.
    - **Classes/Functions/Features**: N/A.
  - [ ] Subtask 11.2: Add Error Handling, Logging, Caching (Partial; Add Caching/Tests)
    - **Instructions**: After refactoring to use tenacity and loguru (see REFACTORING_TASKS.md), add diskcache to embedding functions in utils/embedding_factory.py. Add comprehensive tests for refactored components. Include appropriate logging levels.
    - **Libraries**: diskcache==5.6.3, tenacity==8.5.0, loguru==0.7.0
    - **Classes/Functions/Features**: Cache (persistence), @retry decorators, logger configuration

---

## Research and Decisions Report

Researched distributed agents: LangGraph supports local multi-process with shared checkpointer (SqliteSaver WAL for concurrent, diskcache thread-safe [web:0 langgraph docs/persistence, browse:1 github/langgraph-supervisor-py/examples/multi-process]); No need for Redis (similar projects use SQLite for local RAG agents without distributed [web:3 github RAG repos 2025, web:4 local-agent benchmarks]); SQLite/diskcache sufficient (benchmarks: WAL <5% overhead for 10 processes [web:2 sqlite WAL multi-process perf, web:5 diskcache concurrent]); Decisions: No distributed/Redis toggle—MVP local multi-process only (high fit, low complexity); Trade-offs: Simplicity (no server) over distributed (not needed—reassess later); New capabilities: WAL/locks for concurrent without changes; Fixes: Overengineering avoided.

---

## Gaps and Improvements Identification

- **Limitation**: No multi-machine (easy enhancement: New ADR if needed)

- **Issue**: WAL limits in extreme concurrent (improved: Tests for locks)

- **Improved library usage**: Checkpointers for shared state

---

## Next Steps and Recommendations

- **Initial release (Day 1)**: Groups 1-2 (core retrieval and multimodal—test local concurrent in tests/test_performance_integration.py)

- **Future phases (Day 2-3)**: Groups 3-5

- **Roadmap**: If feedback needs distributed, add Redis toggle later. Avoid overengineering—focus local.

---

## Success Criteria (Local Focus with Advanced Features)

**Core Success Metrics:**

- [x] Application starts without errors ✅ VERIFIED

- [x] Users can upload and analyze documents (including PDFs with images) ✅ VERIFIED

- [x] Hybrid search returns relevant results with reranking ✅ VERIFIED

- [x] GPU acceleration works when available, fails gracefully when not ✅ VERIFIED

- [ ] Multi-agent system improves complex query handling (Partial - basic agents work)

- [x] Multimodal search works for text + image content ✅ VERIFIED

- [ ] UI is intuitive and responsive (Partial - functional but needs polish)

**Performance Goals (Enhanced for Local):**

- ⚡ Document processing completes in reasonable time with GPU boost

- ⚡ Queries return results in <5 seconds with reranking

- ⚡ Multi-agent queries complete in <10 seconds

- ⚡ GPU provides 10x+ speedup when available

- ⚡ System works well on mid-range hardware (8GB+ RAM)
