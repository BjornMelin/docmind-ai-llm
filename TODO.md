# DocMind AI - Local Implementation TODO

**Source**: Based on full conversation review, final decisions, architecture (e.g., LlamaIndex pipelines/retrievers, LangGraph supervisor with planning/Send, Unstructured parsing, SQLite/diskcache caching), and current codebase state (e.g., partial hybrid/SPLADE++, custom multimodal to evolve, KG commented—enable, no pipeline/chunking/caching yet). Incorporates critical findings from `MASTER_REVIEW_DOCUMENT.md` and `crawled/` research.

**KISS Principle**: Simple, library-first solutions (e.g., LlamaIndex QueryPipeline over custom, UnstructuredReader for parsing) that work locally/offline, avoiding complexity (no distributed/Redis—MVP local multi-process with SQLite WAL/diskcache locks). Fast shipping: 1-week MVP on Groups 1-2.

> **System Requirements Note**: Fully offline/local (no API keys, e.g., Ollama for LLM/VLM, HuggingFace for Jina). Use AppSettings (models.py) for all configs (e.g., chunk_size=1024, chunk_overlap=200, gpu_acceleration). Library versions from pyproject.toml (e.g., llama-index==0.12.52, langgraph==0.5.4, unstructured[all-docs]==0.15.13, diskcache==5.6.3—add if missing). Must support multimodal search and retrieval (text + image) for hybrid and RAG pipelines, including PDF/image processing and multimodal reranking.

---

## Phase 1: Critical Fixes (Application Must Start) ✅ COMPLETED

- [x] All tasks completed per codebase—app starts, GPU detection/fallback implemented in utils.py.

  - [x] Fix missing setup_logging() function (utils.py)
  - [x] Fix LlamaParse import error (utils.py:72)
  - [x] Fix app.py imports and initialization (app.py)
  - [x] Remove hardcoded llama2:7b model (app.py:206)
  - [x] Fix ReActAgent tools initialization (app.py)
  - [x] Test basic application startup
  - [x] Setup GPU Infrastructure
  - [x] Install FastEmbed GPU Dependencies
  - [x] Implement GPU Detection and Fallback

---

## Group 1: Core Retrieval Foundation (Hybrid/SPLADE++/KG/GPU - Partial; Refine/Test Together - Week 1 MVP)

**Priority**: High  
**Deadline**: End of Week 1  
Prioritize: End-to-end hybrid retrieval with fixes/enhancements; Test offline in tests/test_hybrid_search.py/test_real_validation.py.

- [x] **Task 0: Upgrade to BGE-Large Dense Embeddings** (Completed from original Phase 2.1)
  - [x] Replaced Jina v4 with BAAI/bge-large-en-v1.5 (1024D)
  - [x] Updated vector dimensions in Qdrant setup

- [ ] **Task 1: Complete SPLADE++ Sparse Embeddings** (Implemented; Refine/Test)
  - [x] Subtask 1.0: Implement sparse embeddings support (Completed from original Phase 2.1)
  - [x] Subtask 1.0.1: Fix Qdrant hybrid search configuration (Completed from original Phase 2.1)
  - [ ] Subtask 1.1: Configure and Test SPLADE++ Model (Implemented; Fix Typo/Test)
    - **Instructions**: In models.py, fix sparse_embedding_model to "prithvida/Splade_PP_en_v1". In utils.py, use AppSettings.sparse_embedding_model for SparseTextEmbedding init. Add test in tests/test_embeddings.py: def test_splade_expansion(): sparse_model = SparseTextEmbedding(AppSettings.sparse_embedding_model); emb = list(sparse_model.embed(["library"]))[0]; assert any(i in emb.indices for i in [vocab for "software"]); logger.info(emb.values). Run pytest tests/test_embeddings.py.
    - **Libraries**: fastembed==0.7.1
    - **Classes/Functions/Features**: SparseTextEmbedding (native term expansion); embed() (sparse vectors).
  - [ ] Subtask 1.2: Integrate into Hybrid Search (Implemented; Evolve to HybridFusionRetriever)
    - **Instructions**: In utils.py create_index, from llama_index.core.retrievers import HybridFusionRetriever; retriever = HybridFusionRetriever(dense=FastEmbedEmbedding(AppSettings.dense_embedding_model, dim=AppSettings.dense_embedding_dimension or 1024), sparse=SparseTextEmbedding(AppSettings.sparse_embedding_model), fusion_type="rrf", alpha=AppSettings.rrf_fusion_alpha or 0.7, prefetch_k=AppSettings.prefetch_factor or 2) with QdrantVectorStore. Add test in tests/test_hybrid_search.py: def test_hybrid_fusion(): retriever = HybridFusionRetriever(...); results = retriever.retrieve("test query"); assert len(results) > 0; assert fusion_scores descending; pytest.mark.parametrize("alpha", [0.5, 0.7]).
    - **Libraries**: qdrant-client==1.15.0, llama-index==0.12.52
    - **Classes/Functions/Features**: HybridFusionRetriever (alpha tuning, offline hybrid); retrieve() (fusion).

- [x] **Task 1.5: Add RRF (Reciprocal Rank Fusion)** (Completed from original Phase 2.1)
  - [x] Simple RRF implementation for combining dense/sparse results
  - [x] Use research-backed weights (dense: 0.7, sparse: 0.3)
  - [x] Implement prefetch mechanism for performance
  - [x] Native Qdrant RRF fusion with optimized prefetch
  - [x] Configuration in models.py (0.7/0.3)
  - [x] Seamless LlamaIndex hybrid_alpha calculation

- [ ] **Task 2: Complete ReActAgent Tools Configuration** (Implemented; Minor Verification)
  - [ ] Subtask 2.1: Verify Vector and KG Query Engines (Partial; Enable KG)
    - **Instructions**: In utils.py create_tools_from_index, uncomment/enable KGIndex.from_documents(docs, llm=Ollama(AppSettings.default_model), extractor=spaCy load_model("en_core_web_sm")). Add QueryEngineTool for KG. Test in tests/test_utils.py: def test_kg_tools(): tools = create_tools_from_index(index); assert any("knowledge_graph" in t.metadata.name for t in tools); result = tools[1].query_engine.query("entities relations"); assert "entity" in str(result).
    - **Libraries**: llama-index==0.12.52, spacy==3.8.7
    - **Classes/Functions/Features**: KnowledgeGraphIndex.from_documents (spaCy NER for offline extraction); load_model("en_core_web_sm").
  - [ ] Subtask 2.2: Add Tool Usage Verification (Implemented; No Change)
    - **Instructions**: Already verbose in app.py. Ensure tools use AppSettings (e.g., similarity_top_k=AppSettings.reranking_top_k or 5). Run existing tests/test_agents.py.
    - **Libraries**: llama-index==0.12.52
    - **Classes/Functions/Features**: ReActAgent.from_tools; chat().

- [ ] **Task 3: Add Advanced GPU Performance Optimization** (Partial; Add Missing)
  - [ ] Subtask 3.1: Implement Mixed Precision and Torch Compile (Partial; Add Compile)
    - **Instructions**: In utils.py embed_model init, if AppSettings.gpu_acceleration and torch.cuda.is_available(): embed_model = torch.compile(FastEmbedEmbedding(..., torch_dtype="bf16")). Test in tests/test_performance_integration.py: def test_gpu_compile(): if torch.cuda.is_available(): emb = embed_model.embed(["test"]); assert len(emb) > 0; measure latency < cpu.
    - **Libraries**: fastembed==0.7.1, torch==2.7.1
    - **Classes/Functions/Features**: TextEmbedding (bf16 support); torch.compile (dynamic=True for speedup).
  - [ ] Subtask 3.2: Add CUDA Streams and Profiling (Not Implemented; Add)
    - **Instructions**: In utils.py create_index_async, if AppSettings.gpu_acceleration: with torch.cuda.Stream(): index = await ...; If AppSettings.debug_mode: with torch.profiler.profile() as p: ...; p.export_chrome_trace("trace.json"). Test in tests/test_performance_integration.py: def test_cuda_streams(): latency = measure_latency(create_index_async(docs)); assert latency < threshold; if debug: assert os.path.exists("trace.json").
    - **Libraries**: torch==2.7.1
    - **Classes/Functions/Features**: cuda.Stream (parallel ops); profiler.profile (trace for bottlenecks).

- [x] **Task 4: Integrate ColBERT Late Interaction Model** (Completed from original Phase 2.2)
  - [x] Deploy colbert-ir/colbertv2.0 via FastEmbed
  - [x] Implement as postprocessor in query pipeline
  - [x] Configure optimal top-k reranking (retrieve 20, rerank to 5)
  - [x] Add performance monitoring and optimization

- [ ] **Task 5: Create Advanced KG Query Tools** (Not Implemented; Enable/Integrate)
  - [ ] Subtask 5.1: Implement Entity Extraction and Mapping (Not Implemented; Add)
    - **Instructions**: In utils.py create_index, kg_index = KnowledgeGraphIndex.from_documents(docs, extractor=spaCy load_model("en_core_web_sm")). Integrate in create_tools_from_index as QueryEngineTool. Use AppSettings for configs (e.g., max_entities=AppSettings.new_max_entities or 50).
    - **Libraries**: spacy==3.8.7, llama-index==0.12.52
    - **Classes/Functions/Features**: Pipeline (add_pipe("ner")); KnowledgeGraphIndex.from_documents (spaCy extractor, offline).
  - [ ] Subtask 5.2: Integrate with Hybrid Search and Test Pipeline (Partial; Test)
    - **Instructions**: In create_tools_from_index, add KG tool to hybrid pipeline (e.g., in HybridFusionRetriever). Benchmark in tests/test_hybrid_search.py: def test_kg_hybrid(): results = kg_query_engine.query("relations"); assert "entity" in str(results); measure recall > baseline.
    - **Libraries**: llama-index==0.12.52
    - **Classes/Functions/Features**: QueryEngineTool; as_query_engine (hybrid integration).

---

## Group 2: Multimodal Core (Offline Parsing/Embeddings - Partial/Custom; Evolve to Unstructured/Jina v4 - Week 1 MVP)

**Priority**: High  
**Deadline**: End of Week 1

- [ ] **Task 4: Implement Jina v4 Multimodal Embeddings** (Partial; Refine to Final Offline)
  - [ ] Subtask 4.1: Add Jina v4 Support for Text+Image (Partial; Switch to Unstructured/HuggingFace)
    - **Instructions**: In utils.py load_documents_llama, from llama_index.readers.unstructured import UnstructuredReader; reader = UnstructuredReader(); elements = reader.load_data(file_path, strategy=AppSettings.parse_strategy or "hi_res"); docs = [Document.from_element(e) for e in elements]. Embed with HuggingFaceEmbedding(model_name="jinaai/jina-embeddings-v4", dimensions=AppSettings.dense_embedding_dimension or 512 for MRL efficiency, task="retrieval.passage", quantization_config=BitsAndBytesConfig(load_in_8bit=True) if AppSettings.enable_quantization else None, device="cuda" if AppSettings.gpu_acceleration else "cpu"). Test in tests/test_utils.py: def test_unstructured_jina(): docs = load_documents_llama([pdf_path]); assert any("image" in d.metadata for d in docs); emb = embed_model.embed(docs[0].text); assert len(emb) == 512.
    - **Libraries**: llama-index-embeddings-huggingface>=0.5.5, transformers==4.53.3 (add unstructured[all-docs]==0.15.13 to deps)
    - **Classes/Functions/Features**: UnstructuredReader (local hi_res parsing for text/tables/images); HuggingFaceEmbedding (local); AutoModel.from_pretrained (int8 quant).
  - [ ] Subtask 4.2: Configure Multimodal Reranking and PDF Processing (Not Implemented; Add)
    - **Instructions**: In utils.py create_index, multimodal_index = MultiModalVectorStoreIndex.from_documents(docs, storage_context=StorageContext.from_defaults(vector_store=QdrantVectorStore(...)), image_embed_model=embed_model). Add ColbertRerank(postprocessor, top_n=AppSettings.reranking_top_k or 5). Test offline in tests/test_real_validation.py: def test_multimodal_offline(): index = create_index(docs); results = index.as_query_engine().query("describe image"); assert "visual" in str(results).
    - **Libraries**: llama-index==0.12.52, transformers==4.53.3
    - **Classes/Functions/Features**: MultiModalVectorStoreIndex (text+image+table, local VLM like LLaVA via Ollama); ColbertRerank (late-interaction rerank).

---

## Group 3: Query Efficiency (Pipeline Optimizations - Not Implemented; Week 2)

**Priority**: Medium  
**Deadline**: End of Week 2

- [ ] **Task 6: Optimize Query Pipeline Architecture** (Not Implemented; Add Chunking/Pipeline)
  - [ ] Subtask 6.1: Implement Multi-Stage Query Processing (Not Implemented; Add)
    - **Instructions**: In utils.py create_index, from llama_index.core import IngestionPipeline; from llama_index.core.node_parser import SentenceSplitter, MetadataExtractor; pipeline = IngestionPipeline(transformations=[SentenceSplitter(chunk_size=AppSettings.chunk_size or 1024, chunk_overlap=AppSettings.chunk_overlap or 200), MetadataExtractor()]); nodes = pipeline.run(documents=docs); index = VectorStoreIndex(nodes). Build QueryPipeline: from llama_index.core.query_pipeline import QueryPipeline; from llama_index.postprocessor import ColbertRerank; from utils import CustomJinaReranker; qp = QueryPipeline(chain=[retriever, ColbertRerank(top_n=AppSettings.reranking_top_k or 5), CustomJinaReranker(model_name="jinaai/jina-reranker-m0", top_n=AppSettings.reranking_top_k or 5), synthesizer], async_mode=True, parallel=True). Use in create_tools_from_index as qp.as_query_engine(). Test in tests/test_performance_integration.py: def test_pipeline_latency(): qp.run("query"); assert latency < threshold.
    - **Libraries**: llama-index==0.12.52 (add diskcache==5.6.3 to deps)`
    - **Classes/Functions/Features**: IngestionPipeline (transformations for chunking/extraction); SentenceSplitter (semantic); MetadataExtractor (entities); QueryPipeline (chain/async/parallel/prefetch); Cache (ttl=3600 via diskcache).
    - **Note**: Implement `CustomJinaReranker` in utils.py to interface with "jinaai/jina-reranker-m0" for multimodal reranking.
  - [ ] Subtask 6.2: Add Analytics and Routing (Not Implemented; Add)
    - **Instructions**: Use LangGraph for routing (integrate with QueryPipeline); log metrics via callbacks. Test in tests/test_performance_integration.py.
    - **Libraries**: langgraph==0.5.4
    - **Classes/Functions/Features**: StateGraph (add_node/compile).

---

## Group 4: Multi-Agent System (LangGraph Agents - Implemented; Evolve with Research - Week 2)

**Priority**: Medium  
**Deadline**: End of Week 2

- [ ] **Task 7: Implement LangGraph Supervisor Architecture** (Implemented; Evolve to Advanced)
  - [ ] Subtask 7.1: Create Supervisor and Specialist Agents (Implemented; Add Custom/Offline)
    - **Instructions**: In agent_factory.py, from langgraph.prebuilt import create_react_agent; workers = [create_react_agent(llm=Ollama(AppSettings.default_model), tools=tools_from_utils, state_modifier="RAG supervisor prompt for routing") for_ in ["doc", "kg", "multimodal"]]. Evolve StateGraph with supervisor node (prompt for RAG decisions). Test in tests/test_agents.py: def test_supervisor_routing(): graph = workflow.compile(); output = graph.invoke({"messages": [{"content": "query"}]}); assert "routed" in output.
    - **Libraries**: langgraph==0.5.4
    - **Classes/Functions/Features**: create_react_agent (workers with local LLM/tools); SupervisorAgent (event-driven StateGraph); add_edge (handoffs via Send API).
  - [ ] Subtask 7.2: Add Query Routing and Coordination (Implemented; Add Persistence/Human-in-Loop)
    - **Instructions**: Already in routing_logic. Add from langgraph.checkpoint import MemorySaver; checkpointer=MemorySaver(); workflow.compile(checkpointer=checkpointer). Add interrupt_before=["tool_call"] for human-in-loop (pause/resume via UI toggle). Test in tests/test_agents.py: def test_persistence_interrupt(): state = graph.invoke(input, config={"thread_id": "1"}); resumed = graph.invoke(None, config={"thread_id": "1"}); assert resumed["human_approved"].
    - **Libraries**: langgraph==0.5.4
    - **Classes/Functions/Features**: MemorySaver (persistence, offline); interrupt_before/interrupt_after (human-in-loop); AgentExecutor (recovery/handoffs).

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
    - **Instructions**: In app.py upload_section, with st.status("Processing..."): try: docs = load_documents_llama(...); except Exception as e: st.error(f"Error: {e}"); logger.error(e); return. Add st.progress for steps. Test in tests/test_app.py: def test_upload_error(): app.file_uploader.upload(invalid_file); app.run(); assert "Error" in app.error.value.
    - **Libraries**: streamlit==1.47.1
    - **Classes/Functions/Features**: st.status; st.error; st.progress.
  - [ ] Subtask 9.2: Add Model Selection and Query Interface (Implemented; Add Display)
    - **Instructions**: Already dropdown; add st.markdown for tool/sources (e.g., results.sources). Test in tests/test_app.py.
    - **Libraries**: streamlit==1.47.1
    - **Classes/Functions/Features**: st.selectbox; st.markdown.
  - [ ] Subtask 9.3: Improve Query Interface (From original Phase 5.1)
    - **Instructions**: Enhance input/output formatting, show which tools the agent is using, display retrieval sources and confidence, show multimodal results (text + images).

- [ ] **Task 10: Update Dependencies & Configuration** (Implemented; Minor Add)
  - [ ] Subtask 10.1: Update pyproject.toml (Implemented; Add Deps)
    - **Instructions**: Add unstructured[all-docs]==0.15.13, diskcache==5.6.3; remove unused. Test in tests/test_real_validation.py: def test_deps_offline(): import unstructured; assert unstructured.partition_pdf("test.pdf").
    - **Libraries**: All pinned.
    - **Classes/Functions/Features**: N/A.
  - [ ] Subtask 10.2: Update models.py (Implemented; Add Chunk Settings)
    - **Instructions**: Add chunk_size=1024, chunk_overlap=200 to AppSettings. Test in tests/test_models.py: def test_chunk_settings(): settings = AppSettings(); assert settings.chunk_size == 1024.
    - **Libraries**: pydantic==2.11.7
    - **Classes/Functions/Features**: BaseSettings (validation).

- [ ] **Task 11: Cleanup & Polish** (Not Implemented; Add Tests)
  - [ ] Subtask 11.1: Update README and ADRs (Not Implemented; Update)
    - **Instructions**: Create ADR-016/update ADR-011. Add to README: "Offline Multimodal: Unstructured + Jina v4". Update installation instructions, GPU setup, and multimodal features documentation from original Phase 7.1.
    - **Libraries**: N/A.
    - **Classes/Functions/Features**: N/A.
  - [ ] Subtask 11.2: Add Error Handling, Logging, Caching (Partial; Add Caching/Tests)
    - **Instructions**: Enhance try/except; add diskcache to embeds (e.g., @cache(ttl=3600) on embed functions). Add tests in tests/test_utils.py (Unstructured), tests/test_embeddings.py (dims/chunking), tests/test_real_validation.py (offline checks). Include logging with different levels and file rotation from original Phase 7.2.
    - **Libraries**: diskcache==5.6.3
    - **Classes/Functions/Features**: Cache (persistence).

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

- **Initial release (1 week)**: Groups 1-2 (core retrieval and multimodal—test local concurrent in tests/test_performance_integration.py)
- **Future phases (2-4 weeks)**: Groups 3-5
- **Roadmap**: If feedback needs distributed, add Redis toggle later. Avoid overengineering—focus local.

---

## Success Criteria (Local Focus with Advanced Features)

**Core Success Metrics:**

- [x] Application starts without errors
- [x] Users can upload and analyze documents (including PDFs with images)
- [x] Hybrid search returns relevant results with reranking
- [x] GPU acceleration works when available, fails gracefully when not
- [ ] Multi-agent system improves complex query handling
- [ ] Multimodal search works for text + image content
- [ ] UI is intuitive and responsive

**Performance Goals (Enhanced for Local):**

- ⚡ Document processing completes in reasonable time with GPU boost
- ⚡ Queries return results in <5 seconds with reranking
- ⚡ Multi-agent queries complete in <10 seconds
- ⚡ GPU provides 10x+ speedup when available
- ⚡ System works well on mid-range hardware (8GB+ RAM)
