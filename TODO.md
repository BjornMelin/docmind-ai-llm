# DocMind AI - Local Implementation TODO

**Source**: Based on `MASTER_REVIEW_DOCUMENT.md` critical findings and `crawled/` research, focused on **local deployment with advanced features**.

**KISS Principle**: Simple solutions that work locally, avoiding enterprise deployment complexity, fast shipping.

> **System Requirements Note:**
> The system must fully support **multimodal search and retrieval** (text + image) for hybrid and RAG pipelines, including PDF/image document processing and multimodal reranking, using Jina v4 Multimodal Embeddings.

---

## Phase 1: Critical Fixes (Application Must Start) ✅ COMPLETED

### 1.1 Application Startup Fixes

- [x] **Fix missing setup_logging() function** (utils.py) - COMPLETED

- [x] **Fix LlamaParse import error** (utils.py:72) - COMPLETED  

- [x] **Fix app.py imports and initialization** (app.py) - COMPLETED

- [x] **Remove hardcoded llama2:7b model** (app.py:206) - COMPLETED

- [x] **Fix ReActAgent tools initialization** (app.py) - COMPLETED

- [x] **Test basic application startup** - COMPLETED

### 1.2 GPU Infrastructure Setup (Research-Enhanced)

- [x] **Setup GPU Infrastructure** - COMPLETED  

- [x] **Install FastEmbed GPU Dependencies** - COMPLETED

- [x] **Implement GPU Detection and Fallback** - COMPLETED

---

## Phase 2: Core RAG Features (Make It Work)

### 2.1 Hybrid Search Implementation (From MASTER_REVIEW + Research)

- [x] **Implement sparse embeddings support** (utils.py) - COMPLETED

- [x] **Fix Qdrant hybrid search configuration** - COMPLETED

- [x] **Upgrade to BGE-Large Dense Embeddings** - COMPLETED
  - Replaced Jina v4 with BAAI/bge-large-en-v1.5 (1024D)
  - Updated vector dimensions in Qdrant setup

- [ ] **Complete SPLADE++ sparse embeddings**
  - Ensure prithvida/Splade_PP_en_v1 is properly configured
  - Test hybrid search with both dense and sparse vectors

- [ ] **Add RRF (Reciprocal Rank Fusion)**
  - Simple RRF implementation for combining dense/sparse results
  - Use research-backed weights (dense: 0.7, sparse: 0.3)
  - Implement prefetch mechanism for performance

### 2.2 ColBERT Reranking Implementation (Research-Backed)

- [ ] **Integrate ColBERT Late Interaction Model**
  - Deploy colbert-ir/colbertv2.0 via FastEmbed
  - Implement as postprocessor in query pipeline
  - Configure optimal top-k reranking (retrieve 20, rerank to 5)
  - Add performance monitoring and optimization

### 2.3 Agent System (From MASTER_REVIEW)

- [ ] **Complete ReActAgent tools configuration**
  - Ensure vector and KG query engines work
  - Test with actual document queries
  - Verify agent can use both tools properly

### 2.4 Advanced GPU Performance Optimization

- [ ] **Add GPU Performance Optimization**
  - Implement torch.compile for embedding models
  - Add mixed precision (fp16/bf16) support
  - Configure CUDA streams for parallel operations
  - Add GPU kernel optimization and profiling

---

## Phase 3: Multimodal & Knowledge Graph Features

### 3.1 Jina v4 Multimodal Embeddings (Core Feature)

- [ ] **Implement Jina v4 Multimodal Embeddings**
  - Add support for visual document processing
  - Implement text + image hybrid search
  - Add PDF with images processing pipeline
  - Configure multimodal reranking with Jina Reranker m0

### 3.2 Knowledge Graph Integration (Enhanced)

- [ ] **Create Advanced KG Query Tools**
  - Implement entity extraction and relationship mapping
  - Add semantic query expansion using KG context
  - Create specialized tools for different query types
  - Integrate with hybrid search for comprehensive retrieval

- [ ] **Test End-to-End Enhanced RAG Pipeline**
  - Verify BGE-Large + SPLADE++ + ColBERT integration
  - Test query routing and multi-stage processing
  - Validate performance improvements vs baseline
  - Run comprehensive accuracy benchmarks

### 3.3 Query Pipeline Optimization

- [ ] **Optimize Query Pipeline Architecture**
  - Implement multi-stage query processing
  - Add query complexity detection and routing
  - Configure intelligent prefetch and caching
  - Add query performance analytics

---

## Phase 4: Multi-Agent System & LangGraph (Core Feature from MASTER_REVIEW)

### 4.1 LangGraph Multi-Agent System (Research-Backed)

- [ ] **Implement LangGraph Supervisor Architecture**
  - Create supervisor agent for task delegation
  - Implement document processing specialist agent
  - Create query analysis specialist agent
  - Add summarization and synthesis specialist agent
  - Reference: `crawled/llamaindex_langgraph_integration.md` supervisor pattern

- [ ] **Add Intelligent Query Routing**
  - Implement query complexity analysis
  - Route simple queries to single-agent RAG
  - Route complex queries to multi-agent system
  - Add performance-based routing optimization

- [ ] **Create Agent Coordination System**
  - Implement agent state management with LangGraph
  - Add inter-agent communication protocols
  - Create result synthesis and validation layers
  - Add fault tolerance and recovery mechanisms

### 4.2 Agent Factory & UI Integration

- [ ] **Create agent_factory.py Module (Research-Enhanced)**
  - Implement research-backed agent specialization patterns
  - Add agent performance monitoring and optimization
  - Create agent coordination and handoff mechanisms
  - Simple cost tracking per agent (local metrics only)

- [ ] **Add intelligent multi-agent toggle to UI**
  - Checkbox to enable/disable multi-agent mode
  - Show agent coordination progress
  - Add performance metrics display
  - Keep single-agent as default for simplicity

---

## Phase 5: UI & User Experience (Make It Usable)

### 5.1 Streamlit UI Improvements (From MASTER_REVIEW)

- [ ] **Fix document upload and processing flow**
  - Ensure uploaded documents are processed correctly
  - Show processing progress to user
  - Handle errors gracefully with user feedback

- [ ] **Add basic model selection UI**
  - Let users choose between Ollama, LM Studio, Llama.cpp
  - Simple dropdown with available models
  - Test that selection actually works

- [ ] **Improve query interface**
  - Better input/output formatting
  - Show which tools the agent is using
  - Display retrieval sources and confidence
  - Show multimodal results (text + images)

---

## Phase 6: Dependencies & Configuration (Research-Enhanced)

### 6.1 Research-Backed Python & GPU Dependencies

- [ ] **Update pyproject.toml with Research-Backed Dependencies**
  - Add FastEmbed GPU acceleration packages
  - Add BGE-Large and SPLADE++ model dependencies
  - Add LangGraph for multi-agent coordination
  - Add ColBERT and multimodal processing packages
  - Remove all unused LangChain dependencies

- [ ] **Add Comprehensive GPU Dependencies**
  - Include fastembed-gpu, colbert-ai, transformers-gpu
  - Add CUDA toolkit and cuDNN requirements
  - Include GPU monitoring tools (gpustat, nvidia-ml-py3)
  - Add memory optimization packages

### 6.2 Model Configuration Updates

- [ ] **Update models.py with Advanced Settings** ✅ COMPLETED
  - Added BGE-Large and SPLADE++ model configurations
  - Added RRF fusion parameters
  - Added GPU acceleration toggles
  - Added batch processing settings

---

## Phase 7: Cleanup & Polish (Make It Ship-Ready)

### 7.1 Documentation Sync (From MASTER_REVIEW)

- [ ] **Update README.md**
  - Fix LangChain vs LlamaIndex discrepancy
  - Update installation instructions
  - Add GPU setup instructions
  - Add multimodal features documentation
  - Remove claims about unimplemented features

- [ ] **Create ADR-012: LlamaIndex Migration**
  - Document why we switched from LangChain to LlamaIndex
  - Explain current architecture decisions

- [ ] **Update or remove obsolete ADRs**
  - Archive ADR-010 (LangChain integration)
  - Update ADR-011 to reflect LangGraph multi-agent approach

### 7.2 Basic Error Handling & Logging

- [ ] **Add proper error handling**
  - Graceful failures when models unavailable
  - User-friendly error messages
  - Log errors for debugging without crashing UI

- [ ] **Implement comprehensive logging**
  - Different log levels for development vs usage
  - Log file rotation
  - Performance metrics logging

### 7.3 Performance Optimizations & Caching

- [ ] **Implement basic caching**
  - Cache embeddings for repeated documents
  - Simple file-based cache, no Redis complexity
  - Cache multimodal embeddings efficiently

---

## REMOVED FEATURES (Enterprise/Over-Engineering Only)

**Over-Engineering Removed:**

- Dynamic parameter tuning systems

**KEPT (Good for Local Users):**

- GPU optimizations and torch.compile

- Multimodal search capabilities

- ColBERT reranking

- LangGraph multi-agent system

- Advanced hybrid search with RRF

- Simple performance monitoring

- File-based caching

---

## Success Criteria (Local Focus with Advanced Features)

**Core Success Metrics:**

- ✅ **Application starts without errors**

- ✅ **Users can upload and analyze documents (including PDFs with images)**  

- ✅ **Hybrid search returns relevant results with reranking**

- ✅ **GPU acceleration works when available, fails gracefully when not**

- ✅ **Multi-agent system improves complex query handling**

- ✅ **Multimodal search works for text + image content**

- ✅ **UI is intuitive and responsive**

**Performance Goals (Enhanced for Local):**

- ⚡ **Document processing completes in reasonable time with GPU boost**

- ⚡ **Queries return results in <5 seconds with reranking**

- ⚡ **Multi-agent queries complete in <10 seconds**

- ⚡ **GPU provides 10x+ speedup when available**

- ⚡ **System works well on mid-range hardware (8GB+ RAM)**

---

## Priority Order (Updated)

1. **Phase 2.1**: Complete SPLADE++ and RRF fusion (core hybrid search)
2. **Phase 2.2**: ColBERT reranking integration (performance boost)
3. **Phase 2.3**: Complete ReActAgent tools (basic functionality)
4. **Phase 4.1**: LangGraph multi-agent system (missing core feature)
5. **Phase 3.1**: Jina v4 multimodal embeddings (core requirement)
6. **Phase 5.1**: UI improvements (usability)
7. **Phase 6.1**: Dependencies cleanup (shipping)
8. **Phase 7**: Documentation and polish

**File Coordination:**

- `utils.py`: ColBERT, multimodal, RRF fusion implementations

- `app.py`: UI improvements, multi-agent toggle, model selection

- `models.py`: Already updated with advanced settings ✅

- `agent_factory.py`: NEW - Multi-agent coordination and specialization

- `pyproject.toml`: Research-backed dependency updates

- `README.md`: Documentation updates with multimodal and multi-agent features

**Timeline Target:** 2-3 weeks for phases 1-6, shipping a comprehensive local document analysis tool with advanced RAG capabilities.
