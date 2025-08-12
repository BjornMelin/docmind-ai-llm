# Phase 3: Advanced Features (Week 3)

**Duration**: 5-7 days  

**Priority**: MEDIUM  

**Goal**: Implement advanced optimizations and production patterns

## Phase Overview

This phase builds on the foundation to add sophisticated features including UI optimization, model compilation, advanced agent patterns, and query optimization techniques.

## Prerequisites

- ✅ Phase 1 & 2 completed successfully

- ✅ Core systems operational (logging, caching, orchestration)

- ✅ Memory optimization active

- ✅ GPU optimization configured

## Tasks

### T3.1: Streamlit Fragment Optimization 🟡 MEDIUM

**Research Foundation**:

- [Infrastructure Core Research](../../../library_research/10-infrastructure_core-research.md#streamlit-fragments)

- [UI Performance Patterns](../../../library_research/10-infrastructure_core-research.json)

**Impact**: 40-60% UI render time reduction

#### Sub-task T3.1.1: Identify Heavy UI Components

**Profile**: `src/app.py` and UI components

```python
"""Profiling script to identify slow components."""
import streamlit as st
import time
from contextlib import contextmanager

@contextmanager
def profile_component(name: str):
    """Profile a UI component."""
    start = time.perf_counter()
    yield
    duration = (time.perf_counter() - start) * 1000
    
    if duration > 100:  # Log slow components
        st.warning(f"⚠️ {name}: {duration:.0f}ms")
    
    # Store in session state for analysis
    if "profile_data" not in st.session_state:
        st.session_state.profile_data = []
    
    st.session_state.profile_data.append({
        "component": name,
        "duration_ms": duration
    })

# Usage in app.py
with profile_component("document_list"):
    render_document_list(documents)

with profile_component("analytics_dashboard"):
    render_analytics()
```

**Components to Target**:

- Document lists with many items

- Analytics dashboards with charts

- Search results rendering

- File upload processing UI

- Settings panels

#### Sub-task T3.1.2: Implement Fragment Wrappers

**Update** heavy components with `@st.fragment`:

```python
"""Streamlit fragment optimization.

Research: library_research/10-infrastructure_core-research.md
40-60% render time reduction with fragments
"""
import streamlit as st
from typing import List, Dict, Any
from loguru import logger

@st.fragment
def render_document_list(documents: List[Dict[str, Any]]):
    """Render document list as independent fragment.
    
    Fragment updates independently without full page rerun
    """
    st.subheader(f"📄 Documents ({len(documents)})")
    
    # Search within fragment
    search = st.text_input("🔍 Search documents", key="doc_search")
    
    # Filter documents
    filtered = documents
    if search:
        filtered = [
            doc for doc in documents
            if search.lower() in doc["title"].lower()
        ]
    
    # Render with pagination
    items_per_page = 20
    total_pages = (len(filtered) - 1) // items_per_page + 1
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input(
            "Page", 
            min_value=1, 
            max_value=total_pages,
            key="doc_page"
        )
    
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    
    for doc in filtered[start_idx:end_idx]:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{doc['title']}**")
                st.caption(f"ID: {doc['id']} | Size: {doc['size']}")
            
            with col2:
                if st.button("View", key=f"view_{doc['id']}"):
                    st.session_state.selected_doc = doc['id']
                    st.rerun(scope="fragment")  # Only rerun fragment

@st.fragment
def render_analytics_dashboard(data: Dict[str, Any]):
    """Render analytics as cached fragment."""
    import plotly.express as px
    
    st.subheader("📊 Analytics Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", data["total_docs"])
    with col2:
        st.metric("Avg Processing Time", f"{data['avg_time']:.2f}s")
    with col3:
        st.metric("Success Rate", f"{data['success_rate']:.1%}")
    with col4:
        st.metric("Cache Hit Rate", f"{data['cache_hit_rate']:.1%}")
    
    # Charts
    if data.get("time_series"):
        fig = px.line(
            data["time_series"],
            x="timestamp",
            y="value",
            title="Processing Throughput"
        )
        st.plotly_chart(fig, use_container_width=True)

@st.fragment
def render_search_results(results: List[Dict[str, Any]], query: str):
    """Render search results as fragment."""
    st.subheader(f"🔍 Search Results for '{query}'")
    
    # Results summary
    st.info(f"Found {len(results)} results")
    
    # Render results with highlighting
    for i, result in enumerate(results, 1):
        with st.expander(f"{i}. {result['title']} (Score: {result['score']:.3f})"):
            # Highlight query terms
            content = result['content']
            for term in query.split():
                content = content.replace(
                    term,
                    f"**{term}**"
                )
            st.markdown(content)
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Copy", key=f"copy_{i}"):
                    st.write("Copied to clipboard!")
            with col2:
                if st.button("🔗 Open", key=f"open_{i}"):
                    st.session_state.selected_result = result
```

#### Sub-task T3.1.3: Add Fragment Caching

Combine fragments with caching for maximum performance:

```python
@st.fragment
@st.cache_data(ttl=300)  # Cache for 5 minutes
def render_expensive_analysis(data_id: str):
    """Cached fragment for expensive computations."""
    # Expensive computation
    result = perform_analysis(data_id)
    
    # Render result
    st.write(result)
    
    return result

# Advanced pattern: Fragment with session state
@st.fragment
def interactive_filter_panel():
    """Interactive filters that don't trigger full rerun."""
    st.sidebar.subheader("Filters")
    
    # These inputs only rerun the fragment
    date_range = st.sidebar.date_input(
        "Date Range",
        key="filter_dates"
    )
    
    categories = st.sidebar.multiselect(
        "Categories",
        options=["A", "B", "C"],
        key="filter_categories"
    )
    
    if st.sidebar.button("Apply Filters"):
        # Update main app state
        st.session_state.filters = {
            "dates": date_range,
            "categories": categories
        }
        st.rerun()  # Full rerun only when applying
```

**Success Criteria**:

- ✅ Heavy components wrapped in fragments

- ✅ 40-60% render time reduction

- ✅ Improved interactivity

---

### T3.2: torch.compile() Optimization 🟡 MEDIUM

**Research Foundation**:

- [Multimodal Processing Research](../../../library_research/10-multimodal_processing-research.md#torch-compile)

- [GPU Optimization Patterns](../../../library_research/10-llm_runtime_core-research.json)

**Impact**: 2-3x processing speed improvement

#### Sub-task T3.2.1: Identify Compilation Targets

```bash

# Find transformer models
rg "AutoModel|from_pretrained" src/ --type py
rg "transformer|bert|gpt" src/ --type py -i

# Find PyTorch model usage
rg "torch.nn.Module" src/ --type py
```

#### Sub-task T3.2.2: Apply torch.compile

**Create**: `src/utils/model_optimization.py`

```python
"""Model optimization with torch.compile.

Research: library_research/10-multimodal_processing-research.md
2-3x speed improvement with compilation
"""
import torch
from transformers import AutoModel, AutoModelForCausalLM
from typing import Optional, Dict, Any
from loguru import logger
import warnings

# Suppress compilation warnings in production
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

def load_optimized_model(
    model_name: str,
    model_type: str = "auto",
    compile_mode: str = "reduce-overhead",
    use_flash_attention: bool = True
) -> Any:
    """Load model with torch.compile optimization.
    
    Args:
        model_name: HuggingFace model name
        model_type: Model type (auto, causal_lm, embedding)
        compile_mode: Compilation mode (default, reduce-overhead, max-autotune)
        use_flash_attention: Enable Flash Attention 2
    
    Returns:
        Optimized model
    """
    logger.info(f"Loading optimized model: {model_name}")
    
    # Determine model class
    if model_type == "causal_lm":
        model_class = AutoModelForCausalLM
    else:
        model_class = AutoModel
    
    # Load model with optimizations
    load_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    
    # Add Flash Attention if supported
    if use_flash_attention and torch.cuda.is_available():
        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        except Exception as e:
            logger.warning(f"Flash Attention not available: {e}")
    
    # Load model
    model = model_class.from_pretrained(model_name, **load_kwargs)
    
    # Apply torch.compile if available (PyTorch 2.0+)
    if hasattr(torch, "compile") and torch.cuda.is_available():
        logger.info(f"Compiling model with mode: {compile_mode}")
        
        compile_kwargs = {
            "mode": compile_mode,
            "backend": "inductor",
            "fullgraph": False,  # Allow graph breaks for flexibility
        }
        
        # Compile the model
        model = torch.compile(model, **compile_kwargs)
        
        # Warmup compilation
        warmup_model(model, model_name)
        
        logger.info("Model compilation complete")
    else:
        logger.warning("torch.compile not available or no GPU")
    
    return model

def warmup_model(model: Any, model_name: str):
    """Warmup compiled model to trigger compilation.
    
    Args:
        model: Compiled model
        model_name: Model name for tokenizer
    """
    from transformers import AutoTokenizer
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create dummy input
        dummy_text = "This is a warmup input for model compilation."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Forward pass to trigger compilation
        with torch.no_grad():
            _ = model(**inputs)
        
        logger.debug("Model warmup complete")
        
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

def optimize_embedding_model(model_name: str) -> Any:
    """Optimize embedding model specifically.
    
    Args:
        model_name: Embedding model name
        
    Returns:
        Optimized embedding model
    """
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        
        # Compile with reduce-overhead for embedding models
        if hasattr(torch, "compile"):
            model = torch.compile(
                model,
                mode="reduce-overhead",
                backend="inductor"
            )
            logger.info(f"Embedding model compiled: {model_name}")
    
    return model

# Benchmark utility
def benchmark_model(model: Any, inputs: Dict[str, torch.Tensor], iterations: int = 100):
    """Benchmark model performance.
    
    Args:
        model: Model to benchmark
        inputs: Input tensors
        iterations: Number of iterations
        
    Returns:
        Average inference time in ms
    """
    import time
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(**inputs)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(**inputs)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    duration = (time.perf_counter() - start) * 1000 / iterations
    
    logger.info(f"Average inference time: {duration:.2f}ms")
    return duration
```

#### Sub-task T3.2.3: Integration with Existing Models

**Update** model loading code to use optimization:

```python
from src.utils.model_optimization import load_optimized_model

# Replace standard model loading

# OLD:

# model = AutoModel.from_pretrained("bert-base-uncased")

# NEW:
model = load_optimized_model(
    "bert-base-uncased",
    compile_mode="reduce-overhead",
    use_flash_attention=True
)
```

**Success Criteria**:

- ✅ Models compiled with torch.compile

- ✅ 2-3x speed improvement verified

- ✅ Flash Attention enabled where supported

---

### T3.3: LangGraph Supervisor Pattern 🟡 HIGH

**Research Foundation**:

- [Orchestration & Agents Research](../../../library_research/10-orchestration_agents-research.md#supervisor-patterns)

**Impact**: Complete multi-agent system with minimal code

#### Sub-task T3.3.1: Create Specialized Agents

**Create**: `src/agents/specialized/` directory with agents:

```python
"""Specialized agents for document processing.

Research: library_research/10-orchestration_agents-research.md
"""
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core import VectorStoreIndex
from typing import List, Dict, Any, Optional
from loguru import logger

def create_search_agent(index: VectorStoreIndex) -> ReActAgent:
    """Create document search specialist.
    
    Args:
        index: Vector store index
        
    Returns:
        Search agent
    """
    # Create search tool
    search_tool = QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(
            similarity_top_k=10,
            response_mode="no_text"  # Return nodes only
        ),
        name="search_documents",
        description="Search for relevant documents using semantic similarity"
    )
    
    # Create keyword search tool
    def keyword_search(query: str, limit: int = 10) -> List[Dict]:
        """Perform keyword search."""
        # Implementation would use BM25 or similar
        return [{"doc": f"Result for {query}"}]
    
    keyword_tool = FunctionTool.from_defaults(
        fn=keyword_search,
        name="keyword_search",
        description="Search using exact keyword matching"
    )
    
    # Create agent
    agent = ReActAgent.from_tools(
        [search_tool, keyword_tool],
        verbose=True,
        max_iterations=3
    )
    
    logger.info("Search agent created")
    return agent

def create_analyzer_agent(index: VectorStoreIndex) -> ReActAgent:
    """Create analysis specialist.
    
    Args:
        index: Vector store index
        
    Returns:
        Analyzer agent
    """
    # Analysis tools
    def extract_entities(text: str) -> List[str]:
        """Extract named entities from text."""
        from src.utils.spacy_memory import SpacyMemoryManager
        
        manager = SpacyMemoryManager()
        with manager.nlp.memory_zone():
            doc = manager.nlp(text)
            return [ent.text for ent in doc.ents]
    
    def sentiment_analysis(text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        # Simplified sentiment
        return {
            "positive": 0.3,
            "negative": 0.2,
            "neutral": 0.5
        }
    
    entity_tool = FunctionTool.from_defaults(
        fn=extract_entities,
        name="extract_entities",
        description="Extract named entities from text"
    )
    
    sentiment_tool = FunctionTool.from_defaults(
        fn=sentiment_analysis,
        name="analyze_sentiment",
        description="Analyze sentiment of text"
    )
    
    # Create agent
    agent = ReActAgent.from_tools(
        [entity_tool, sentiment_tool],
        verbose=True
    )
    
    logger.info("Analyzer agent created")
    return agent

def create_summary_agent(index: VectorStoreIndex) -> ReActAgent:
    """Create summarization specialist.
    
    Args:
        index: Vector store index
        
    Returns:
        Summary agent
    """
    # Summarization tool
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True
        ),
        name="summarize_content",
        description="Generate comprehensive summaries"
    )
    
    # Extractive summary tool
    def extractive_summary(text: str, num_sentences: int = 3) -> str:
        """Create extractive summary."""
        # Simple extractive summary
        sentences = text.split(". ")[:num_sentences]
        return ". ".join(sentences)
    
    extractive_tool = FunctionTool.from_defaults(
        fn=extractive_summary,
        name="extractive_summary",
        description="Create extractive summary"
    )
    
    # Create agent
    agent = ReActAgent.from_tools(
        [summary_tool, extractive_tool],
        verbose=True
    )
    
    logger.info("Summary agent created")
    return agent

def create_validator_agent() -> ReActAgent:
    """Create validation specialist.
    
    Returns:
        Validator agent
    """
    def validate_facts(claims: List[str]) -> Dict[str, bool]:
        """Validate factual claims."""
        # Simplified validation
        return {claim: True for claim in claims}
    
    def check_consistency(texts: List[str]) -> bool:
        """Check consistency across texts."""
        # Simplified consistency check
        return True
    
    fact_tool = FunctionTool.from_defaults(
        fn=validate_facts,
        name="validate_facts",
        description="Validate factual claims"
    )
    
    consistency_tool = FunctionTool.from_defaults(
        fn=check_consistency,
        name="check_consistency",
        description="Check consistency across documents"
    )
    
    agent = ReActAgent.from_tools(
        [fact_tool, consistency_tool],
        verbose=True
    )
    
    logger.info("Validator agent created")
    return agent
```

#### Sub-task T3.3.2: Implement Agent Registry

**Create**: `src/agents/agent_registry.py`

```python
"""Agent registry and management.

Central registry for all specialized agents
"""
from typing import Dict, Any, Optional
from llama_index.core.agent import BaseAgent
from src.agents.specialized import (
    create_search_agent,
    create_analyzer_agent,
    create_summary_agent,
    create_validator_agent
)
from loguru import logger

class AgentRegistry:
    """Registry for managing specialized agents."""
    
    def __init__(self, index=None):
        """Initialize agent registry.
        
        Args:
            index: Vector store index for agents
        """
        self.index = index
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all specialized agents."""
        if self.index:
            self.agents["searcher"] = create_search_agent(self.index)
            self.agents["analyzer"] = create_analyzer_agent(self.index)
            self.agents["summarizer"] = create_summary_agent(self.index)
        
        self.agents["validator"] = create_validator_agent()
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    def get_agent(self, role: str) -> Optional[BaseAgent]:
        """Get agent by role.
        
        Args:
            role: Agent role
            
        Returns:
            Agent instance or None
        """
        return self.agents.get(role)
    
    def list_agents(self) -> List[str]:
        """List available agent roles."""
        return list(self.agents.keys())
    
    def execute_task(
        self,
        role: str,
        task: str,
        context: Dict[str, Any] = None
    ) -> Any:
        """Execute task with specific agent.
        
        Args:
            role: Agent role
            task: Task description
            context: Optional context
            
        Returns:
            Agent response
        """
        agent = self.get_agent(role)
        if not agent:
            raise ValueError(f"Agent not found: {role}")
        
        # Add context to task if provided
        if context:
            task = f"Context: {context}\n\nTask: {task}"
        
        response = agent.chat(task)
        return response
```

#### Sub-task T3.3.3: Enhanced Supervisor with Handoffs

**Update**: `src/agents/supervisor.py`

```python

# Add to existing supervisor
from src.agents.agent_registry import AgentRegistry

class EnhancedDocMindSupervisor(DocMindSupervisor):
    """Enhanced supervisor with agent registry."""
    
    def __init__(self, index=None, **kwargs):
        super().__init__(**kwargs)
        self.registry = AgentRegistry(index)
    
    def _delegate_to_agent(self, role: str, task: str, state: AgentState):
        """Delegate task to specialized agent.
        
        Args:
            role: Agent role
            task: Task to execute
            state: Current state
        """
        try:
            result = self.registry.execute_task(
                role=role,
                task=task,
                context=state.get("context", {})
            )
            
            logger.info(f"Agent {role} completed task")
            return result
            
        except Exception as e:
            logger.error(f"Agent {role} failed: {e}")
            return {"error": str(e)}
```

**Success Criteria**:

- ✅ Specialized agents created

- ✅ Agent registry functional

- ✅ Supervisor can delegate tasks

---

### T3.4: ColBERT Batch Processing 🟡 MEDIUM

**Research Foundation**:

- [RAG & Reranking Research](../../../library_research/10-rag_reranking-research.md#batch-processing)

**Impact**: 2-3x throughput improvement for reranking

#### Sub-task T3.4.1: Implement Batch Reranker

**Create**: `src/services/batch_reranker.py`

```python
"""Batch processing for ColBERT reranking.

Research: library_research/10-rag_reranking-research.md
2-3x throughput with batch processing
"""
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.schema import NodeWithScore
from typing import List, Dict, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import numpy as np

class BatchReranker:
    """Batch-optimized ColBERT reranker."""
    
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        top_n: int = 5,
        batch_size: int = 32,
        max_workers: int = 4
    ):
        """Initialize batch reranker.
        
        Args:
            model_name: ColBERT model
            top_n: Number of top results
            batch_size: Batch size for processing
            max_workers: Max parallel workers
        """
        self.model_name = model_name
        self.top_n = top_n
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Initialize reranker
        self.reranker = ColbertRerank(
            model=model_name,
            top_n=top_n,
            keep_retrieval_score=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"Batch reranker initialized",
            model=model_name,
            batch_size=batch_size
        )
    
    def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[NodeWithScore]]
    ) -> List[List[NodeWithScore]]:
        """Rerank multiple queries in batch.
        
        Args:
            queries: List of queries
            documents_list: List of document lists
            
        Returns:
            Reranked documents for each query
        """
        if len(queries) != len(documents_list):
            raise ValueError("Queries and documents must have same length")
        
        results = []
        total = len(queries)
        
        # Process in batches
        for i in range(0, total, self.batch_size):
            batch_queries = queries[i:i + self.batch_size]
            batch_docs = documents_list[i:i + self.batch_size]
            
            # Parallel reranking
            batch_results = self._process_batch(batch_queries, batch_docs)
            results.extend(batch_results)
            
            # Progress logging
            processed = min(i + self.batch_size, total)
            logger.debug(f"Reranked {processed}/{total} queries")
        
        return results
    
    def _process_batch(
        self,
        queries: List[str],
        docs_list: List[List[NodeWithScore]]
    ) -> List[List[NodeWithScore]]:
        """Process a batch of queries.
        
        Args:
            queries: Batch of queries
            docs_list: Batch of document lists
            
        Returns:
            Reranked results
        """
        # Create tasks for parallel execution
        tasks = []
        for query, docs in zip(queries, docs_list):
            task = self.executor.submit(
                self._rerank_single,
                query,
                docs
            )
            tasks.append(task)
        
        # Collect results
        results = []
        for task in tasks:
            result = task.result()
            results.append(result)
        
        return results
    
    def _rerank_single(
        self,
        query: str,
        documents: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Rerank documents for single query.
        
        Args:
            query: Query text
            documents: Documents to rerank
            
        Returns:
            Reranked documents
        """
        try:
            # Set query for reranker
            self.reranker.query = query
            
            # Rerank documents
            reranked = self.reranker.postprocess_nodes(documents)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:self.top_n]  # Fallback to top-k
    
    async def arerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[NodeWithScore]]
    ) -> List[List[NodeWithScore]]:
        """Async batch reranking.
        
        Args:
            queries: List of queries
            documents_list: List of document lists
            
        Returns:
            Reranked documents
        """
        loop = asyncio.get_event_loop()
        
        # Run in executor
        result = await loop.run_in_executor(
            None,
            self.rerank_batch,
            queries,
            documents_list
        )
        
        return result
    
    def rerank_with_memory_limit(
        self,
        documents: List[NodeWithScore],
        max_memory_mb: int = 1000
    ) -> List[NodeWithScore]:
        """Rerank with memory constraints.
        
        Args:
            documents: Documents to rerank
            max_memory_mb: Max memory in MB
            
        Returns:
            Reranked documents
        """
        # Estimate memory per document
        if not documents:
            return []
        
        sample_size = min(10, len(documents))
        sample_memory = sum(
            len(doc.node.text) for doc in documents[:sample_size]
        ) / sample_size
        
        # Calculate safe batch size
        docs_per_batch = int(max_memory_mb * 1024 * 1024 / sample_memory)
        docs_per_batch = max(1, min(docs_per_batch, 100))
        
        logger.debug(f"Memory-aware batch size: {docs_per_batch}")
        
        # Process in memory-safe batches
        all_reranked = []
        
        for i in range(0, len(documents), docs_per_batch):
            batch = documents[i:i + docs_per_batch]
            reranked = self.reranker.postprocess_nodes(batch)
            all_reranked.extend(reranked)
        
        # Final rerank of combined results
        if len(all_reranked) > self.top_n:
            all_reranked = self.reranker.postprocess_nodes(all_reranked)
        
        return all_reranked[:self.top_n]
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
```

**Success Criteria**:

- ✅ Batch processing implemented

- ✅ 2-3x throughput improvement

- ✅ Memory-aware processing

---

### T3.5: QueryPipeline Integration 🟡 MEDIUM

**Research Foundation**:

- [LlamaIndex Ecosystem Research](../../../library_research/10-llamaindex_ecosystem-research.md#query-pipeline)

**Impact**: Advanced query orchestration and routing

#### Sub-task T3.5.1: Create Query Pipeline

**Create**: `src/services/query_pipeline.py`

```python
"""Advanced query pipeline with LlamaIndex.

Research: library_research/10-llamaindex_ecosystem-research.md
"""
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.query_pipeline.components import (
    InputComponent,
    FnComponent,
    RouterComponent,
    ArgPackComponent
)
from llama_index.core.response_synthesizers import TreeSummarize
from typing import Dict, Any, List
from loguru import logger

def create_advanced_pipeline(
    retriever,
    reranker,
    synthesizer=None
) -> QueryPipeline:
    """Create advanced query pipeline.
    
    Args:
        retriever: Document retriever
        reranker: Reranking component
        synthesizer: Response synthesizer
        
    Returns:
        Configured query pipeline
    """
    # Initialize pipeline
    pipeline = QueryPipeline(verbose=True)
    
    # Create synthesizer if not provided
    if synthesizer is None:
        synthesizer = TreeSummarize()
    
    # Define components
    components = {
        "input": InputComponent(),
        "query_rewriter": FnComponent(fn=rewrite_query),
        "router": RouterComponent(
            choices=["simple", "complex", "comparative"],
            selector=FnComponent(fn=classify_query)
        ),
        "retriever": retriever,
        "reranker": reranker,
        "synthesizer": synthesizer,
        "formatter": FnComponent(fn=format_response)
    }
    
    # Add components to pipeline
    pipeline.add_modules(components)
    
    # Define pipeline flow
    # Input -> Query Rewriter -> Router
    pipeline.add_link("input", "query_rewriter")
    pipeline.add_link("query_rewriter", "router")
    
    # Router -> Different paths
    pipeline.add_link("router", "retriever", condition="simple")
    pipeline.add_link("router", "retriever", condition="complex")
    pipeline.add_link("router", "retriever", condition="comparative")
    
    # Retriever -> Reranker -> Synthesizer -> Formatter
    pipeline.add_link("retriever", "reranker")
    pipeline.add_link("reranker", "synthesizer")
    pipeline.add_link("synthesizer", "formatter")
    
    logger.info("Advanced query pipeline created")
    return pipeline

def rewrite_query(query: str) -> str:
    """Rewrite query for better retrieval.
    
    Args:
        query: Original query
        
    Returns:
        Rewritten query
    """
    # Simple query expansion
    expansions = {
        "ML": "machine learning",
        "AI": "artificial intelligence",
        "NLP": "natural language processing",
        "CV": "computer vision"
    }
    
    rewritten = query
    for abbr, full in expansions.items():
        if abbr in rewritten:
            rewritten = f"{rewritten} {full}"
    
    logger.debug(f"Query rewritten: {query} -> {rewritten}")
    return rewritten

def classify_query(query: str) -> str:
    """Classify query complexity.
    
    Args:
        query: Query text
        
    Returns:
        Query classification
    """
    # Simple classification based on keywords
    if any(word in query.lower() for word in ["compare", "contrast", "versus"]):
        return "comparative"
    elif any(word in query.lower() for word in ["analyze", "explain", "why", "how"]):
        return "complex"
    else:
        return "simple"

def format_response(response: Any) -> Dict[str, Any]:
    """Format final response.
    
    Args:
        response: Raw response
        
    Returns:
        Formatted response
    """
    return {
        "answer": str(response),
        "metadata": {
            "timestamp": time.time(),
            "pipeline_version": "1.0"
        }
    }

# Specialized pipelines
def create_rag_pipeline(index) -> QueryPipeline:
    """Create RAG-specific pipeline."""
    from src.services.batch_reranker import BatchReranker
    
    pipeline = QueryPipeline()
    
    # RAG components
    retriever = index.as_retriever(similarity_top_k=20)
    reranker = BatchReranker(top_n=5)
    
    # Build pipeline
    pipeline.add_modules({
        "input": InputComponent(),
        "retriever": retriever,
        "reranker": FnComponent(
            fn=lambda nodes: reranker.rerank_single("", nodes)
        ),
        "synthesizer": TreeSummarize()
    })
    
    # Connect components
    pipeline.add_link("input", "retriever")
    pipeline.add_link("retriever", "reranker")
    pipeline.add_link("reranker", "synthesizer")
    
    return pipeline
```

**Success Criteria**:

- ✅ Query pipeline created

- ✅ Query routing implemented

- ✅ Advanced orchestration patterns

---

### T3.6: Pillow Security Upgrade 🟢 LOW

**Research Foundation**:

- [Document Ingestion Research](../../../library_research/10-document_ingestion-research.md)

**Impact**: Security patches and performance improvements

#### Sub-task T3.6.1: Test Pillow 11.x Compatibility

```bash

# Create test environment
uv venv test-pillow
source test-pillow/bin/activate  # or test-pillow\Scripts\activate on Windows

# Install Pillow 11.x
uv pip install "pillow>=11.3.0"

# Run image processing tests
uv run pytest tests/unit/test_image_processing.py -v
```

#### Sub-task T3.6.2: Update Pillow Version

**Update**: `pyproject.toml`

```toml
[project.dependencies]

# Update from ~10.4.0 to 11.3.0+
pillow = ">=11.3.0,<12.0.0"
```

```bash

# Update and verify
uv lock --upgrade-package pillow
uv sync

# Verify version
python -c "import PIL; print(f'Pillow {PIL.__version__}')"
```

#### Sub-task T3.6.3: Performance Validation

```python
"""Benchmark Pillow performance."""
import time
from PIL import Image
import numpy as np

def benchmark_pillow():
    """Benchmark image operations."""
    # Create test image
    img = Image.new("RGB", (1920, 1080), color="red")
    
    operations = {
        "resize": lambda: img.resize((640, 480)),
        "rotate": lambda: img.rotate(45),
        "convert": lambda: img.convert("L"),
        "thumbnail": lambda: img.thumbnail((200, 200))
    }
    
    results = {}
    iterations = 100
    
    for name, op in operations.items():
        start = time.perf_counter()
        for _ in range(iterations):
            _ = op()
        duration = (time.perf_counter() - start) * 1000 / iterations
        results[name] = duration
        print(f"{name}: {duration:.2f}ms")
    
    return results

# Run benchmark
results = benchmark_pillow()
```

**Success Criteria**:

- ✅ Pillow 11.x installed

- ✅ All tests passing

- ✅ No performance regression

---

## Phase 3 Validation Checklist

### UI Optimization

- [ ] Streamlit fragments implemented

- [ ] Heavy components optimized

- [ ] 40-60% render time reduction verified

### Model Optimization

- [ ] torch.compile applied to models

- [ ] Flash Attention enabled

- [ ] 2-3x speed improvement confirmed

### Agent System

- [ ] Specialized agents created

- [ ] Agent registry functional

- [ ] Supervisor delegation working

### Query Processing

- [ ] Batch reranking implemented

- [ ] Query pipeline created

- [ ] Advanced routing functional

### Dependencies

- [ ] Pillow upgraded to 11.x

- [ ] Security patches applied

- [ ] Performance validated

## Performance Benchmarks

Create `benchmark_phase3.py`:

```python
import asyncio
import time
from src.utils.model_optimization import load_optimized_model, benchmark_model
from src.services.batch_reranker import BatchReranker
from src.agents.agent_registry import AgentRegistry

async def benchmark_phase3():
    print("=== Phase 3 Validation ===\n")
    
    # Test model optimization
    print("Testing torch.compile optimization...")
    model = load_optimized_model(
        "bert-base-uncased",
        compile_mode="reduce-overhead"
    )
    
    # Create dummy input
    import torch
    dummy_input = {
        "input_ids": torch.randint(0, 1000, (1, 128)),
        "attention_mask": torch.ones(1, 128)
    }
    
    if torch.cuda.is_available():
        dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
    
    avg_time = benchmark_model(model, dummy_input, iterations=50)
    print(f"Model inference: {avg_time:.2f}ms average")
    
    # Test batch reranking
    print("\nTesting batch reranking...")
    reranker = BatchReranker(batch_size=16)
    
    # Create test data
    queries = ["test query"] * 32
    docs = [[NodeWithScore(node=Node(text=f"Doc {i}"), score=0.9)] * 10 
            for _ in range(32)]
    
    start = time.perf_counter()
    results = reranker.rerank_batch(queries, docs)
    duration = (time.perf_counter() - start) * 1000
    
    print(f"Batch reranking (32 queries): {duration:.2f}ms")
    print(f"Throughput: {32000/duration:.1f} queries/sec")
    
    # Test agent registry
    print("\nTesting agent registry...")
    registry = AgentRegistry()
    agents = registry.list_agents()
    print(f"Available agents: {agents}")
    
    print("\n✅ Phase 3 validation complete")

# Run benchmark
asyncio.run(benchmark_phase3())
```

## Next Steps

After completing Phase 3:
1. Run validation benchmarks
2. Document advanced features
3. Prepare for production deployment
4. Proceed to [Phase 4: Production Readiness](./04-phase-production.md)

## Rollback Procedures

```bash

# Streamlit fragments
git checkout -- src/app.py

# Model optimization
git checkout -- src/utils/model_optimization.py

# Agent system
git checkout -- src/agents/

# Pillow downgrade
uv add "pillow~=10.4.0"
uv lock && uv sync
```

Time to rollback: <10 minutes for any component
