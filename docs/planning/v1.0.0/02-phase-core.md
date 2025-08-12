# Phase 2: Core Optimizations (Week 2)

**Duration**: 5-7 days  

**Priority**: HIGH  

**Goal**: Implement core performance optimizations and library-first patterns

## Phase Overview

Building on Phase 1 foundations, this phase focuses on memory optimization, performance improvements, and modern orchestration patterns that provide 2-3x speedups and 40-60% memory reduction.

## Prerequisites

- ✅ Phase 1 completed successfully

- ✅ Dependencies cleaned up (55+ packages removed)

- ✅ CUDA optimization configured

- ✅ LlamaIndex Settings migrated

## Tasks

### T2.1: Structured JSON Logging Implementation 🟡 HIGH

**Research Foundation**:

- [Infrastructure Core Research](../../../library_research/10-infrastructure_core-research.md)

- [Logging Patterns JSON](../../../library_research/10-infrastructure_core-research.json)

**Impact**: 50-70% debugging efficiency improvement

#### Sub-task T2.1.1: Configure Loguru for JSON Output

**Create**: `src/utils/logging_config.py`

```python
"""Structured logging configuration with loguru.

Research: library_research/10-infrastructure_core-research.md
50-70% debugging efficiency improvement with structured logs
"""
from loguru import logger
import sys
import json
from typing import Any, Dict
import os

def configure_logging(
    level: str = "INFO", 
    json_output: bool = None,
    log_file: str = None
):
    """Configure structured logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: Force JSON output (auto-detected if None)
        log_file: Optional log file path
    """
    # Auto-detect based on environment
    if json_output is None:
        json_output = os.getenv("ENVIRONMENT") == "production"
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    if json_output:
        # JSON format for production
        logger.add(
            sys.stdout,
            format="{message}",
            serialize=True,
            level=level,
            filter=lambda record: record["extra"].get("name") != "metrics"
        )
    else:
        # Human-readable for development
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level=level,
            colorize=True
        )
    
    # File handler if specified
    if log_file:
        logger.add(
            log_file,
            format="{message}",
            serialize=True,
            rotation="100 MB",
            retention="7 days",
            compression="gz",
            level=level
        )
    
    # Add correlation ID support
    logger.configure(
        extra={"correlation_id": None, "user_id": None}
    )
    
    logger.info("Logging configured", level=level, json_output=json_output)

def get_logger(name: str = None):
    """Get a contextualized logger.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Contextualized logger instance
    """
    return logger.bind(name=name or "app")
```

**Validation**:
```python

# Test logging configuration
from src.utils.logging_config import configure_logging, get_logger

# Development mode
configure_logging(level="DEBUG", json_output=False)
log = get_logger(__name__)
log.info("Development logging test", user="test_user")

# Production mode
configure_logging(level="INFO", json_output=True)
log.info("Production logging test", metric_value=42)
```

**Success Criteria**:

- ✅ JSON logs in production

- ✅ Readable logs in development

- ✅ Correlation ID support

#### Sub-task T2.1.2: Add Context to Log Messages

**Update**: All files with logging

Replace basic logging with context-rich messages:

```python
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# OLD: print(f"Processing {doc_id}")

# NEW: Context-rich logging
logger.info("Processing document", 
    doc_id=doc_id, 
    doc_type=doc_type,
    doc_size_bytes=len(doc_content)
)

# Error logging with full context
try:
    result = process_document(doc)
except Exception as e:
    logger.error("Failed to process document",
        doc_id=doc.id,
        error_type=type(e).__name__,
        error_message=str(e),
        traceback=True  # Include full traceback
    )
    raise
```

**Files to Update**:

- `src/services/*.py` - Service layer logging

- `src/agents/*.py` - Agent logging

- `src/utils/*.py` - Utility logging

**Success Criteria**:

- ✅ All logs include relevant context

- ✅ Errors include traceback

- ✅ Searchable JSON fields

#### Sub-task T2.1.3: Implement Performance Logging

**Create**: `src/utils/performance_logger.py`

```python
"""Performance logging utilities.

Research: Structured logging provides 50-70% debugging efficiency
"""
from contextlib import contextmanager
from loguru import logger
import time
from typing import Any, Dict, Optional
import psutil
import torch

@contextmanager
def log_performance(
    operation: str,
    log_memory: bool = False,
    log_gpu: bool = False,
    **context
):
    """Context manager for performance logging.
    
    Usage:
        with log_performance("embedding_generation", doc_count=100):
            embeddings = generate_embeddings(documents)
    """
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024 if log_memory else None
    start_gpu = torch.cuda.memory_allocated() / 1024 / 1024 if log_gpu and torch.cuda.is_available() else None
    
    try:
        yield
        
        # Calculate metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        metrics = {
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            "status": "success",
            **context
        }
        
        if log_memory:
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            metrics["memory_delta_mb"] = round(end_memory - start_memory, 2)
        
        if log_gpu and torch.cuda.is_available():
            end_gpu = torch.cuda.memory_allocated() / 1024 / 1024
            metrics["gpu_memory_delta_mb"] = round(end_gpu - start_gpu, 2)
        
        logger.info(f"{operation} completed", **metrics)
        
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        logger.error(f"{operation} failed",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            status="error",
            error=str(e),
            **context
        )
        raise

class PerformanceTracker:
    """Track performance across multiple operations."""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(time.time())
        self.metrics = []
    
    def track(self, operation: str, duration_ms: float, **metadata):
        """Track a performance metric."""
        metric = {
            "session_id": self.session_id,
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
            **metadata
        }
        self.metrics.append(metric)
        logger.debug("Performance tracked", **metric)
    
    def summary(self):
        """Get performance summary."""
        if not self.metrics:
            return {}
        
        total_time = sum(m["duration_ms"] for m in self.metrics)
        
        return {
            "session_id": self.session_id,
            "total_operations": len(self.metrics),
            "total_time_ms": round(total_time, 2),
            "average_time_ms": round(total_time / len(self.metrics), 2),
            "operations": [m["operation"] for m in self.metrics]
        }
```

**Validation**:
```python
from src.utils.performance_logger import log_performance, PerformanceTracker

# Test performance logging
tracker = PerformanceTracker("test_session")

with log_performance("test_operation", log_memory=True, items=100):
    time.sleep(0.1)  # Simulate work

print(tracker.summary())
```

**Success Criteria**:

- ✅ Performance metrics captured

- ✅ Memory tracking available

- ✅ GPU metrics when available

---

### T2.2: spaCy Memory Zone Implementation 🟡 HIGH

**Research Foundation**:

- [Multimodal Processing Research](../../../library_research/10-multimodal_processing-research.md)

- [Memory Optimization Patterns](../../../library_research/10-multimodal_processing-research.json)

**Impact**: 40-60% memory reduction in NLP processing

#### Sub-task T2.2.1: Create spaCy Memory Manager

**Create**: `src/utils/spacy_memory.py`

```python
"""Memory-efficient spaCy processing.

Research: library_research/10-multimodal_processing-research.md
40-60% memory reduction with memory_zone() context manager
"""
import spacy
from typing import Iterator, List, Dict, Any, Optional
from loguru import logger
import gc

class SpacyMemoryManager:
    """Manage spaCy processing with automatic memory cleanup."""
    
    def __init__(
        self, 
        model_name: str = "en_core_web_sm",
        disable_components: List[str] = None,
        batch_size: int = 100
    ):
        """Initialize spaCy with memory optimization.
        
        Args:
            model_name: spaCy model to load
            disable_components: Components to disable for speed
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Disable unnecessary components for speed
        disable = disable_components or ["parser", "ner"]
        self.nlp = spacy.load(model_name, disable=disable)
        
        # Configure for efficiency
        self.nlp.max_length = 1_000_000  # 1M chars
        
        # Add doc_cleaner component
        self._add_doc_cleaner()
        
        logger.info(f"SpaCy memory manager initialized",
            model=model_name,
            disabled=disable,
            batch_size=batch_size
        )
    
    def _add_doc_cleaner(self):
        """Add doc_cleaner component for GPU memory optimization."""
        if "doc_cleaner" not in self.nlp.pipe_names:
            @spacy.Language.component("doc_cleaner")
            def doc_cleaner(doc):
                """Clean up tensor data after processing."""
                # Clear tensor to free GPU memory
                doc.tensor = None
                
                # Clear other memory-intensive attributes
                for token in doc:
                    token.tensor = None
                
                return doc
            
            # Add at the end of pipeline
            self.nlp.add_pipe("doc_cleaner", last=True)
            logger.debug("doc_cleaner component added")
    
    def process_batch(
        self, 
        texts: List[str],
        extract_entities: bool = True,
        extract_tokens: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """Process texts with automatic memory cleanup.
        
        Args:
            texts: List of texts to process
            extract_entities: Extract named entities
            extract_tokens: Extract tokens
            
        Yields:
            Processed document information
        """
        total_texts = len(texts)
        processed = 0
        
        # Process in batches with memory_zone
        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Use memory_zone for automatic cleanup
            with self.nlp.memory_zone():
                for doc in self.nlp.pipe(batch, batch_size=self.batch_size):
                    result = {
                        "text": doc.text[:1000],  # Limit stored text
                        "num_tokens": len(doc),
                        "num_sentences": len(list(doc.sents)) if doc.has_annotation("SENT_START") else 0
                    }
                    
                    if extract_entities and doc.has_annotation("ENT_IOB"):
                        result["entities"] = [
                            {"text": ent.text, "label": ent.label_}
                            for ent in doc.ents
                        ]
                    
                    if extract_tokens:
                        result["tokens"] = [
                            {"text": token.text, "pos": token.pos_}
                            for token in doc[:100]  # Limit to first 100 tokens
                        ]
                    
                    yield result
                    processed += 1
                    
                    # Periodic logging
                    if processed % 100 == 0:
                        logger.debug(f"Processed {processed}/{total_texts} texts")
            
            # Force garbage collection after each batch
            gc.collect()
    
    def process_large_document(
        self,
        text: str,
        chunk_size: int = 10000
    ) -> List[Dict[str, Any]]:
        """Process large document in chunks.
        
        Args:
            text: Large text to process
            chunk_size: Size of each chunk in characters
            
        Returns:
            List of processed chunks
        """
        chunks = []
        
        # Split into chunks
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            
            # Process chunk with memory cleanup
            with self.nlp.memory_zone():
                doc = self.nlp(chunk_text)
                
                chunks.append({
                    "chunk_index": i // chunk_size,
                    "start_char": i,
                    "end_char": min(i + chunk_size, len(text)),
                    "entities": [
                        {"text": ent.text, "label": ent.label_}
                        for ent in doc.ents
                    ] if doc.has_annotation("ENT_IOB") else [],
                    "num_tokens": len(doc)
                })
        
        logger.info(f"Processed large document",
            total_chars=len(text),
            num_chunks=len(chunks)
        )
        
        return chunks
```

**Success Criteria**:

- ✅ Memory usage stays constant during batch processing

- ✅ GPU memory freed after each document

- ✅ 40-60% memory reduction verified

#### Sub-task T2.2.2: Integrate with Document Processing

**Update**: `src/services/document_processor.py`

```python
from src.utils.spacy_memory import SpacyMemoryManager
from src.utils.performance_logger import log_performance
from typing import List, Dict, Any

class DocumentProcessor:
    """Process documents with memory-efficient NLP."""
    
    def __init__(self):
        # Initialize memory-efficient spaCy
        self.nlp_manager = SpacyMemoryManager(
            model_name="en_core_web_sm",
            disable_components=["parser"],  # Keep NER
            batch_size=50
        )
    
    def process_documents(
        self,
        documents: List[str]
    ) -> List[Dict[str, Any]]:
        """Process documents with NLP.
        
        OLD: Basic spaCy processing with memory issues
        NEW: Memory-efficient processing with 40-60% reduction
        """
        results = []
        
        with log_performance("nlp_processing", 
                           doc_count=len(documents),
                           log_memory=True):
            # Process with memory management
            for processed in self.nlp_manager.process_batch(
                documents,
                extract_entities=True,
                extract_tokens=False
            ):
                results.append(processed)
        
        logger.info(f"Processed {len(results)} documents")
        return results
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities from text efficiently."""
        # For single document, use memory_zone
        with self.nlp_manager.nlp.memory_zone():
            doc = self.nlp_manager.nlp(text)
            entities = [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
            ]
        
        return entities
```

**Validation**:
```python

# Test memory-efficient processing
from src.services.document_processor import DocumentProcessor

processor = DocumentProcessor()

# Create test documents
test_docs = ["This is a test document." * 100 for _ in range(100)]

# Process with memory tracking
import psutil
process = psutil.Process()

mem_before = process.memory_info().rss / 1024 / 1024
results = processor.process_documents(test_docs)
mem_after = process.memory_info().rss / 1024 / 1024

print(f"Memory used: {mem_after - mem_before:.2f} MB")
print(f"Processed: {len(results)} documents")
```

**Success Criteria**:

- ✅ Integration complete

- ✅ Memory reduction verified

- ✅ Performance logging active

---

### T2.3: LangGraph StateGraph Foundation 🟡 HIGH

**Research Foundation**:

- [Orchestration & Agents Research](../../../library_research/10-orchestration_agents-research.md)

- [StateGraph Patterns](../../../library_research/10-orchestration_agents-research.json)

**Impact**: 93% agent orchestration code reduction

#### Sub-task T2.3.1: Install LangGraph Dependencies

```bash

# Install LangGraph and supervisor
uv add "langgraph>=0.5.4" "langgraph-supervisor-py>=1.0.0"

# Verify installation
python -c "from langgraph.graph import StateGraph; print('LangGraph installed')"
python -c "from langgraph_supervisor import create_supervisor; print('Supervisor installed')"
```

#### Sub-task T2.3.2: Create Agent State Schema

**Create**: `src/agents/state_schema.py`

```python
"""LangGraph state schema for multi-agent orchestration.

Research: library_research/10-orchestration_agents-research.md
93% code reduction with StateGraph patterns
"""
from typing import TypedDict, Annotated, Sequence, Optional, Any, Dict, List
from langgraph.graph import add_messages
from datetime import datetime
from enum import Enum

class AgentRole(str, Enum):
    """Agent roles in the system."""
    SUPERVISOR = "supervisor"
    SEARCHER = "searcher"
    ANALYZER = "analyzer"
    SUMMARIZER = "summarizer"
    VALIDATOR = "validator"

class Message(TypedDict):
    """Message format for agent communication."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]

class AgentState(TypedDict):
    """Global state for multi-agent orchestration.
    
    This replaces 200+ lines of custom state management
    """
    # Message history with automatic merging
    messages: Annotated[Sequence[Message], add_messages]
    
    # Current query and context
    query: str
    context: Dict[str, Any]
    
    # Agent coordination
    next_agent: Optional[str]
    current_agent: Optional[str]
    agent_history: List[str]
    
    # Results and intermediate data
    search_results: Optional[List[Dict[str, Any]]]
    analysis_results: Optional[Dict[str, Any]]
    summary: Optional[str]
    
    # Error handling
    error: Optional[str]
    error_count: int
    max_retries: int
    
    # Performance tracking
    start_time: datetime
    end_time: Optional[datetime]
    total_tokens: int
    total_cost: float

class WorkflowConfig(TypedDict):
    """Configuration for workflow execution."""
    thread_id: str
    max_iterations: int
    timeout_seconds: int
    stream_mode: str  # values, updates, debug
    recursion_limit: int
    
def create_initial_state(query: str) -> AgentState:
    """Create initial state for new workflow."""
    return AgentState(
        messages=[],
        query=query,
        context={},
        next_agent=AgentRole.SUPERVISOR,
        current_agent=None,
        agent_history=[],
        search_results=None,
        analysis_results=None,
        summary=None,
        error=None,
        error_count=0,
        max_retries=3,
        start_time=datetime.now(),
        end_time=None,
        total_tokens=0,
        total_cost=0.0
    )
```

#### Sub-task T2.3.3: Implement Basic Supervisor

**Create**: `src/agents/supervisor.py`

```python
"""LangGraph supervisor for multi-agent coordination.

Research: library_research/10-orchestration_agents-research.md
Replaces custom orchestration with library patterns
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Optional
from loguru import logger
from src.agents.state_schema import AgentState, AgentRole, create_initial_state
from src.utils.performance_logger import log_performance

class DocMindSupervisor:
    """Supervisor for multi-agent document analysis."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.1
    ):
        """Initialize supervisor with LangGraph.
        
        This replaces 300+ lines of custom orchestration code
        """
        self.model = ChatOpenAI(model=model, temperature=temperature)
        self.workflow = self._build_workflow()
        self.checkpointer = MemorySaver()  # In-memory for development
        
        # Compile the graph
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("DocMind supervisor initialized", model=model)
    
    def _build_workflow(self) -> StateGraph:
        """Build the agent workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("searcher", self._searcher_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("summarizer", self._summarizer_node)
        
        # Add edges (transitions)
        workflow.set_entry_point("supervisor")
        
        # Conditional routing from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_supervisor,
            {
                "searcher": "searcher",
                "analyzer": "analyzer",
                "summarizer": "summarizer",
                "end": END
            }
        )
        
        # Agents return to supervisor
        workflow.add_edge("searcher", "supervisor")
        workflow.add_edge("analyzer", "supervisor")
        workflow.add_edge("summarizer", "supervisor")
        
        return workflow
    
    def _supervisor_node(self, state: AgentState) -> Dict[str, Any]:
        """Supervisor logic for routing queries."""
        with log_performance("supervisor_decision", query=state["query"]):
            # Analyze query and current state
            prompt = f"""
            You are a document analysis supervisor.
            Query: {state['query']}
            Current context: {state.get('context', {})}
            Search results available: {state['search_results'] is not None}
            Analysis complete: {state['analysis_results'] is not None}
            
            Decide next action:
            - If need to find documents: route to 'searcher'
            - If have results to analyze: route to 'analyzer'
            - If need summary: route to 'summarizer'
            - If complete: route to 'end'
            
            Respond with just the agent name.
            """
            
            response = self.model.invoke(prompt)
            next_agent = response.content.strip().lower()
            
            logger.info(f"Supervisor routing to: {next_agent}")
            
            return {
                "next_agent": next_agent,
                "current_agent": "supervisor",
                "agent_history": state["agent_history"] + ["supervisor"]
            }
    
    def _route_supervisor(self, state: AgentState) -> str:
        """Route based on supervisor decision."""
        return state.get("next_agent", "end")
    
    def _searcher_node(self, state: AgentState) -> Dict[str, Any]:
        """Search agent implementation."""
        logger.info("Searcher agent activated")
        
        # Placeholder for actual search
        # In real implementation, this would use vector store
        search_results = [
            {"doc_id": "1", "content": "Sample result 1", "score": 0.95},
            {"doc_id": "2", "content": "Sample result 2", "score": 0.87}
        ]
        
        return {
            "search_results": search_results,
            "current_agent": "searcher",
            "agent_history": state["agent_history"] + ["searcher"]
        }
    
    def _analyzer_node(self, state: AgentState) -> Dict[str, Any]:
        """Analysis agent implementation."""
        logger.info("Analyzer agent activated")
        
        # Placeholder for actual analysis
        analysis = {
            "key_points": ["Point 1", "Point 2"],
            "sentiment": "neutral",
            "entities": ["Entity 1", "Entity 2"]
        }
        
        return {
            "analysis_results": analysis,
            "current_agent": "analyzer",
            "agent_history": state["agent_history"] + ["analyzer"]
        }
    
    def _summarizer_node(self, state: AgentState) -> Dict[str, Any]:
        """Summarizer agent implementation."""
        logger.info("Summarizer agent activated")
        
        # Placeholder for actual summarization
        summary = "This is a summary of the analysis results."
        
        return {
            "summary": summary,
            "current_agent": "summarizer",
            "agent_history": state["agent_history"] + ["summarizer"],
            "next_agent": "end"  # Usually final step
        }
    
    async def process_query(
        self,
        query: str,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """Process a query through the agent workflow.
        
        Args:
            query: User query
            thread_id: Thread ID for conversation continuity
            
        Returns:
            Final state with results
        """
        # Create initial state
        initial_state = create_initial_state(query)
        
        # Configuration
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 10
        }
        
        # Run the workflow
        with log_performance("workflow_execution", query=query):
            final_state = await self.app.ainvoke(
                initial_state,
                config=config
            )
        
        # Log execution summary
        logger.info("Workflow completed",
            query=query,
            agents_used=final_state["agent_history"],
            has_results=final_state["summary"] is not None
        )
        
        return final_state
    
    async def stream_query(
        self,
        query: str,
        thread_id: str = "default"
    ):
        """Stream query processing for real-time updates.
        
        Yields state updates as agents process
        """
        initial_state = create_initial_state(query)
        
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 10
        }
        
        # Stream updates
        async for chunk in self.app.astream(
            initial_state,
            config=config,
            stream_mode="updates"
        ):
            yield chunk
```

**Validation**:
```python
import asyncio
from src.agents.supervisor import DocMindSupervisor

async def test_supervisor():
    supervisor = DocMindSupervisor()
    
    # Test query processing
    result = await supervisor.process_query("Find documents about machine learning")
    
    print(f"Query processed")
    print(f"Agents used: {result['agent_history']}")
    print(f"Has summary: {result['summary'] is not None}")
    
    # Test streaming
    print("\nStreaming test:")
    async for update in supervisor.stream_query("Analyze sentiment"):
        print(f"Update: {update}")

# Run test
asyncio.run(test_supervisor())
```

**Success Criteria**:

- ✅ StateGraph workflow created

- ✅ 93% reduction in orchestration code

- ✅ Streaming support working

---

### T2.4: FastEmbed Consolidation & Multi-GPU 🟡 MEDIUM

**Research Foundation**:

- [Embedding & Vector Store Research](../../../library_research/10-embedding_vectorstore-research.md)

**Impact**: 1.84x throughput improvement, eliminate redundant providers

#### Sub-task T2.4.1: Remove Redundant Embedding Providers

```bash

# Find HuggingFace embedding usage
rg "HuggingFaceEmbedding" src/

# Find other embedding providers
rg "OpenAIEmbedding|JinaEmbedding" src/
```

Replace all with FastEmbed in identified files.

#### Sub-task T2.4.2: Configure Multi-GPU Support

**Create**: `src/utils/fastembed_gpu.py`

```python
"""FastEmbed GPU optimization and multi-GPU support.

Research: library_research/10-embedding_vectorstore-research.md
1.84x throughput with multi-GPU, eliminates API costs
"""
import os
from typing import List, Optional, Union
from fastembed import TextEmbedding
import torch
from loguru import logger
import numpy as np

class FastEmbedGPU:
    """GPU-optimized FastEmbed wrapper."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        max_length: int = 512,
        use_gpu: bool = True,
        device_ids: Optional[List[int]] = None
    ):
        """Initialize FastEmbed with GPU support.
        
        Args:
            model_name: Model to use
            max_length: Maximum sequence length
            use_gpu: Enable GPU acceleration
            device_ids: GPU IDs for multi-GPU (e.g., [0, 1])
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Configure for GPU if available
        if use_gpu and torch.cuda.is_available():
            # Set environment for optimal performance
            os.environ["OMP_NUM_THREADS"] = "1"
            
            # Determine devices
            if device_ids is None:
                device_ids = [0]  # Default to first GPU
            
            # Initialize with CUDA provider
            self.model = TextEmbedding(
                model_name=model_name,
                max_length=max_length,
                providers=["CUDAExecutionProvider"],
                provider_options=[{
                    "device_id": device_ids,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": 4 * 1024 * 1024 * 1024  # 4GB limit per GPU
                }]
            )
            
            logger.info(f"FastEmbed initialized with GPU",
                model=model_name,
                devices=device_ids
            )
        else:
            # CPU fallback
            self.model = TextEmbedding(
                model_name=model_name,
                max_length=max_length
            )
            logger.info(f"FastEmbed initialized with CPU", model=model_name)
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 256
    ) -> np.ndarray:
        """Embed documents with optimized batching.
        
        Args:
            texts: Documents to embed
            batch_size: Batch size for processing
            
        Returns:
            Embeddings array
        """
        all_embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            
            # Generate embeddings
            embeddings = list(self.model.embed(batch))
            all_embeddings.extend(embeddings)
            
            # Progress logging
            processed = min(i + batch_size, total)
            if processed % 1000 == 0 or processed == total:
                logger.debug(f"Embedded {processed}/{total} documents")
        
        return np.array(all_embeddings)
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query.
        
        Args:
            text: Query text
            
        Returns:
            Query embedding
        """
        embedding = list(self.model.embed([text]))[0]
        return np.array(embedding)

def create_multi_gpu_embedder() -> FastEmbedGPU:
    """Create embedder with multi-GPU if available.
    
    Automatically detects and uses all available GPUs
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        
        if gpu_count > 1:
            # Use all available GPUs
            device_ids = list(range(gpu_count))
            logger.info(f"Multi-GPU embedding enabled", gpu_count=gpu_count)
        else:
            # Single GPU
            device_ids = [0]
            logger.info("Single GPU embedding enabled")
        
        return FastEmbedGPU(
            use_gpu=True,
            device_ids=device_ids
        )
    else:
        logger.warning("No GPU available, using CPU embedding")
        return FastEmbedGPU(use_gpu=False)
```

**Integration**: Update `src/config/llama_settings.py`:

```python
from src.utils.fastembed_gpu import create_multi_gpu_embedder

def configure_llama_index():
    """Configure with GPU-optimized embeddings."""
    # ... existing config ...
    
    # Use GPU-optimized FastEmbed
    embedder = create_multi_gpu_embedder()
    
    # Wrap for LlamaIndex compatibility
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    Settings.embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        max_length=512
    )
```

**Success Criteria**:

- ✅ Single embedding provider

- ✅ Multi-GPU support configured

- ✅ 1.84x throughput improvement

---

### T2.5: Native Caching with Redis 🟡 MEDIUM

**Research Foundation**:

- [LlamaIndex Ecosystem Research](../../../library_research/10-llamaindex_ecosystem-research.md#native-caching)

**Impact**: 300-500% performance improvement for repeated queries

#### Sub-task T2.5.1: Setup Redis Infrastructure

**Install Redis** (if not already):
```bash

# Using Docker
docker run -d --name redis -p 6379:6379 redis:alpine

# Or install locally

# Ubuntu/Debian: sudo apt-get install redis-server

# macOS: brew install redis
```

**Create**: `src/utils/redis_setup.py`

```python
"""Redis setup and connection management.

Research: library_research/10-llamaindex_ecosystem-research.md
Native caching provides 300-500% performance improvement
"""
import redis
from typing import Optional
from loguru import logger
import os

class RedisManager:
    """Singleton Redis connection manager."""
    
    _instance: Optional[redis.Redis] = None
    _config = {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", 6379)),
        "db": int(os.getenv("REDIS_DB", 0)),
        "decode_responses": True,
        "socket_keepalive": True,
        "socket_keepalive_options": {
            1: 1,  # TCP_KEEPIDLE
            2: 1,  # TCP_KEEPINTVL
            3: 3,  # TCP_KEEPCNT
        },
        "connection_pool_kwargs": {
            "max_connections": 50,
            "socket_connect_timeout": 5,
            "socket_timeout": 5,
        }
    }
    
    @classmethod
    def get_client(cls) -> redis.Redis:
        """Get Redis client instance."""
        if cls._instance is None:
            try:
                cls._instance = redis.Redis(**cls._config)
                
                # Test connection
                cls._instance.ping()
                logger.info("Redis connection established",
                    host=cls._config["host"],
                    port=cls._config["port"]
                )
            except redis.ConnectionError as e:
                logger.error(f"Redis connection failed: {e}")
                # Fallback to in-memory cache
                logger.warning("Falling back to in-memory cache")
                from fakeredis import FakeRedis
                cls._instance = FakeRedis(decode_responses=True)
        
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset connection (for testing)."""
        if cls._instance:
            cls._instance.close()
            cls._instance = None
```

#### Sub-task T2.5.2: Implement IngestionCache

**Create**: `src/services/ingestion_cache.py`

```python
"""Document ingestion caching with Redis.

Prevents re-processing of unchanged documents
"""
from llama_index.core.ingestion import IngestionCache
from llama_index.storage.kvstore.redis import RedisKVStore
from src.utils.redis_setup import RedisManager
from loguru import logger
import hashlib
from typing import List, Optional

class DocMindIngestionCache:
    """Redis-backed ingestion cache."""
    
    def __init__(self, collection: str = "docmind_docs"):
        """Initialize ingestion cache.
        
        Args:
            collection: Cache collection name
        """
        self.collection = collection
        self.redis_client = RedisManager.get_client()
        
        # Create LlamaIndex cache
        kvstore = RedisKVStore(
            redis_client=self.redis_client,
            collection=collection
        )
        
        self.cache = IngestionCache(
            cache=kvstore,
            collection=collection
        )
        
        logger.info(f"Ingestion cache initialized", collection=collection)
    
    def get_doc_hash(self, content: str) -> str:
        """Generate hash for document content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def is_cached(self, doc_id: str, content: str) -> bool:
        """Check if document is already processed.
        
        Args:
            doc_id: Document identifier
            content: Document content
            
        Returns:
            True if document unchanged and cached
        """
        cache_key = f"{self.collection}:{doc_id}"
        cached_hash = self.redis_client.get(cache_key)
        
        if cached_hash:
            current_hash = self.get_doc_hash(content)
            return cached_hash == current_hash
        
        return False
    
    def mark_processed(self, doc_id: str, content: str):
        """Mark document as processed.
        
        Args:
            doc_id: Document identifier
            content: Document content
        """
        cache_key = f"{self.collection}:{doc_id}"
        content_hash = self.get_doc_hash(content)
        
        # Store hash with TTL (7 days)
        self.redis_client.setex(
            cache_key,
            7 * 24 * 3600,  # 7 days
            content_hash
        )
    
    def clear_cache(self):
        """Clear all cached entries."""
        pattern = f"{self.collection}:*"
        keys = self.redis_client.keys(pattern)
        
        if keys:
            self.redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries")
```

#### Sub-task T2.5.3: Add Semantic Caching

**Create**: `src/services/semantic_cache.py`

```python
"""Semantic caching for query results.

Caches similar queries to improve response time
"""
from llama_index.core.query_engine import BaseQueryEngine
from typing import Optional, Dict, Any
import numpy as np
from src.utils.redis_setup import RedisManager
from src.utils.fastembed_gpu import create_multi_gpu_embedder
from loguru import logger
import json
import time

class SemanticCache:
    """Semantic similarity-based query cache."""
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600
    ):
        """Initialize semantic cache.
        
        Args:
            similarity_threshold: Minimum similarity for cache hit
            ttl_seconds: Cache entry time-to-live
        """
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.redis_client = RedisManager.get_client()
        self.embedder = create_multi_gpu_embedder()
        
        logger.info("Semantic cache initialized",
            threshold=similarity_threshold,
            ttl=ttl_seconds
        )
    
    def _get_cache_key(self, query_embedding: np.ndarray) -> str:
        """Generate cache key from embedding."""
        # Use first 8 dimensions as key (for simplicity)
        key_values = query_embedding[:8]
        key = "query:" + ":".join([f"{v:.4f}" for v in key_values])
        return key
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result for query.
        
        Args:
            query: Query text
            
        Returns:
            Cached result if found and similar enough
        """
        start_time = time.perf_counter()
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search for similar cached queries
        pattern = "query:*"
        cursor = 0
        best_match = None
        best_similarity = 0
        
        while True:
            cursor, keys = self.redis_client.scan(
                cursor, 
                match=pattern,
                count=100
            )
            
            for key in keys:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    try:
                        data = json.loads(cached_data)
                        cached_embedding = np.array(data["embedding"])
                        
                        # Calculate cosine similarity
                        similarity = np.dot(query_embedding, cached_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
                        )
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = data
                    
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            if cursor == 0:
                break
        
        # Check if best match meets threshold
        if best_match and best_similarity >= self.similarity_threshold:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"Cache hit",
                similarity=best_similarity,
                elapsed_ms=elapsed
            )
            return best_match["result"]
        
        return None
    
    def set(self, query: str, result: Dict[str, Any]):
        """Cache query result.
        
        Args:
            query: Query text
            result: Result to cache
        """
        # Generate embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Prepare cache data
        cache_data = {
            "query": query,
            "embedding": query_embedding.tolist(),
            "result": result,
            "timestamp": time.time()
        }
        
        # Store with TTL
        cache_key = self._get_cache_key(query_embedding)
        self.redis_client.setex(
            cache_key,
            self.ttl_seconds,
            json.dumps(cache_data)
        )
        
        logger.debug(f"Cached query result", query_length=len(query))

def add_semantic_cache(
    query_engine: BaseQueryEngine,
    cache: Optional[SemanticCache] = None
) -> BaseQueryEngine:
    """Wrap query engine with semantic caching.
    
    Args:
        query_engine: Original query engine
        cache: Semantic cache instance
        
    Returns:
        Cached query engine
    """
    if cache is None:
        cache = SemanticCache()
    
    # Store original query method
    original_query = query_engine.query
    
    def cached_query(query_str: str, **kwargs):
        """Query with caching."""
        # Try cache first
        cached_result = cache.get(query_str)
        if cached_result:
            return cached_result
        
        # Execute query
        result = original_query(query_str, **kwargs)
        
        # Cache result
        cache.set(query_str, result)
        
        return result
    
    # Replace query method
    query_engine.query = cached_query
    
    logger.info("Semantic caching enabled for query engine")
    return query_engine
```

**Validation**:
```python

# Test caching
from src.services.semantic_cache import SemanticCache

cache = SemanticCache(similarity_threshold=0.9)

# First query (cache miss)
result1 = {"answer": "Test answer", "score": 0.95}
cache.set("What is machine learning?", result1)

# Similar query (cache hit)
cached = cache.get("What is ML?")  # Similar query
print(f"Cached result: {cached}")

# Different query (cache miss)
cached2 = cache.get("What is deep learning?")
print(f"Different query cached: {cached2}")
```

**Success Criteria**:

- ✅ Redis connection established

- ✅ Ingestion cache preventing duplicates

- ✅ Semantic cache improving query performance

- ✅ 300-500% performance on repeated queries

---

### T2.6: MoviePy Evaluation & Removal 🟢 LOW

**Research Foundation**:

- [Document Ingestion Research](../../../library_research/10-document_ingestion-research.md)

**Impact**: ~129MB saved, ~20 fewer packages

#### Sub-task T2.6.1: Audit MoviePy Usage

```bash

# Search for MoviePy imports
rg "import moviepy" src/ tests/
rg "from moviepy" src/ tests/

# Check if used in tests only
rg "moviepy" tests/ --files-with-matches
```

**Expected**: Only used in test mocks, not production

#### Sub-task T2.6.2: Update Test Mocks

If MoviePy is only in tests, replace with generic mocks:

```python

# OLD: MoviePy-specific mock
from unittest.mock import MagicMock

# Instead of importing moviepy
def test_video_processing():
    # Create generic mock
    video_processor = MagicMock()
    video_processor.process.return_value = {"status": "processed"}
    
    # Test without MoviePy dependency
    result = video_processor.process("test.mp4")
    assert result["status"] == "processed"
```

#### Sub-task T2.6.3: Remove MoviePy Dependency

```bash

# Remove MoviePy
uv remove moviepy

# Verify tests still pass
uv run pytest tests/

# Check package reduction
uv pip list | wc -l
```

**Success Criteria**:

- ✅ MoviePy removed

- ✅ Tests still passing

- ✅ ~129MB disk space saved

- ✅ ~20 fewer packages

---

## Phase 2 Validation Checklist

### Core Systems

- [ ] Structured JSON logging configured

- [ ] Performance logging implemented

- [ ] spaCy memory zones active (40-60% reduction)

- [ ] LangGraph StateGraph created

- [ ] Supervisor pattern implemented

### Embeddings & Caching

- [ ] FastEmbed consolidated (single provider)

- [ ] Multi-GPU support configured

- [ ] Redis caching active

- [ ] Semantic cache implemented

- [ ] 300-500% performance on cached queries

### Memory & Performance

- [ ] Memory usage reduced by 40-60%

- [ ] GPU utilization optimized

- [ ] Batch processing improved

- [ ] MoviePy removed (if unused)

## Performance Benchmarks

Create `benchmark_phase2.py`:

```python
import asyncio
import time
import psutil
import torch
from src.utils.logging_config import configure_logging, get_logger
from src.utils.spacy_memory import SpacyMemoryManager
from src.agents.supervisor import DocMindSupervisor
from src.services.semantic_cache import SemanticCache

configure_logging(json_output=False)
logger = get_logger(__name__)

async def benchmark_phase2():
    print("=== Phase 2 Validation ===\n")
    
    # Memory baseline
    process = psutil.Process()
    mem_start = process.memory_info().rss / 1024 / 1024
    
    # Test structured logging
    logger.info("Phase 2 benchmark started", phase=2)
    
    # Test spaCy memory management
    print("Testing spaCy memory zones...")
    nlp_manager = SpacyMemoryManager()
    test_texts = ["Test document " * 100 for _ in range(100)]
    
    results = list(nlp_manager.process_batch(test_texts))
    mem_after_spacy = process.memory_info().rss / 1024 / 1024
    print(f"spaCy processing: {len(results)} docs, Memory delta: {mem_after_spacy - mem_start:.2f} MB")
    
    # Test LangGraph supervisor
    print("\nTesting LangGraph supervisor...")
    supervisor = DocMindSupervisor()
    result = await supervisor.process_query("Test query")
    print(f"Supervisor agents used: {result['agent_history']}")
    
    # Test semantic cache
    print("\nTesting semantic cache...")
    cache = SemanticCache()
    
    # First query (miss)
    start = time.perf_counter()
    cache.get("What is artificial intelligence?")
    miss_time = (time.perf_counter() - start) * 1000
    
    # Cache it
    cache.set("What is artificial intelligence?", {"answer": "AI is..."})
    
    # Second query (hit)
    start = time.perf_counter()
    result = cache.get("What is AI?")  # Similar query
    hit_time = (time.perf_counter() - start) * 1000
    
    print(f"Cache miss: {miss_time:.2f}ms, Cache hit: {hit_time:.2f}ms")
    print(f"Performance improvement: {miss_time/hit_time:.1f}x")
    
    # GPU check
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    
    print("\n✅ Phase 2 validation complete")

# Run benchmark
asyncio.run(benchmark_phase2())
```

## Next Steps

After completing Phase 2:
1. Run validation benchmarks
2. Document performance improvements
3. Commit with detailed message
4. Proceed to [Phase 3: Advanced Features](./03-phase-advanced.md)

## Rollback Procedures

```bash

# Logging rollback
git checkout -- src/utils/logging_config.py

# spaCy rollback
git checkout -- src/utils/spacy_memory.py

# LangGraph rollback
uv remove langgraph langgraph-supervisor-py
git checkout -- src/agents/

# Redis rollback
docker stop redis
git checkout -- src/services/*cache*.py
```

Time to rollback: <10 minutes for any component
