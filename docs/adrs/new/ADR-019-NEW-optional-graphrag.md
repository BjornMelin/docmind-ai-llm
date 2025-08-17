# ADR-019-NEW: Optional GraphRAG Module

## Title

Microsoft GraphRAG as Optional Enhancement for Multi-Hop Reasoning

## Version/Date

1.0 / 2025-08-17

## Status

Accepted (Optional Module)

## Description

Implements Microsoft's GraphRAG as an optional module for enhanced multi-hop reasoning, relationship extraction, and thematic analysis. This module is disabled by default and can be enabled for specific use cases requiring graph-based retrieval and reasoning. GraphRAG provides superior performance for complex queries involving relationships, themes, and multi-document synthesis.

## Context

Traditional RAG systems struggle with:

- Multi-hop reasoning across documents
- Understanding entity relationships
- Identifying themes and patterns across corpus
- "What are the main themes?" type questions
- Complex relationship queries

Microsoft's GraphRAG addresses these limitations through:

- Hierarchical graph construction
- Community detection and summarization
- Entity and relationship extraction
- Graph-based retrieval and reasoning

## Related Requirements

### Functional Requirements

- **FR-1:** Extract entities and relationships from documents
- **FR-2:** Build knowledge graph with hierarchical communities
- **FR-3:** Support graph-based retrieval for complex queries
- **FR-4:** Enable theme and pattern identification
- **FR-5:** Maintain compatibility with vector-based RAG

### Non-Functional Requirements

- **NFR-1:** **(Performance)** Graph construction in background
- **NFR-2:** **(Scalability)** Support incremental graph updates
- **NFR-3:** **(Local-First)** All graph operations run locally
- **NFR-4:** **(Optional)** Can be completely disabled with no impact

## Alternatives

### 1. Vector-Only RAG (Current Default)

- **Description**: Traditional dense/sparse vector retrieval
- **Issues**: Poor at multi-hop reasoning and relationships
- **Score**: 6/10 (simplicity: 10, capability: 4, relationships: 2)

### 2. Microsoft GraphRAG (Selected for Optional)

- **Description**: Full GraphRAG implementation as optional module
- **Benefits**: Excellent for relationships, themes, multi-hop
- **Score**: 8/10 (capability: 10, complexity: 6, optional: 10)

### 3. LlamaIndex Knowledge Graph

- **Description**: Simpler KG implementation in LlamaIndex
- **Issues**: Less sophisticated than GraphRAG
- **Score**: 7/10 (capability: 7, simplicity: 8, integration: 8)

### 4. Neo4j with Custom Implementation

- **Description**: Full graph database with custom logic
- **Issues**: Heavy infrastructure, complex setup
- **Score**: 5/10 (capability: 9, complexity: 3, maintenance: 3)

## Decision

We will implement **Microsoft GraphRAG as an optional module** with:

1. **Feature Flag**: Disabled by default, enabled via config
2. **Lightweight Integration**: Minimal impact when disabled
3. **Hybrid Approach**: Complements vector RAG, doesn't replace
4. **Selective Usage**: Only for queries requiring graph reasoning
5. **Background Processing**: Graph construction happens async

## Related Decisions

- **ADR-003-NEW** (Adaptive Retrieval): Routes to GraphRAG for complex queries
- **ADR-009-NEW** (Document Processing): Provides input for graph construction
- **ADR-011-NEW** (Agent Orchestration): Planning agent can invoke GraphRAG
- **ADR-007-NEW** (Persistence): Stores graph data alongside vectors

## Design

### GraphRAG Integration Architecture

```python
from graphrag import GraphRAGPipeline, GraphRAGConfig
from graphrag.index import create_knowledge_graph
from graphrag.query import GraphRAGQueryEngine
from typing import Optional, List, Dict, Any
import asyncio
from pathlib import Path

class OptionalGraphRAG:
    """Optional GraphRAG module for enhanced reasoning."""
    
    def __init__(
        self, 
        enabled: bool = False,
        config_path: str = "config/graphrag.yaml"
    ):
        self.enabled = enabled
        self.pipeline = None
        self.query_engine = None
        self.graph_store = None
        
        if enabled:
            self._initialize_graphrag(config_path)
    
    def _initialize_graphrag(self, config_path: str):
        """Initialize GraphRAG components."""
        config = GraphRAGConfig.from_yaml(config_path)
        
        # Configure for local operation
        config.llm.type = "ollama"
        config.llm.model = "qwen3:14b"
        config.embeddings.type = "local"
        config.embeddings.model = "BAAI/bge-m3"
        
        self.pipeline = GraphRAGPipeline(config)
        self.query_engine = GraphRAGQueryEngine(config)
    
    def is_graph_query(self, query: str) -> bool:
        """Determine if query benefits from graph reasoning."""
        if not self.enabled:
            return False
        
        # Keywords indicating graph reasoning needed
        graph_indicators = [
            "relationship", "related", "connection",
            "theme", "pattern", "trend",
            "compare", "contrast", "between",
            "how does", "why does", "main points",
            "summarize across", "common"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in graph_indicators)
    
    async def build_graph(
        self, 
        documents: List[Document],
        force_rebuild: bool = False
    ):
        """Build knowledge graph from documents."""
        if not self.enabled:
            return
        
        # Check if graph exists and skip if not forcing
        if self._graph_exists() and not force_rebuild:
            return
        
        # Run graph construction in background
        await self.pipeline.create_index_async(
            documents=documents,
            chunk_size=1200,
            chunk_overlap=100,
            enable_community_detection=True,
            max_community_level=3
        )
    
    def query(
        self, 
        query: str,
        query_type: str = "global"
    ) -> Optional[Dict[str, Any]]:
        """Query the knowledge graph."""
        if not self.enabled:
            return None
        
        # GraphRAG query types
        if query_type == "global":
            # For themes, patterns, summaries
            result = self.query_engine.query_global(query)
        else:
            # For specific entity/relationship queries
            result = self.query_engine.query_local(query)
        
        return {
            "answer": result.answer,
            "context": result.context,
            "communities": result.communities,
            "confidence": result.confidence
        }
```

### Hybrid RAG Router

```python
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever

class HybridRAGRouter:
    """Routes queries between vector and graph RAG."""
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        graph_rag: OptionalGraphRAG,
        llm_router: bool = True
    ):
        self.vector_retriever = vector_retriever
        self.graph_rag = graph_rag
        self.llm_router = llm_router
    
    def route_query(self, query: str) -> str:
        """Determine routing strategy for query."""
        
        # If GraphRAG disabled, always use vector
        if not self.graph_rag.enabled:
            return "vector"
        
        # Quick heuristic check
        if self.graph_rag.is_graph_query(query):
            return "graph"
        
        # Optional LLM-based routing for ambiguous cases
        if self.llm_router:
            routing_prompt = f"""
            Classify if this query needs graph-based reasoning:
            Query: {query}
            
            Graph reasoning needed for:
            - Relationship questions
            - Theme/pattern identification  
            - Multi-hop reasoning
            - Cross-document synthesis
            
            Reply with: "graph" or "vector"
            """
            
            # Use lightweight model for routing
            response = self.llm.complete(routing_prompt)
            return response.strip().lower()
        
        return "vector"
    
    def retrieve(
        self, 
        query: str,
        hybrid_weight: float = 0.7
    ) -> List[Document]:
        """Retrieve using appropriate strategy."""
        
        route = self.route_query(query)
        
        if route == "graph" and self.graph_rag.enabled:
            # Use GraphRAG
            graph_result = self.graph_rag.query(query)
            
            if graph_result and graph_result["confidence"] > 0.7:
                # High confidence graph answer
                return self._format_graph_results(graph_result)
            else:
                # Low confidence, blend with vector
                vector_results = self.vector_retriever.retrieve(
                    QueryBundle(query_str=query)
                )
                return self._blend_results(
                    graph_result, 
                    vector_results,
                    hybrid_weight
                )
        else:
            # Use vector RAG
            return self.vector_retriever.retrieve(
                QueryBundle(query_str=query)
            )
```

### GraphRAG Configuration

```yaml
# config/graphrag.yaml
graphrag:
  enabled: false  # Disabled by default
  
  # LLM Configuration
  llm:
    type: "ollama"
    model: "qwen3:14b"
    temperature: 0.3
    max_tokens: 2000
  
  # Embedding Configuration  
  embeddings:
    type: "local"
    model: "BAAI/bge-m3"
    batch_size: 16
  
  # Graph Construction
  index:
    chunk_size: 1200
    chunk_overlap: 100
    entity_extraction:
      enabled: true
      types: ["person", "organization", "location", "concept", "technology"]
    relationship_extraction:
      enabled: true
      max_relationships_per_chunk: 20
    community_detection:
      enabled: true
      algorithm: "leiden"
      max_level: 3
      min_community_size: 5
  
  # Storage
  storage:
    type: "sqlite"
    path: "data/graphrag.db"
    cache_enabled: true
    cache_ttl: 3600
  
  # Query Configuration
  query:
    global:
      map_reduce_prompts: true
      community_summaries: true
      max_tokens: 3000
    local:
      include_community_context: true
      max_hops: 2
      context_window: 5000
```

### Integration with Main Pipeline

```python
from typing import Optional
import os

class DocMindRAGPipeline:
    """Main RAG pipeline with optional GraphRAG."""
    
    def __init__(self, config: Dict[str, Any]):
        # Vector RAG (always enabled)
        self.vector_index = self._build_vector_index(config)
        self.vector_retriever = self.vector_index.as_retriever()
        
        # GraphRAG (optional)
        graph_enabled = config.get("graphrag", {}).get("enabled", False)
        self.graph_rag = OptionalGraphRAG(
            enabled=graph_enabled,
            config_path=config.get("graphrag_config", "config/graphrag.yaml")
        )
        
        # Hybrid router
        self.router = HybridRAGRouter(
            vector_retriever=self.vector_retriever,
            graph_rag=self.graph_rag,
            llm_router=True
        )
        
        # Build graph if enabled
        if graph_enabled:
            asyncio.create_task(self._build_graph_index())
    
    async def _build_graph_index(self):
        """Build graph index in background."""
        if not self.graph_rag.enabled:
            return
        
        print("Building GraphRAG index in background...")
        documents = self._load_all_documents()
        await self.graph_rag.build_graph(documents)
        print("GraphRAG index ready")
    
    def query(
        self, 
        query: str,
        use_graph: Optional[bool] = None
    ) -> str:
        """Process query through appropriate pipeline."""
        
        # Allow override of routing
        if use_graph is not None:
            if use_graph and self.graph_rag.enabled:
                result = self.graph_rag.query(query)
                if result:
                    return result["answer"]
        
        # Use router for automatic selection
        documents = self.router.retrieve(query)
        
        # Generate response
        response = self.response_generator.generate(
            query=query,
            documents=documents
        )
        
        return response
```

### Monitoring and Metrics

```python
class GraphRAGMonitor:
    """Monitor GraphRAG performance and usage."""
    
    def __init__(self):
        self.metrics = {
            "graph_queries": 0,
            "vector_queries": 0,
            "hybrid_queries": 0,
            "graph_build_time": 0,
            "avg_graph_latency": 0,
            "routing_decisions": {}
        }
    
    def record_query(
        self, 
        query_type: str,
        latency: float,
        confidence: float
    ):
        """Record query metrics."""
        self.metrics[f"{query_type}_queries"] += 1
        
        # Update average latency
        if query_type == "graph":
            current_avg = self.metrics["avg_graph_latency"]
            count = self.metrics["graph_queries"]
            self.metrics["avg_graph_latency"] = (
                (current_avg * (count - 1) + latency) / count
            )
    
    def should_enable_graphrag(self) -> bool:
        """Recommend enabling GraphRAG based on usage."""
        total_queries = sum([
            self.metrics["graph_queries"],
            self.metrics["vector_queries"]
        ])
        
        if total_queries < 100:
            return False  # Not enough data
        
        # Check if many queries would benefit from graph
        graph_beneficial_ratio = (
            self.metrics.get("graph_beneficial", 0) / total_queries
        )
        
        return graph_beneficial_ratio > 0.2  # 20% threshold
```

## Consequences

### Positive Outcomes

- **Enhanced Reasoning**: Superior multi-hop and relationship queries
- **Theme Identification**: Automatic pattern and theme extraction
- **Optional Complexity**: No impact when disabled
- **Hybrid Flexibility**: Combines with vector RAG seamlessly
- **Background Processing**: Graph built without blocking

### Negative Consequences / Trade-offs

- **Build Time**: Initial graph construction can take minutes
- **Storage Overhead**: Graph data adds ~2x document storage
- **Complexity**: Additional system to understand and maintain
- **Query Latency**: Graph queries slower than vector (2-5s vs <1s)

### Migration Strategy

1. **Start Disabled**: Ship with GraphRAG disabled
2. **Selective Enable**: Enable for specific use cases
3. **Monitor Usage**: Track which queries benefit
4. **Gradual Rollout**: Enable based on usage patterns
5. **User Choice**: Let power users opt-in

## Performance Targets

- **Graph Build**: <30 seconds per 100 documents
- **Query Latency**: <3 seconds for graph queries
- **Storage Overhead**: <2x original document size
- **Memory Usage**: <500MB additional when active

## Dependencies

- **Python**: `graphrag>=0.1.0`
- **Optional**: `networkx>=3.0`, `leiden-algorithm`
- **Storage**: SQLite or PostgreSQL for graph data
- **Models**: Local LLM for entity extraction

## Monitoring Metrics

- Graph construction time and success rate
- Query routing decisions (graph vs vector)
- Graph query latency and confidence scores
- Storage growth over time
- Cache hit rates for community summaries

## Future Enhancements

- Incremental graph updates
- Graph visualization interface
- Custom entity types per domain
- Graph pruning and optimization
- Integration with knowledge bases

## Changelog

- **1.0 (2025-08-17)**: Initial optional GraphRAG module design with Microsoft GraphRAG integration
