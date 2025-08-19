# ADR-019: Optional GraphRAG Module

## Title

LlamaIndex PropertyGraphIndex for Optional Graph-Enhanced Retrieval

## Version/Date

3.0 / 2025-08-19

## Status

Accepted (Optional Module)

## Description

Implements LlamaIndex's native PropertyGraphIndex as an optional module leveraging Qwen3-4B-Instruct-2507's 262K context capability for entity and relationship extraction within large context windows. This approach enables processing entire document collections for graph construction without chunking limitations. The module requires ZERO additional infrastructure - using SimplePropertyGraphStore (in-memory) for graph storage while reusing existing Qdrant vector store for embeddings. Large context enables knowledge graph construction from documents in single processing passes.

## Context

Traditional RAG systems struggle with:

- Multi-hop reasoning across documents
- Understanding entity relationships
- Identifying themes and patterns across corpus
- "What are the main themes?" type questions
- Complex relationship queries

LlamaIndex PropertyGraphIndex addresses these limitations through:

- Native integration with existing LlamaIndex pipeline
- Automatic entity and relationship extraction via LLM
- Reuses existing Qdrant vector store for embeddings
- Multiple retrieval strategies (vector, graph traversal, hybrid)
- ZERO additional infrastructure with SimplePropertyGraphStore

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

### 2. LlamaIndex PropertyGraphIndex (Selected)

- **Description**: Native LlamaIndex graph index with in-memory storage
- **Benefits**: Zero infrastructure, reuses Qdrant, minimal code
- **Score**: 9/10 (capability: 8, simplicity: 10, integration: 10)

### 3. Microsoft GraphRAG

- **Description**: Full GraphRAG implementation with hierarchical clustering
- **Issues**: Heavy dependencies, complex setup, requires separate graph DB
- **Score**: 6/10 (capability: 10, complexity: 4, maintenance: 4)

### 4. Neo4j with Custom Implementation

- **Description**: Full graph database with custom logic
- **Issues**: Heavy infrastructure, complex setup
- **Score**: 5/10 (capability: 9, complexity: 3, maintenance: 3)

## Decision

We will implement **LlamaIndex PropertyGraphIndex as an optional module** with:

1. **Feature Flag**: Disabled by default, enabled via config
2. **Zero Infrastructure**: Uses in-memory SimplePropertyGraphStore
3. **Qdrant Reuse**: Leverages existing vector store for embeddings
4. **Native Integration**: Works with LlamaIndex pipeline
5. **Minimal Code**: <100 lines of integration code required

## Related Decisions

- **ADR-003** (Adaptive Retrieval): Routes to GraphRAG for complex queries
- **ADR-009** (Document Processing): Provides input for graph construction
- **ADR-011** (Agent Orchestration): Planning agent can invoke GraphRAG
- **ADR-007** (Persistence): Stores graph data alongside vectors

## Design

### PropertyGraphIndex Integration Architecture

```python
from llama_index.core import PropertyGraphIndex, VectorStoreIndex
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    ImplicitPathExtractor
)
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from typing import Optional, List, Dict, Any
import pickle
from pathlib import Path

class OptionalGraphRAG:
    """Optional PropertyGraphIndex for graph-based reasoning."""
    
    def __init__(
        self, 
        enabled: bool = False,
        vector_store: Optional[QdrantVectorStore] = None,
        persist_dir: Optional[str] = "data/graph_store"
    ):
        self.enabled = enabled
        self.graph_index = None
        self.graph_store = None
        self.vector_store = vector_store  # Reuse existing Qdrant
        
        if enabled:
            self._initialize_graph_index(persist_dir)
    
    def _initialize_graph_index(self, persist_dir: str):
        """Initialize PropertyGraphIndex with minimal setup."""
        # Use in-memory graph store (can persist to disk)
        self.graph_store = SimplePropertyGraphStore()
        
        # Try to load existing graph if available
        graph_path = Path(persist_dir) / "graph_store.pkl"
        if graph_path.exists():
            with open(graph_path, "rb") as f:
                self.graph_store = pickle.load(f)
    
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
    
    def build_graph(
        self, 
        documents: List[Document],
        llm,  # Local LLM instance
        embed_model,  # BGE-M3 embedding model
        force_rebuild: bool = False
    ):
        """Build property graph from documents."""
        if not self.enabled:
            return
        
        # Skip if graph exists and not forcing rebuild
        if self.graph_index and not force_rebuild:
            return
        
        # Create extractors - minimal setup
        kg_extractors = [
            SimpleLLMPathExtractor(
                llm=llm,
                max_paths_per_chunk=10,  # Limit for efficiency
                num_workers=1  # Single worker for local
            ),
            ImplicitPathExtractor()  # Extract implicit relationships
        ]
        
        # Build index reusing existing Qdrant vector store
        self.graph_index = PropertyGraphIndex.from_documents(
            documents,
            kg_extractors=kg_extractors,
            property_graph_store=self.graph_store,
            vector_store=self.vector_store,  # Reuse existing!
            embed_model=embed_model,  # BGE-M3
            show_progress=True
        )
        
        # Persist graph to disk
        self._save_graph()
    
    def query(
        self, 
        query: str,
        mode: str = "hybrid"
    ) -> Optional[Dict[str, Any]]:
        """Query the property graph."""
        if not self.enabled or not self.graph_index:
            return None
        
        # Create retriever with specified mode
        if mode == "vector":
            # Pure vector similarity from graph nodes
            retriever = self.graph_index.as_retriever(
                similarity_top_k=5,
                include_text=True
            )
        elif mode == "graph":
            # Pure graph traversal
            retriever = self.graph_index.as_retriever(
                retriever_mode="keyword",
                include_text=True
            )
        else:  # hybrid (default)
            # Combine vector and graph retrieval
            retriever = self.graph_index.as_retriever(
                similarity_top_k=3,
                path_depth=2,
                include_text=True
            )
        
        # Retrieve relevant nodes
        nodes = retriever.retrieve(query)
        
        return {
            "nodes": nodes,
            "num_results": len(nodes),
            "mode": mode
        }
    
    def _save_graph(self):
        """Persist graph store to disk."""
        if self.graph_store:
            persist_dir = Path("data/graph_store")
            persist_dir.mkdir(parents=True, exist_ok=True)
            with open(persist_dir / "graph_store.pkl", "wb") as f:
                pickle.dump(self.graph_store, f)
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

### PropertyGraphIndex Configuration

```yaml
# config/graphrag.yaml
graphrag:
  enabled: false  # Disabled by default
  
  # Graph Extraction
  extraction:
    max_paths_per_chunk: 10  # Limit for local efficiency
    num_workers: 1  # Single worker for local processing
    include_implicit: true  # Extract implicit relationships
  
  # Storage
  storage:
    type: "simple"  # In-memory SimplePropertyGraphStore
    persist_dir: "data/graph_store"
    persist_on_build: true
  
  # Retrieval Configuration
  retrieval:
    default_mode: "hybrid"  # vector, graph, or hybrid
    vector_top_k: 5
    graph_depth: 2
    include_text: true  # Include source text with results
  
  # Query Routing
  routing:
    graph_indicators:  # Keywords that trigger graph retrieval
      - "relationship"
      - "related"
      - "connection"
      - "theme"
      - "pattern"
      - "compare"
      - "between"
      - "how does"
      - "main points"
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
        
        # PropertyGraphIndex (optional)
        graph_enabled = config.get("graphrag", {}).get("enabled", False)
        self.graph_rag = OptionalGraphRAG(
            enabled=graph_enabled,
            vector_store=self.vector_store,  # Reuse existing Qdrant!
            persist_dir=config.get("graphrag", {}).get("storage", {}).get("persist_dir", "data/graph_store")
        )
        
        # Hybrid router
        self.router = HybridRAGRouter(
            vector_retriever=self.vector_retriever,
            graph_rag=self.graph_rag,
            llm_router=True
        )
        
        # Build graph if enabled
        if graph_enabled:
            self._build_graph_index()
    
    def _build_graph_index(self):
        """Build property graph index."""
        if not self.graph_rag.enabled:
            return
        
        print("Building PropertyGraphIndex...")
        documents = self._load_all_documents()
        self.graph_rag.build_graph(
            documents,
            llm=self.llm,  # Local Qwen3-14B
            embed_model=self.embed_model  # BGE-M3
        )
        print("PropertyGraphIndex ready")
    
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
- **Hybrid Flexibility**: Combines with vector RAG
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

- **Core**: Already included in `llama-index-core>=0.10.53`
- **No Additional**: Uses existing LlamaIndex, Qdrant, BGE-M3
- **Storage**: In-memory with optional pickle persistence
- **Models**: Reuses existing local LLM and embeddings

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
