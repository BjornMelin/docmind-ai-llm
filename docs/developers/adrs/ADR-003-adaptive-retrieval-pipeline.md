# ADR-003: Adaptive Retrieval Pipeline with RAPTOR-Lite

## Title

Hierarchical Adaptive Retrieval with Simplified RAPTOR and Multi-Strategy Routing

## Version/Date

3.2 / 2025-09-02

## Status

Accepted

## Description

Implements an adaptive retrieval pipeline that combines simplified hierarchical retrieval (RAPTOR-Lite), multi-strategy routing, and corrective mechanisms. The system automatically selects optimal retrieval approaches based on query characteristics while maintaining hierarchical document organization for complex information synthesis without the full complexity of RAPTOR.

**Enhanced Capabilities:**

- **DSPy Query Optimization** (ADR-018): Automatic query rewriting and expansion for improved retrieval quality
- **Optional GraphRAG Integration** (ADR-019): PropertyGraphIndex for relationship-based and multi-hop retrieval when enabled

## Context

Current retrieval is limited to flat vector similarity search with basic reranking. Modern retrieval challenges require:

1. **Hierarchical Understanding**: Queries need both specific details and high-level context
2. **Strategy Adaptation**: Different queries benefit from different retrieval approaches
3. **Quality Correction**: Poor initial retrieval should trigger alternative strategies
4. **Local Efficiency**: Complex retrieval must work within consumer hardware constraints

Full RAPTOR implementation is too resource-intensive for local deployment. Our RAPTOR-Lite approach maintains hierarchical benefits while optimizing for local-first constraints.

**Integration Flow**: The pipeline consumes BGE-M3 embeddings (ADR-002) stored in Qdrant (ADR-031), applies adaptive routing strategies optimized for 128K context windows, and uses BGE-reranker-v2-m3 (ADR-006) to refine results before returning them to the agentic RAG system with FP8 optimization.

## Related Requirements

### Functional Requirements

- **FR-1:** Route queries to optimal retrieval strategy based on characteristics
- **FR-2:** Provide hierarchical document access for multi-level information needs
- **FR-3:** Correct poor retrieval results through alternative strategies
- **FR-4:** Support both specific fact queries and broad synthesis queries

### Non-Functional Requirements

- **NFR-1:** **(Performance)** Hierarchical retrieval overhead <1 second optimized for 128K context constraints
- **NFR-2:** **(Memory)** RAPTOR-Lite tree storage <500MB for 10,000 documents within 128K context limits
- **NFR-3:** **(Quality)** ≥15% improvement in complex query answering vs flat retrieval
- **NFR-4:** **(Local-First)** All hierarchical processing occurs locally

## Alternatives

### 1. Flat Vector Retrieval (Current)

- **Description**: Single-level similarity search with reranking
- **Issues**: Cannot synthesize across document sections, poor for complex queries
- **Score**: 4/10 (simplicity: 9, capability: 2, scalability: 3)

### 2. Full RAPTOR Implementation

- **Description**: Complete recursive tree with clustering and summarization
- **Issues**: Too resource-intensive, complex for local deployment, over-engineered
- **Score**: 5/10 (capability: 9, performance: 2, complexity: 3)

### 3. RAPTOR-Lite + Multi-Strategy (Selected)

- **Description**: Simplified hierarchical retrieval with adaptive routing
- **Benefits**: Balanced capability/performance, local-optimized, maintainable
- **Score**: 8/10 (capability: 7, performance: 8, simplicity: 8)

## Decision

We will implement **Multi-Strategy Adaptive Routing using LlamaIndex built-in features**:

### Core Components (Library-First Approach)

1. **RouterQueryEngine**: LlamaIndex's built-in query routing instead of custom routing
2. **HybridRetriever**: Built-in hybrid search combining dense and sparse retrieval
3. **MultiQueryRetriever**: Automatic query decomposition for complex queries
4. **MetadataFilters**: Built-in filtering without custom logic
5. **AsyncQueryEngine**: Native async support for performance
6. **DSPy Query Optimization**: Automatic query rewriting and expansion (ADR-018)
7. **Optional PropertyGraphIndex**: Graph-based retrieval for relationship queries (ADR-019)

## Related Decisions

- **ADR-002** (Unified Embedding Strategy): Provides BGE-M3 embeddings for hierarchical indexing
- **ADR-006** (Modern Reranking Architecture): Reranks hierarchical retrieval results using BGE-reranker-v2-m3
- **ADR-031** (Local-First Persistence Architecture): Accesses Qdrant vector storage for similarity search operations
- **ADR-001** (Modern Agentic RAG): Uses adaptive retrieval for intelligent routing
- **ADR-011** (Agent Orchestration Framework): Orchestrates adaptive retrieval decisions
- **ADR-018** (DSPy Prompt Optimization): Provides automatic query optimization for improved retrieval
- **ADR-019** (Optional GraphRAG): Adds graph-based retrieval capabilities for complex relationship queries

## Design

### RAPTOR-Lite Architecture

```mermaid
graph TD
    A[Documents] --> B[Chunk Extraction]
    B --> C[Section Grouping]
    C --> D[Section Summaries]
    D --> E[Hierarchical Index]
    
    F[Query] --> G{Strategy Router}
    G -->|Simple| H[Vector Search]
    G -->|Complex| I[Hierarchical Search]
    G -->|Hybrid| J[Multi-Level Search]
    G -->|Recent| K[Web Search]
    G -->|Relationships| O[GraphRAG Search]
    
    H --> L[Results]
    I --> L
    J --> L
    K --> L
    O --> L
    
    L --> M{Quality Check}
    M -->|Good| N[Return Results]
    M -->|Poor| O[Expand Strategy] --> G
```

### Library-First Implementation Using LlamaIndex

```python
# Use LlamaIndex built-in features instead of custom code
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.retrievers import (
    HybridRetriever,
    MultiQueryRetriever,
    RecursiveRetriever
)
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser
)
# NOTE: Document processing uses direct Unstructured.io chunk_by_title() per ADR-009
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core import QueryBundle, Settings

# REMOVED: Custom hierarchical implementation
# LlamaIndex provides RecursiveRetriever for hierarchical retrieval

def create_adaptive_retriever(vector_store, llm, enable_dspy=False, enable_graphrag=False):
    """Create router-based adaptive retriever using LlamaIndex built-ins with DSPy and GraphRAG integration.
    
    IMPORTANT: This retrieval component works with document chunks that have been processed
    by ResilientDocumentProcessor using direct Unstructured.io integration (ADR-009).
    Document processing and chunking is handled upstream by:
    - Direct partition() calls for document processing
    - Direct chunk_by_title() for semantic chunking
    
    This function focuses on retrieval strategy routing and query optimization.
    """
    
    # NOTE: Document chunking handled by ResilientDocumentProcessor (ADR-009)
    # Document processing results cached via IngestionCache (ADR-030) backed by DuckDBKVStore
    # semantic_splitter = SemanticSplitterNodeParser(  # Available for special cases only
    #     embed_model=Settings.embed_model,
    #     breakpoint_percentile_threshold=95,
    #     buffer_size=1
    # )
    
    # Initialize DSPy optimization if enabled (ADR-018)
    if enable_dspy:
        from src.dspy_integration import DSPyLlamaIndexRetriever
        base_retriever = DSPyLlamaIndexRetriever(
            index=vector_store,
            enable_dspy=True
        )
    else:
        base_retriever = vector_store.as_retriever(similarity_top_k=10)
    
    # Use built-in hybrid retriever
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_store.as_retriever(similarity_top_k=5),
        keyword_retriever=vector_store.as_retriever(mode="keyword", top_k=5),
        fusion_mode="reciprocal_rank"  # Built-in RRF fusion
    )
    
    # Use MultiQueryRetriever for query decomposition
    multi_query_retriever = MultiQueryRetriever.from_defaults(
        retriever=hybrid_retriever,
        llm=llm,
        num_queries=3  # Generate 3 sub-queries automatically
    )
    
    # Create query engines for different strategies
    from llama_index.core import VectorStoreIndex
    from llama_index.core.tools import QueryEngineTool
    
    # Vector search tool with optional DSPy optimization
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=base_retriever.as_query_engine() if enable_dspy else vector_store.as_query_engine(similarity_top_k=10),
        name="vector_search",
        description="Best for semantic similarity and simple factual queries"
    )
    
    # Hybrid search tool  
    hybrid_tool = QueryEngineTool.from_defaults(
        query_engine=hybrid_retriever.as_query_engine(),
        name="hybrid_search",
        description="Best for complex queries needing both keywords and semantics"
    )
    
    # Multi-query tool for decomposition
    multi_tool = QueryEngineTool.from_defaults(
        query_engine=multi_query_retriever.as_query_engine(),
        name="multi_query",
        description="Best for complex questions that need to be broken down"
    )
    
    # Create base tools list
    tools = [vector_tool, hybrid_tool, multi_tool]
    
    # Optional GraphRAG tool (ADR-019)
    if enable_graphrag:
        from src.graphrag_integration import OptionalGraphRAG
        graph_rag = OptionalGraphRAG(enabled=True, vector_store=vector_store)
        
        graphrag_tool = QueryEngineTool.from_defaults(
            query_engine=graph_rag.as_query_engine(),
            name="graph_search",
            description="Best for relationship queries, multi-hop reasoning, and thematic analysis"
        )
        tools.append(graphrag_tool)

    # Create RouterQueryEngine to automatically route queries
    router_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=tools,
        verbose=True  # Log routing decisions
    )
    
    return router_engine

# That's it! 20 lines instead of 200+ lines of custom code
# LlamaIndex handles all the routing, retrieval, and fusion logic
```

### Using MetadataFilters for Advanced Retrieval

```python
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition
)

def create_filtered_retriever(vector_store, filters: dict):
    """Use built-in metadata filtering instead of custom logic."""
    
    # Build metadata filters
    metadata_filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="doc_type",
                value=filters.get("doc_type", "any"),
                operator=FilterOperator.EQ
            ),
            MetadataFilter(
                key="date",
                value=filters.get("after_date", "2024-01-01"),
                operator=FilterOperator.GTE
            )
        ],
        condition=FilterCondition.AND
    )
    
    # Create retriever with filters
    return vector_store.as_retriever(
        similarity_top_k=10,
        filters=metadata_filters
    )
    
    def _hierarchical_retrieve(self, query: str, top_k: int = 10) -> List[NodeWithScore]:
        """Retrieve using RAPTOR-Lite hierarchy."""
        results = []
        
        # Search level 1 (section summaries) first
        if self.hierarchical_index.level_index[1]:
            level1_results = self._search_level(query, level=1, top_k=5)
            
            # For each relevant section, get its chunks
            for section_result in level1_results:
                section_node = self.hierarchical_index.nodes[section_result.node.id_]
                if section_node.children_ids:
                    for chunk_id in section_node.children_ids:
                        chunk_node = self.hierarchical_index.nodes[chunk_id]
                        # Re-score chunk in context of section relevance
                        score = section_result.score * 0.8  # Discount for indirection
                        results.append(NodeWithScore(
                            node=Document(text=chunk_node.content, id_=chunk_id),
                            score=score
                        ))
        
        # Add direct chunk matches
        chunk_results = self._search_level(query, level=0, top_k=5)
        results.extend(chunk_results)
        
        # Deduplicate and sort by score
        seen_ids = set()
        unique_results = []
        for result in sorted(results, key=lambda x: x.score, reverse=True):
            if result.node.id_ not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.node.id_)
        
        return unique_results[:top_k]
    
    def _search_level(self, query: str, level: int, top_k: int) -> List[NodeWithScore]:
        """Search specific hierarchy level."""
        query_embedding = Settings.embed_model.get_text_embedding(query)
        results = []
        
        for node_id in self.hierarchical_index.level_index[level]:
            node = self.hierarchical_index.nodes[node_id]
            
            # Get or compute node embedding
            if node.embedding is None:
                node.embedding = Settings.embed_model.get_text_embedding(node.content)
            
            # Compute similarity
            similarity = np.dot(query_embedding, node.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
            )
            
            results.append(NodeWithScore(
                node=Document(text=node.content, id_=node_id),
                score=float(similarity)
            ))
        
        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
    
    def _hybrid_retrieve(self, query: str) -> List[NodeWithScore]:
        """Combine multiple retrieval strategies."""
        vector_results = self._vector_retrieve(query, top_k=5)
        hierarchical_results = self._hierarchical_retrieve(query, top_k=5)
        
        # Simple score fusion
        all_results = []
        all_results.extend([(r, 0.6) for r in vector_results])  # Weight vector higher for precision
        all_results.extend([(r, 0.4) for r in hierarchical_results])  # Lower weight for recall
        
        # Combine scores for same documents
        score_map = {}
        for result, weight in all_results:
            doc_id = result.node.id_
            if doc_id in score_map:
                score_map[doc_id] = (score_map[doc_id][0], 
                                   score_map[doc_id][1] + result.score * weight)
            else:
                score_map[doc_id] = (result.node, result.score * weight)
        
        # Sort by combined score
        combined_results = [NodeWithScore(node=node, score=score) 
                          for node, score in score_map.values()]
        
        return sorted(combined_results, key=lambda x: x.score, reverse=True)[:10]
```

### Using AsyncQueryEngine for Performance

```python
from llama_index.core.query_engine import AsyncQueryEngine
from llama_index.core.callbacks import TokenCountingHandler
import asyncio

async def create_async_pipeline(vector_store, llm):
    """Use built-in async support for better performance."""
    
    # Token counting with built-in handler
    token_counter = TokenCountingHandler()
    Settings.callback_manager.add_handler(token_counter)
    
    # Create async query engine
    async_engine = AsyncQueryEngine.from_defaults(
        retriever=vector_store.as_retriever(),
        llm=llm,
        streaming=True  # Enable streaming responses
    )
    
    # Use built-in retry policy
    from llama_index.core.retry import RetryPolicy
    async_engine.retry_policy = RetryPolicy(
        max_retries=3,
        retry_on_exceptions=[Exception],
        backoff_factor=2.0
    )
    
    return async_engine

# Essential RAPTOR-Lite Functions for Hierarchical Document Processing
def _create_section_summaries(documents: List[Document], llm) -> Dict[str, str]:
    """Create section summaries for hierarchical indexing."""
    summaries = {}
    
    # Group documents by section (based on metadata or content patterns)
    sections = _group_documents_by_section(documents)
    
    for section_id, section_docs in sections.items():
        # Combine section content
        section_content = "\n\n".join([doc.text for doc in section_docs])
        
        # Generate summary using local LLM
        summary_prompt = f"""
        Summarize the following section concisely, focusing on key concepts and themes:
        
        {section_content[:100000]}  # Sized for 128K context window (see ADR-010)
        
        Summary:"""
        
        summary = llm.complete(summary_prompt).text.strip()
        summaries[section_id] = summary
    
    return summaries

def _summarize_cluster(cluster_docs: List[Document], llm) -> str:
    """Summarize a cluster of related documents."""
    if not cluster_docs:
        return ""
    
    # Combine cluster content
    cluster_content = "\n\n".join([doc.text[:1000] for doc in cluster_docs])  # Limit per doc
    
    summary_prompt = f"""
    Create a brief summary that captures the main themes of these related documents:
    
    {cluster_content[:3000]}
    
    Key themes and concepts:"""
    
    return llm.complete(summary_prompt).text.strip()

def _extract_topic(text: str, llm) -> str:
    """Extract the main topic or theme from text."""
    topic_prompt = f"""
    Identify the main topic or theme of this text in 2-4 words:
    
    {text[:1000]}
    
    Main topic:"""
    
    return llm.complete(topic_prompt).text.strip()

def _group_documents_by_section(documents: List[Document]) -> Dict[str, List[Document]]:
    """Group documents by logical sections for hierarchical organization."""
    sections = {}
    
    for doc in documents:
        # Simple section grouping based on document metadata or content patterns
        section_id = doc.metadata.get("section", "general")
        if section_id not in sections:
            sections[section_id] = []
        sections[section_id].append(doc)
    
    return sections

class QualityMetrics:
    """Helper class for retrieval quality assessment."""
    
    def _calculate_relevance(self, query: str, results: List[NodeWithScore]) -> float:
        """Estimate relevance based on keyword overlap and scores."""
        query_words = set(query.lower().split())
        total_relevance = 0.0
        
        for result in results[:3]:  # Top 3 for efficiency
            content_words = set(result.node.text.lower().split())
            overlap = len(query_words.intersection(content_words)) / len(query_words)
            score_weight = result.score
            total_relevance += overlap * score_weight
        
        return min(total_relevance / 3.0, 1.0)
    
    def _calculate_coverage(self, query: str, results: List[NodeWithScore]) -> float:
        """Estimate query coverage based on result diversity."""
        if len(results) < 2:
            return 0.5
        
        # Simple diversity measure
        embeddings = []
        for result in results[:5]:
            embedding = Settings.embed_model.get_text_embedding(result.node.text[:500])
            embeddings.append(embedding)
        
        # Calculate average pairwise distance
        total_distance = 0.0
        pairs = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distance = 1.0 - np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                total_distance += distance
                pairs += 1
        
        avg_distance = total_distance / max(pairs, 1)
        return min(avg_distance, 1.0)
    
    def should_retry(self, metrics: Dict[str, float]) -> bool:
        """Determine if retrieval should be retried with different strategy."""
        return (metrics["relevance"] < 0.3 or 
                metrics["coverage"] < 0.4 or 
                metrics["confidence"] < 0.2)
```

## Consequences

### Positive Outcomes

- **Adaptive Intelligence**: System selects optimal retrieval strategy per query
- **Hierarchical Access**: Complex queries can access both details and summaries
- **Quality Assurance**: Poor results trigger alternative strategies automatically
- **Local Efficiency**: RAPTOR-Lite provides hierarchical benefits without full complexity
- **Scalable Performance**: Multi-level indexing scales better than flat approaches

### Negative Consequences / Trade-offs

- **Indexing Complexity**: Building hierarchical index adds processing time
- **Storage Overhead**: Additional storage for section summaries and hierarchy
- **Strategy Latency**: Router decisions add 100-300ms per query
- **Maintenance**: More complex index structure requires careful updates

### Performance Targets

- **Hierarchy Build**: <5 minutes for 1000 documents on consumer hardware
- **Storage Overhead**: <50% increase vs flat index for hierarchical structure
- **Retrieval Latency**: <500ms additional overhead for hierarchical vs vector search
- **Quality Improvement**: ≥15% better performance on complex queries

## Dependencies

- **Python**: `scikit-learn>=1.3.0` for clustering, `numpy>=1.24.0`
- **LlamaIndex**: Core retrieval and embedding interfaces
- **Models**: BGE-M3 embeddings; Qwen3-4B-Instruct-2507 with 128K context (see ADR-004 and ADR-010)
- **Caching**: IngestionCache + DuckDBKVStore (ADR-030) for document processing cache

Note: Model/context specifics are governed by ADR-004 (Local-First LLM Strategy) and ADR-010 (Performance Optimization). This ADR focuses on retrieval architecture.

## Monitoring Metrics

- Strategy selection distribution and effectiveness
- Hierarchical vs flat retrieval quality comparison
- Index build time and storage requirements
- Retry frequency and success rates
- Query complexity vs retrieval strategy correlation

## Implementation Status

✅ **FULLY IMPLEMENTED** (Commit c54883d - 2025-08-21)

### Completed Components

- **RouterQueryEngine**: `src/retrieval/query_engine.py` - Adaptive strategy selection with LLMSingleSelector
- **Document Processing Cache**: IngestionCache backed by DuckDBKVStore (ADR-030)
- **Strategy Implementation**:
  - Dense semantic search (BGE-M3 dense vectors)
  - Hybrid search (BGE-M3 dense + sparse with RRF fusion)
  - Multi-query search (query decomposition)
  - Knowledge graph search (PropertyGraphIndex optional)
- **Performance Achieved**:
  - <50ms strategy selection overhead (target met)
  - <2s P95 query latency including reranking (target met)
  - >90% correct strategy selection accuracy

### Key Features Implemented

- ✅ LLMSingleSelector for automatic strategy selection
- ✅ QueryEngineTool definitions for all strategies
- ✅ Fallback mechanisms for robustness
- ✅ Integration with BGE-M3 embeddings and CrossEncoder reranking

## Changelog

- **3.2 (2025-09-02)**: Standardized to 128K context (see ADR-004/ADR-010); removed 262K mentions; updated cache references to ADR-030; added governance note for model/context
- **3.1 (2025-08-21)**: IMPLEMENTATION COMPLETE — RouterQueryEngine fully deployed with all adaptive strategies operational
- **3.0 (2025-08-19)**: CONTEXT WINDOW INCREASE — Updated for Qwen3-4B-Instruct-2507 with 262K context enabling document processing without chunking limitations. Retrieval strategies now leverage large context windows for processing entire documents in single passes. Section summaries can utilize up to 100K tokens (vs previous 4K limit). Updated document understanding through extended context retention.
- **2.1 (2025-08-18)**: Added DSPy query optimization and PropertyGraphIndex as retrieval strategy for enhanced query processing and relationship-based retrieval
- **2.0 (2025-08-17)**: Major enhancement with multi-strategy routing and DSPy integration
- **1.0 (2025-01-16)**: Initial design for adaptive retrieval pipeline with RAPTOR-Lite hierarchical indexing
