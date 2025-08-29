#!/usr/bin/env python3
"""DocMind AI Python API Usage Example.

This example demonstrates how to use the DocMind AI internal Python API
for document analysis and multi-agent coordination.
"""

import asyncio
from pathlib import Path

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

# DocMind AI imports
from src.agents.coordinator import MultiAgentCoordinator
from src.config import settings


async def main():
    """Main example demonstrating DocMind AI capabilities."""
    print("DocMind AI Python API Example")
    print("=" * 40)

    # 1. Configuration
    print("\n1. Configuration:")
    print(f"   Model: {settings.vllm.model}")
    print(f"   Context Window: {settings.vllm.context_window:,} tokens")
    print(f"   GPU Enabled: {settings.enable_gpu_acceleration}")
    print(f"   Multi-Agent: {settings.agents.enable_multi_agent}")

    # 2. Initialize Multi-Agent System
    print("\n2. Initializing Multi-Agent System...")
    coordinator = MultiAgentCoordinator(
        model_path=settings.vllm.model,
        max_context_length=settings.vllm.context_window,
        backend="vllm",
        enable_fallback=True,
        max_agent_timeout=200,
    )

    # 3. Create Memory Buffer
    print("\n3. Setting up conversation memory...")
    memory = ChatMemoryBuffer.from_defaults(token_limit=settings.vllm.context_window)

    # 4. Load Documents
    print("\n4. Loading documents...")
    document_paths = [
        "/path/to/Q4_2024_Financial_Report.pdf",
        "/path/to/Healthcare_Division_Analysis.xlsx",
        "/path/to/Technology_Metrics_Q4.docx",
    ]

    # In a real scenario, you would have actual document paths
    # For this example, we'll simulate document loading
    print(f"   Documents to process: {len(document_paths)}")
    for path in document_paths:
        print(f"   - {Path(path).name}")

    # Uncomment this line when you have actual documents:
    # documents = await load_documents_unstructured(document_paths, settings)

    # For demonstration, we'll proceed without actual documents
    documents = []  # Placeholder

    # 5. Process Queries
    print("\n5. Processing queries with multi-agent coordination...")

    queries = [
        "What are the key financial trends in Q4 2024?",
        "How did the Technology division perform compared to Healthcare?",
        "What strategic recommendations emerge from the quarterly analysis?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: {query}")

        try:
            # Add user query to memory
            memory.put(ChatMessage(role="user", content=query))

            # Process query with multi-agent coordination
            response = await coordinator.aprocess_query(
                query=query,
                context=memory,
                documents=documents,  # In real usage, pass actual documents
                options={
                    "enable_citations": True,
                    "performance_mode": "balanced",
                    "include_agent_trace": True,
                },
            )

            # Add response to memory
            memory.put(ChatMessage(role="assistant", content=response.content))

            # Display results
            print(f"   Response ({response.execution_time:.2f}s):")
            print(f"   {response.content[:200]}...")
            print(f"   Confidence: {response.confidence:.2f}")
            print(f"   Agents Used: {len(response.agents_invoked)}")

            if hasattr(response, "citations") and response.citations:
                print(f"   Citations: {len(response.citations)}")
                for citation in response.citations[:2]:  # Show first 2
                    print(f"     - {citation.source}: {citation.relevance:.2f}")

        except Exception as e:
            print(f"   Error processing query: {e}")

    # 6. System Status
    print("\n6. System Performance:")
    try:
        status = coordinator.get_system_status()
        print(f"   Overall Status: {status.overall_status}")
        print(f"   Active Agents: {len(status.active_agents)}")
        print(f"   GPU Utilization: {status.gpu_utilization}%")
        print(f"   VRAM Usage: {status.vram_usage_gb:.1f}GB / {status.vram_total_gb}GB")
    except Exception as e:
        print(f"   Status unavailable: {e}")

    # 7. Memory Usage
    print("\n7. Conversation Memory:")
    messages = memory.get_all()
    token_usage = memory.get_all_token_count()
    print(f"   Total Messages: {len(messages)}")
    print(f"   Token Usage: {token_usage:,} / {memory.token_limit:,}")
    print(f"   Context Utilization: {token_usage / memory.token_limit:.1%}")

    print("\n" + "=" * 40)
    print("Example completed successfully!")


async def document_search_example():
    """Example of advanced document search capabilities."""
    print("\nDocument Search Example")
    print("-" * 30)

    # This would work with actual documents
    # documents = await load_documents_unstructured(file_paths, settings)
    # index = await create_index_async(documents, settings)

    print("Search capabilities:")
    print("- Hybrid search (dense + sparse embeddings)")
    print("- BGE-M3 unified embeddings")
    print("- Reranking with BGE-reranker-v2-m3")
    print("- RRF fusion with α=0.7")
    print("- Semantic filtering and relevance scoring")


async def streaming_example():
    """Example of streaming analysis."""
    print("\nStreaming Analysis Example")
    print("-" * 30)

    _coordinator = MultiAgentCoordinator()

    print("Streaming features:")
    print("- Real-time response generation")
    print("- Progress tracking with agent updates")
    print("- Chunked content delivery")
    print("- WebSocket support (REST API)")

    # Simulated streaming output
    async def simulate_stream():
        updates = [
            {"type": "agent_start", "agent": "query_router", "task": "Analyzing query"},
            {"type": "agent_complete", "agent": "query_router", "duration": 0.12},
            {
                "type": "agent_start",
                "agent": "retrieval_expert",
                "task": "Searching documents",
            },
            {"type": "response_chunk", "content": "Based on the quarterly analysis"},
            {"type": "response_chunk", "content": ", revenue growth accelerated"},
            {"type": "agent_complete", "agent": "retrieval_expert", "duration": 1.23},
            {"type": "response_chunk", "content": " to 18% in Q4 2024."},
            {"type": "complete", "total_time": 2.45, "confidence": 0.89},
        ]

        for update in updates:
            await asyncio.sleep(0.2)  # Simulate processing time
            if update["type"] == "agent_start":
                print(f"🚀 {update['agent']} started: {update['task']}")
            elif update["type"] == "agent_complete":
                print(f"✅ {update['agent']} completed in {update['duration']}s")
            elif update["type"] == "response_chunk":
                print(update["content"], end="", flush=True)
            elif update["type"] == "complete":
                print(f"\n\n📊 Analysis completed in {update['total_time']}s")
                print(f"   Confidence: {update['confidence']:.2f}")

    await simulate_stream()


def configuration_example():
    """Example of configuration management."""
    print("\nConfiguration Management Example")
    print("-" * 35)

    print("Current configuration:")
    print(f"  Model: {settings.vllm.model}")
    print(f"  Context Window: {settings.vllm.context_window:,}")
    print(f"  GPU Memory: {settings.vllm.gpu_memory_utilization}")
    print(f"  Agent Timeout: {settings.agents.decision_timeout}ms")

    print("\nDynamic configuration:")
    print("- Runtime settings modification")
    print("- Hardware optimization")
    print("- Environment variable overrides")
    print("- Configuration validation")

    # Example configuration patterns
    print("\nConfiguration patterns:")
    print("  # Temporary override")
    print("  with settings.temporary_override({")
    print("      'agents.decision_timeout': 500,")
    print("      'retrieval.top_k': 15")
    print("  }):")
    print("      # Operations with modified settings")
    print("      pass")


if __name__ == "__main__":
    """Run the examples."""

    print("DocMind AI Python API Examples")
    print("============================\n")

    # Run main example
    asyncio.run(main())

    # Additional examples
    asyncio.run(document_search_example())
    asyncio.run(streaming_example())
    configuration_example()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("")
    print("Next steps:")
    print("1. Install dependencies: uv sync --extra gpu")
    print("2. Configure environment: cp .env.example .env")
    print("3. Start services: ollama serve")
    print("4. Run your own analysis with actual documents")
    print("")
    print("For more information:")
    print("- Documentation: docs/api/internal-api.md")
    print("- Examples: docs/api/examples/")
    print("- Developer Guide: docs/developers/")
