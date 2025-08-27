"""Dependency injection demonstration module.

This module demonstrates the use of @inject decorators and dependency injection
patterns throughout the DocMind AI application. It serves as an example of how
to properly implement dependency injection with the ApplicationContainer.
"""

from typing import Any

from dependency_injector.wiring import Provide, inject

from src.containers import ApplicationContainer
from src.interfaces import CacheInterface


@inject
def get_cache_with_injection(
    cache: CacheInterface = Provide[ApplicationContainer.cache],
) -> CacheInterface:
    """Get cache instance using dependency injection.

    Args:
        cache: Injected cache instance

    Returns:
        CacheInterface instance
    """
    return cache


@inject
def get_embedding_model_with_injection(
    embedding_model=Provide[ApplicationContainer.embedding_model],
) -> Any:
    """Get embedding model using dependency injection.

    Args:
        embedding_model: Injected embedding model instance

    Returns:
        Embedding model instance
    """
    return embedding_model


@inject
def get_document_processor_with_injection(
    document_processor=Provide[ApplicationContainer.document_processor],
) -> Any:
    """Get document processor using dependency injection.

    Args:
        document_processor: Injected document processor instance

    Returns:
        Document processor instance
    """
    return document_processor


@inject
def get_multi_agent_coordinator_with_injection(
    coordinator=Provide[ApplicationContainer.multi_agent_coordinator],
) -> Any:
    """Get multi-agent coordinator using dependency injection.

    Args:
        coordinator: Injected coordinator instance

    Returns:
        MultiAgentCoordinator instance
    """
    return coordinator


@inject
async def process_document_with_injection(
    file_path: str,
    cache: CacheInterface = Provide[ApplicationContainer.cache],
    document_processor=Provide[ApplicationContainer.document_processor],
) -> Any:
    """Process a document using injected dependencies.

    Args:
        file_path: Path to document to process
        cache: Injected cache instance
        document_processor: Injected document processor instance

    Returns:
        Processing result
    """
    # Check cache first
    cached_result = await cache.get_document(file_path)
    if cached_result:
        return cached_result

    # Process document
    result = await document_processor.process_document_async(file_path)

    # Store in cache
    await cache.store_document(file_path, result)

    return result


def demo_dependency_injection() -> None:
    """Demonstrate dependency injection functionality.

    This function shows how to wire the container and use injected dependencies.
    """
    from src.containers import wire_container

    # Wire the container to this module
    wire_container([__name__])

    try:
        # Test dependency injection
        cache = get_cache_with_injection()
        print(f"Cache injected: {type(cache).__name__}")

        embedding_model = get_embedding_model_with_injection()
        print(f"Embedding model injected: {type(embedding_model).__name__}")

        document_processor = get_document_processor_with_injection()
        print(f"Document processor injected: {type(document_processor).__name__}")

        coordinator = get_multi_agent_coordinator_with_injection()
        print(f"Coordinator injected: {type(coordinator).__name__}")

        print("Dependency injection working correctly!")

    except Exception as e:
        print(f"Dependency injection failed: {e}")

    finally:
        # Clean up
        from src.containers import unwire_container

        unwire_container()


if __name__ == "__main__":
    demo_dependency_injection()
