"""Dependency injection container for DocMind AI.

This module provides centralized dependency injection configuration using
dependency-injector framework. It follows library-first principles and
implements clean architecture patterns for testability and maintainability.

Key features:
- Configuration provider for environment-based settings
- Factory providers for stateless services
- Singleton providers for stateful resources
- Easy testing through provider overrides
- Environment-aware configuration loading
"""

import os
from typing import Any

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from loguru import logger

from src.config import settings


class ApplicationContainer(containers.DeclarativeContainer):
    """Main dependency injection container for DocMind AI.

    Provides centralized configuration and dependency management following
    clean architecture principles. All dependencies are configured here
    and can be easily overridden for testing.
    """

    # Configuration provider - loads from environment
    config = providers.Configuration()

    # Cache dependencies
    cache = providers.Singleton(
        "src.cache.simple_cache.SimpleCache",
        cache_dir=config.cache_dir.as_(str, default="./cache"),
    )

    # Embedding dependencies
    embedding_model = providers.Factory(
        "src.retrieval.embeddings.create_bgem3_embedding",
        model_name=config.embedding_model.as_(
            str, default=settings.embedding.model_name
        ),
        use_fp16=config.use_fp16.as_(bool, default=True),
        device=config.device.as_(str, default="cuda"),
    )

    # Document processing dependencies
    document_processor = providers.Factory(
        "src.processing.document_processor.DocumentProcessor",
        settings=providers.Object(settings),
    )

    # Multi-agent coordinator (when not in testing mode)
    multi_agent_coordinator = providers.Singleton(
        "src.agents.coordinator.MultiAgentCoordinator",
        model_path=config.model_path.as_(str, default=settings.vllm.model),
        max_context_length=config.max_context_length.as_(
            int, default=settings.vllm.context_window
        ),
        enable_fallback=config.enable_fallback.as_(bool, default=True),
    )



def create_container() -> ApplicationContainer:
    """Create and configure the appropriate container.

    Returns:
        ApplicationContainer or MockContainer based on environment
    """
    container = ApplicationContainer()
    logger.info("Created ApplicationContainer")

    # Load configuration from environment
    container.config.from_env("DOCMIND")
    container.config.from_dict(
        {
            "cache_dir": os.getenv("DOCMIND_CACHE_DIR", "./cache"),
            "embedding_model": os.getenv(
                "DOCMIND_EMBEDDING_MODEL", settings.embedding.model_name
            ),
            "model_path": os.getenv("DOCMIND_MODEL_NAME", settings.vllm.model),
            "max_context_length": int(
                os.getenv(
                    "DOCMIND_CONTEXT_WINDOW_SIZE", str(settings.vllm.context_window)
                )
            ),
            "device": os.getenv("DOCMIND_DEVICE", "cuda"),
            "use_fp16": os.getenv("DOCMIND_USE_FP16", "true").lower() == "true",
            "enable_fallback": os.getenv("DOCMIND_ENABLE_FALLBACK_RAG", "true").lower()
            == "true",
        }
    )

    return container


# Global container instance
container = create_container()


def get_container() -> ApplicationContainer:
    """Get the global container instance.

    Returns:
        The global ApplicationContainer instance
    """
    return container


def wire_container(modules: list[str]) -> None:
    """Wire the container to specified modules.

    Args:
        modules: List of module names to wire
    """
    container.wire(modules=modules)


def unwire_container() -> None:
    """Unwire the container."""
    container.unwire()


# Example dependency injection functions
@inject
def get_cache(cache=Provide[ApplicationContainer.cache]) -> Any:
    """Get cache instance with dependency injection."""
    return cache


@inject
def get_embedding_model(
    embedding_model=Provide[ApplicationContainer.embedding_model],
) -> Any:
    """Get embedding model with dependency injection."""
    return embedding_model


@inject
def get_document_processor(
    processor=Provide[ApplicationContainer.document_processor],
) -> Any:
    """Get document processor with dependency injection."""
    return processor


@inject
def get_multi_agent_coordinator(
    coordinator=Provide[ApplicationContainer.multi_agent_coordinator],
) -> Any:
    """Get multi-agent coordinator with dependency injection."""
    return coordinator


# Factory functions for manual instantiation (when not using injection)
def create_cache(**kwargs) -> Any:
    """Create cache instance manually."""
    return container.cache(**kwargs)


def create_embedding_model(**kwargs) -> Any:
    """Create embedding model manually."""
    return container.embedding_model(**kwargs)


def create_document_processor(**kwargs) -> Any:
    """Create document processor manually."""
    return container.document_processor(**kwargs)


def create_multi_agent_coordinator(**kwargs) -> Any:
    """Create multi-agent coordinator manually."""
    return container.multi_agent_coordinator(**kwargs)
