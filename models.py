"""Pydantic models for DocMind AI.

This module defines the data models and configuration schemas used throughout
the DocMind AI application. It includes structured output models for document
analysis results and application settings loaded from environment variables.

The models ensure type safety, data validation, and provide clear interfaces
for the application's core data structures.

Example:
    Using the models::

        from models import AnalysisOutput, Settings

        # Create structured analysis output
        result = AnalysisOutput(
            summary="Document summary",
            key_insights=["insight1", "insight2"],
            action_items=["action1"],
            open_questions=["question1"]
        )

        # Load application settings
        settings = Settings()
        print(settings.ollama_base_url)

Classes:
    AnalysisOutput: Structured schema for document analysis results.
    Settings: Application configuration loaded from environment variables.

"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnalysisOutput(BaseModel):
    """Structured output schema for document analysis results.

    Defines the expected format for analysis results from the language model,
    ensuring consistent structure for summaries, insights, action items, and
    questions. Used with Pydantic output parsing to validate and structure
    LLM responses.

    Attributes:
        summary: Brief overview of the document content and main themes.
        key_insights: List of important findings and observations extracted
            from the document.
        action_items: List of concrete steps or tasks identified from the
            document content.
        open_questions: List of unresolved questions or areas requiring
            further investigation.

    """

    summary: str = Field(description="Summary of the document")
    key_insights: list[str] = Field(description="Key insights extracted")
    action_items: list[str] = Field(description="Action items identified")
    open_questions: list[str] = Field(description="Open questions surfaced")


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables.

    Manages all configurable parameters for the DocMind AI application,
    including backend URLs, model specifications, and service endpoints.
    Settings are automatically loaded from environment variables or .env file.

    Attributes:
        backend: Default backend type for LLM inference.
        ollama_base_url: Base URL for Ollama server API.
        lmstudio_base_url: Base URL for LM Studio server API.
        llamacpp_model_path: File path to the Llama.cpp model file.
        default_model: Default model name/identifier to use.
        context_size: Maximum context window size for language models.
        qdrant_url: URL for the Qdrant vector database server.
        default_embedding_model: Model name for document embeddings.
        default_reranker_model: Model name for document reranking.
        model_config: Pydantic configuration for settings loading.

    """

    backend: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    llamacpp_model_path: str = "/path/to/model.gguf"
    default_model: str = "Qwen/Qwen3-8B"
    context_size: int = 4096
    qdrant_url: str = "http://localhost:6333"
    default_embedding_model: str = "jinaai/jina-embeddings-v4"
    default_reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"

    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, case_sensitive=False
    )
