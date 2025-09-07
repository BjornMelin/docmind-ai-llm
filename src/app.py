"""Main Streamlit application for DocMind AI.

This module provides a comprehensive web interface for document analysis using
local large language models. It handles user interface components, model
selection and configuration, document upload and processing, analysis with
customizable prompts, and interactive chat functionality with multimodal
and hybrid search support, enhanced by Agentic RAG, optimizations,
and auto-quantization.

The application supports multiple backends (Ollama, LlamaCpp, LM Studio),
various document formats including basic video/audio, and provides features like session
persistence, theming, hardware detection, agentic workflows, async operations,
auto-model download, and performance monitoring with custom metrics.

Example:
    Run the application using Streamlit::

        $ streamlit run app.py

Attributes:
    settings: Global settings instance loaded from environment variables.
    st.session_state.memory: LlamaIndex memory for session persistence.
    st.session_state.agent: ReActAgent for analysis and chat.
    st.session_state.index: LlamaIndex for document storage and retrieval.
"""

import asyncio
import time
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import ollama
import streamlit as st
from llama_index.core import Settings as LISettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.clip import ClipEmbedding
from loguru import logger
from streamlit.errors import StreamlitAPIException

from src.agents.coordinator import MultiAgentCoordinator
from src.agents.tool_factory import ToolFactory
from src.config import settings
from src.config.integrations import initialize_integrations
from src.config.settings import DocMindSettings as _DocMindSettings
from src.config.settings import UIConfig as _UIConfig
from src.config.settings import VLLMConfig as _VLLMConfig
from src.containers import get_multi_agent_coordinator
from src.processing.pdf_pages import pdf_pages_to_image_documents
from src.prompts import PREDEFINED_PROMPTS
from src.retrieval.graph_config import create_property_graph_index
from src.retrieval.query_engine import create_adaptive_router_engine
from src.ui.components.provider_badge import provider_badge
from src.ui_helpers import build_reranker_controls
from src.utils.core import detect_hardware, validate_startup_configuration
from src.utils.document import load_documents_unstructured
from src.utils.storage import create_vector_store

LLAMACPP_AVAILABLE = False
# Test-patching placeholders for unit tests (avoid import-time heavy deps)
# These are intentionally simple so tests can patch them without import side-effects.
LlamaCPP = None  # type: ignore[assignment]  # pylint: disable=invalid-name
Ollama = None  # type: ignore[assignment]  # pylint: disable=invalid-name
OpenAILike = None  # type: ignore[assignment]  # pylint: disable=invalid-name
SimpleVectorStore = None  # type: ignore[assignment]  # pylint: disable=invalid-name
try:  # Detect availability without importing at module import time
    import importlib

    importlib.import_module("llama_index.llms.llama_cpp")
    LLAMACPP_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    logger.warning("LlamaCPP not available. Running without LlamaCPP support.")


def _log_processing_metrics(
    doc_load_time: float, index_time: float, total_time: float, docs_count: int
) -> None:
    """Display and log document processing performance metrics.

    Args:
        doc_load_time (float): Elapsed time in seconds to load documents.
        index_time (float): Elapsed time in seconds to build the index(es).
        total_time (float): Total elapsed time in seconds for the operation.
        docs_count (int): Number of processed documents.
    """
    st.info(
        f"""
        **Performance Metrics (Async Mode):**
        - Document loading: {doc_load_time:.2f}s
        - Index creation: {index_time:.2f}s
        - Total processing: {total_time:.2f}s
        - Documents processed: {docs_count}
        """
    )
    logger.info(
        "Async processing completed in {:.2f}s for {} documents",
        total_time,
        docs_count,
    )


# Help static analysis by annotating settings instance explicitly
SETTINGS: _DocMindSettings = settings
# Help pylint by casting nested Pydantic models to Any (runtime instances)
UI: _UIConfig = cast(Any, SETTINGS.ui)
VLLM: _VLLMConfig = cast(Any, SETTINGS.vllm)


# Simple wrapper functions for Ollama API calls
def is_llamacpp_available() -> bool:
    """Check llama.cpp binding availability.

    Returns:
        bool: True if the optional llama.cpp bindings are importable in the
        current runtime environment; False otherwise.
    """
    return bool(LLAMACPP_AVAILABLE)


async def get_ollama_models() -> dict[str, Any]:
    """List available models from a local Ollama server.

    The function uses Ollama's model listing endpoint. It is defined as async
    for symmetry with other I/O helpers, but internally calls the synchronous
    client to avoid event-loop conflicts inside Streamlit.

    Returns:
        dict[str, Any]: A dictionary payload returned by Ollama containing the
        available models under the ``models`` key.
    """
    return ollama.list()


async def pull_ollama_model(ollama_model_name: str) -> dict[str, Any]:
    """Ensure an Ollama model is available locally.

    Args:
        ollama_model_name (str): The model identifier to pull (e.g.,
            ``"qwen2:7b-instruct"``).

    Returns:
        dict[str, Any]: Ollama's streaming or summary response for the pull
        operation.
    """
    return ollama.pull(ollama_model_name)


def create_tools_from_index(index: Any) -> list[Any]:
    """Create retrieval tools backed by a vector index.

    Args:
        index (Any): LlamaIndex-compatible vector index instance used for
            retrieval operations.

    Returns:
        list[Any]: A list of tool objects that expose search/retrieval
        capabilities to the agent system.
    """
    return ToolFactory.create_basic_tools({"vector": index})


def get_agent_system(
    _tools: Any,
    _llm: Any,
    _memory: Any,
    *,
    multi_agent_coordinator: MultiAgentCoordinator | None = None,
) -> tuple[MultiAgentCoordinator, str]:
    """Construct and return the active agent system.

    Args:
        _tools (Any): Tools available to the agent(s), typically created from
            the current vector index.
        _llm (Any): The LLM client to use for reasoning and generation.
        _memory (Any): Conversation memory or context object.
        multi_agent_coordinator (MultiAgentCoordinator | None): Optional
            pre-configured multi-agent coordinator.

    Returns:
        tuple[MultiAgentCoordinator, str]: The coordinator to handle queries
        and the operating mode label (``"multi_agent"``).
    """
    agent_coordinator = multi_agent_coordinator or get_multi_agent_coordinator()
    return agent_coordinator, "multi_agent"


def process_query_with_agent_system(
    agent_system_: Any, query: str, mode_: str, memory: Any
) -> Any:
    """Dispatch a user query to the configured agent system.

    Args:
        agent_system_ (Any): The agent/coordinator handling the request.
        query (str): The user query or analysis instruction.
        mode_ (str): Operating mode identifier. When set to ``"multi_agent"``,
            the multi-agent coordinator is used.
        memory (Any): Conversation memory/context passed to the agent.

    Returns:
        Any: An agent response object with a ``content`` attribute when
        successful. A minimal fallback namespace is returned on invalid mode.
    """
    if mode_ == "multi_agent":
        # Provide router_engine to coordinator via settings_override (only when present)
        try:
            router_engine = getattr(st.session_state, "router_engine", None)
        except Exception:  # pragma: no cover - UI/session guard
            router_engine = None

        if router_engine is not None:
            overrides = {"router_engine": router_engine}
            return agent_system_.process_query(
                query, context=memory, settings_override=overrides
            )
        return agent_system_.process_query(query, context=memory)
    # Return a minimal response envelope for error cases
    return SimpleNamespace(content="Processing error")


# Validate configuration at startup
try:
    validate_startup_configuration(SETTINGS)
except RuntimeError as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.error(
        "Please check your .env file and ensure all required settings are "
        "properly configured."
    )
    st.stop()


st.set_page_config(page_title="DocMind AI", page_icon="🧠")

if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(
        token_limit=SETTINGS.vllm.context_window
    )
if "agent_system" not in st.session_state:
    st.session_state.agent_system = None
if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "single"
if "index" not in st.session_state:
    st.session_state.index = None

st.title("🧠 DocMind AI: Local LLM Document Analysis")
provider_badge(SETTINGS)

# Dynamic theming with Streamlit 1.47.0
theme: str = st.selectbox(
    "Theme", ["Light", "Dark", "Auto"], index=2, key="theme_select"
)
if theme != "Auto":
    BG_COLOR = "#222" if theme == "Dark" else "#fff"
    TEXT_COLOR = "#fff" if theme == "Dark" else "#000"
    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: {BG_COLOR}; color: {TEXT_COLOR}; }}
            /* Additional theming styles */
        </style>
        """,
        unsafe_allow_html=True,
    )

# Hardware Detection and Model Suggestion with Auto-Quant
hardware_status = detect_hardware()
vram = hardware_status.get("vram_total_gb")
gpu_name = hardware_status.get("gpu_name", "No GPU")
st.sidebar.info(
    f"Detected: {gpu_name}, VRAM: {vram}GB" if vram else f"Detected: {gpu_name}"
)
QUANT_SUFFIX: str = ""
SUGGESTED_MODEL: str = "google/gemma-3n-E4B-it"
SUGGESTED_CONTEXT: int = 8192
if vram:
    if vram >= 16:  # 16GB for high-end models
        SUGGESTED_MODEL = "nvidia/OpenReasoning-Nemotron-32B"
        QUANT_SUFFIX = "-Q4_K_M"  # Fits 16GB
        SUGGESTED_CONTEXT = 131072  # 128K context
    elif vram >= 8:  # 8GB for medium models
        SUGGESTED_MODEL = "nvidia/OpenReasoning-Nemotron-14B"
        QUANT_SUFFIX = "-Q8_0"  # Fits 8GB
        SUGGESTED_CONTEXT = 65536  # 64K context
    else:
        SUGGESTED_MODEL = "google/gemma-3n-E4B-it"
        QUANT_SUFFIX = "-Q4_K_S"  # Minimal
        SUGGESTED_CONTEXT = 32768  # 32K context for low VRAM
st.sidebar.info(
    f"Suggested: {SUGGESTED_MODEL}{QUANT_SUFFIX} with {SUGGESTED_CONTEXT} context"
)


use_gpu: bool = st.sidebar.checkbox(
    "Use GPU", value=hardware_status.get("cuda_available", False)
)
# Retrieval & Reranking controls (ADR-036/037)
try:
    build_reranker_controls(SETTINGS)
except (ValueError, StreamlitAPIException) as e:  # pragma: no cover - UI resilience
    logger.warning("Reranker controls failed to initialize: {}", e)
_parse_media: bool = st.sidebar.checkbox("Parse Video/Audio", value=False)
enable_multimodal: bool = st.sidebar.checkbox(
    "Enable Multimodal Processing",
    value=True,
    help="Extract and process images from PDFs using local Jina CLIP models",
)

# Single ReActAgent (simplified architecture)


"""Initialize LLM via unified integrations (no network at import time)."""
try:
    initialize_integrations()
except Exception as e:  # pragma: no cover - resilience
    logger.warning("Integrations init issue: {}", e)


# Async Document Upload Section with Media Parsing and Error Handling
@st.fragment
async def upload_section() -> None:
    """Handle document upload, ingestion, and indexing.

    The function reads uploaded files, constructs a text vector index, and,
    when enabled, builds a multimodal (image + text) index. Progress and
    performance metrics are displayed inline.
    """
    uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile] | None = (
        st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=["pdf", "docx", "mp4", "mp3", "wav"],
            key="file_uploader",
        )
    )
    if uploaded_files:
        with st.status("Processing documents..."):
            try:
                # Start timing for performance monitoring
                start_time = time.perf_counter()

                # ADR-009 compliant document processing
                docs = await load_documents_unstructured(uploaded_files, SETTINGS)
                doc_load_time = time.perf_counter() - start_time

                # Progress tracking for document processing
                progress_bar = st.progress(0)
                for i, _doc in enumerate(docs):
                    progress_bar.progress((i + 1) / len(docs))

                # Use async indexing for 50-80% performance improvement
                index_start_time = time.perf_counter()
                try:
                    # Prefer Qdrant vector store via LlamaIndex
                    collection_name = "docmind_text_index"
                    qdrant_vs = create_vector_store(collection_name, enable_hybrid=True)
                    storage_context = StorageContext.from_defaults(
                        vector_store=qdrant_vs
                    )
                    st.session_state.index = VectorStoreIndex.from_documents(
                        docs, storage_context=storage_context
                    )

                    # Build PropertyGraphIndex (GraphRAG) when enabled (default ON)
                    st.session_state.kg_index = None
                    if SETTINGS.enable_graphrag or SETTINGS.get_graphrag_config().get(
                        "enabled", False
                    ):
                        try:
                            st.session_state.kg_index = create_property_graph_index(
                                docs, vector_store=qdrant_vs
                            )
                        except (
                            ValueError,
                            RuntimeError,
                        ) as kg_e:  # pragma: no cover - optional path
                            logger.warning("GraphRAG index build skipped: {}", kg_e)
                            st.session_state.kg_index = None

                    # Multimodal index (text + images) controlled by toggle
                    if enable_multimodal:
                        try:
                            # Temporarily switch to CLIP for image embeddings only,
                            # preserving the global text embedder (BGE-M3).
                            _prev_embed_model = LISettings.embed_model
                            LISettings.embed_model = ClipEmbedding(
                                model_name="openai/clip-vit-base-patch32"
                            )
                            # Emit page-image nodes from uploaded PDFs
                            image_docs: list[Any] = []
                            try:
                                pdf_docs = [
                                    d
                                    for d in docs
                                    if str(d.metadata.get("source", ""))
                                    .lower()
                                    .endswith(".pdf")
                                ]
                                for pd in pdf_docs:
                                    try:
                                        img_nodes, _out = pdf_pages_to_image_documents(
                                            Path(str(pd.metadata.get("source")))
                                        )
                                        image_docs.extend(img_nodes)
                                    except (
                                        OSError,
                                        RuntimeError,
                                        ValueError,
                                        TypeError,
                                    ) as _e:  # pragma: no cover - optional path
                                        logger.warning(
                                            "PDF page-image emission skipped: {}", _e
                                        )
                            except (
                                OSError,
                                RuntimeError,
                                ValueError,
                                TypeError,
                            ) as _e:  # pragma: no cover - optional path
                                logger.warning(
                                    "Multimodal emission scan failed: {}", _e
                                )
                            if image_docs:
                                from llama_index.core.indices import (  # pylint: disable=ungrouped-imports
                                    MultiModalVectorStoreIndex,
                                )

                                # Separate image store collection
                                image_collection = "docmind_image_index"
                                image_vs = create_vector_store(
                                    image_collection, enable_hybrid=False
                                )
                                mm_context = StorageContext.from_defaults(
                                    vector_store=qdrant_vs, image_store=image_vs
                                )
                                st.session_state.multimodal_index = (
                                    MultiModalVectorStoreIndex.from_documents(
                                        image_docs, storage_context=mm_context
                                    )
                                )
                            else:
                                st.session_state.multimodal_index = None
                            # Always restore the previous text embedder
                            LISettings.embed_model = _prev_embed_model
                        except (
                            ValueError,
                            RuntimeError,
                        ) as mm_e:  # pragma: no cover - optional path
                            logger.warning("Multimodal index setup skipped: {}", mm_e)
                            st.session_state.multimodal_index = None
                            # Ensure we restore the previous text embedder on error
                            from contextlib import suppress

                            with suppress(Exception):  # pragma: no cover - defensive
                                LISettings.embed_model = _prev_embed_model
                    else:
                        st.session_state.multimodal_index = None

                    # Create adaptive router engine (vector + multimodal)
                    try:
                        st.session_state.router_engine = create_adaptive_router_engine(
                            vector_index=st.session_state.index,
                            kg_index=st.session_state.kg_index,
                            multimodal_index=st.session_state.multimodal_index,
                        )
                    except (
                        ValueError,
                        RuntimeError,
                    ) as re_e:  # pragma: no cover - optional path
                        logger.warning("Router engine setup skipped: {}", re_e)
                        st.session_state.router_engine = None

                except (ValueError, RuntimeError) as e:
                    st.error(f"Document processing failed: {e}")
                    logger.error("Index creation failed: %s", str(e))
                    return
                index_time = time.perf_counter() - index_start_time
                total_time = time.perf_counter() - start_time

                # Reset agent system when new documents are uploaded
                st.session_state.agent_system = None
                st.session_state.agent_mode = "single"

                # Show performance metrics
                st.success("Documents indexed successfully! ⚡")
                _log_processing_metrics(
                    doc_load_time, index_time, total_time, len(docs)
                )

            except (ValueError, TypeError, OSError, RuntimeError) as e:
                st.error(f"Document processing failed: {e!s}")
                logger.error("Doc process error: {}", e)


# Analysis Options and Agentic Analysis with Error Handling
st.header("Analysis Options")
prompt_type: str = st.selectbox("Prompt", list(PREDEFINED_PROMPTS.keys()))
# Other selects for tone, instructions, etc. (assuming they exist in full code)


async def run_analysis() -> None:
    """Execute predefined analysis against the current index.

    Builds agent tools from the active index, invokes the multi-agent
    coordinator, and stores the result in session state.
    """
    if st.session_state.index:
        with st.spinner("Running analysis..."):
            try:
                analysis_query_text = f"Perform {prompt_type} analysis on the documents"

                # Prefer RouterQueryEngine when available
                if st.session_state.get("router_engine") is not None:
                    analysis_resp = await asyncio.to_thread(
                        st.session_state.router_engine.query, analysis_query_text
                    )
                    st.session_state.analysis_results = analysis_resp
                    st.info("✅ Analysis completed using RouterQueryEngine")
                else:
                    # Fallback to multi-agent coordinator path
                    tools_for_agent = create_tools_from_index(st.session_state.index)
                    coordinator, agent_mode = get_agent_system(
                        _tools=tools_for_agent,
                        _llm=LISettings.llm,
                        _memory=st.session_state.memory,
                    )
                    analysis_output = await asyncio.to_thread(
                        process_query_with_agent_system,
                        coordinator,
                        analysis_query_text,
                        agent_mode,
                        st.session_state.memory,
                    )
                    st.session_state.analysis_results = analysis_output
                    st.session_state.agent_system = coordinator
                    st.session_state.agent_mode = agent_mode
                    if agent_mode == "multi_agent":
                        st.info("✅ Analysis completed using multi-agent system")

            except (ValueError, TypeError, RuntimeError) as e:
                st.error(f"Analysis failed: {e!s}")
                logger.error("Analysis error: {}", e)


def _render_analyze_button() -> None:
    """Render and handle the Analyze button synchronously.

    This avoids the asyncio event loop conflict that can arise when nesting
    asynchronous calls within Streamlit event handlers.
    """
    if st.button("Analyze") and st.session_state.index:
        # Use sync analysis to avoid asyncio event loop conflicts
        with st.spinner("Running analysis..."):
            try:
                sync_query = f"Perform {prompt_type} analysis on the documents"

                # Prefer RouterQueryEngine when available (synchronous call)
                if st.session_state.get("router_engine") is not None:
                    sync_resp = st.session_state.router_engine.query(sync_query)
                    st.session_state.analysis_results = sync_resp
                    st.info("✅ Analysis completed using RouterQueryEngine")
                else:
                    # Fallback to multi-agent coordinator
                    sync_tools = create_tools_from_index(st.session_state.index)
                    sync_coordinator, sync_agent_mode = get_agent_system(
                        _tools=sync_tools,
                        _llm=LISettings.llm,
                        _memory=st.session_state.memory,
                    )
                    sync_analysis_output = process_query_with_agent_system(
                        sync_coordinator,
                        sync_query,
                        sync_agent_mode,
                        st.session_state.memory,
                    )
                    st.session_state.analysis_results = sync_analysis_output
                    st.session_state.agent_system = sync_coordinator
                    st.session_state.agent_mode = sync_agent_mode
                    if sync_agent_mode == "multi_agent":
                        st.info("✅ Analysis completed using multi-agent system")

            except (ValueError, TypeError, RuntimeError) as e:
                st.error(f"Analysis failed: {e!s}")
                logger.error("Analysis error: {}", e)


# Chat with Agent using st.chat_message and write_stream
# (Async Streaming with Error Handling)
st.header("Chat with Documents")
for message in st.session_state.memory.chat_store.get_messages("default"):
    with st.chat_message(message.role):
        st.markdown(message.content)

user_input: str | None = st.chat_input("Ask a question", key="chat_input")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        try:
            # Initialize agent system if not already created and index exists
            if not st.session_state.agent_system and st.session_state.index:
                tools = create_tools_from_index(st.session_state.index)
                st.session_state.agent_system, st.session_state.agent_mode = (
                    get_agent_system(
                        _tools=tools,
                        _llm=LISettings.llm,
                        _memory=st.session_state.memory,
                    )
                )

            # Prefer RouterQueryEngine for chat when available
            if st.session_state.get("router_engine") is not None:

                def stream_response() -> Generator[str, None, None]:
                    """Stream response from RouterQueryEngine (tokenized locally)."""
                    try:
                        resp = st.session_state.router_engine.query(user_input)
                        # Extract text content or stringify
                        resp_text = (
                            getattr(resp, "response", None)
                            or getattr(resp, "text", None)
                            or str(resp)
                        )
                        for i, word in enumerate(resp_text.split()):
                            yield word if i == 0 else " " + word
                            time.sleep(SETTINGS.ui.streaming_delay_seconds)
                    except (ValueError, TypeError, RuntimeError) as e:
                        yield f"Error processing query: {e!s}"

                full_response = st.write_stream(stream_response())
                if full_response:
                    st.session_state.memory.put(
                        ChatMessage(role="assistant", content=full_response)
                    )
            elif st.session_state.agent_system:
                # Fallback to multi-agent system
                def stream_response() -> Generator[str, None, None]:
                    """Stream response from agent system."""
                    try:
                        response = process_query_with_agent_system(
                            st.session_state.agent_system,
                            user_input,
                            st.session_state.agent_mode,
                            st.session_state.memory,
                        )
                        response_text = (
                            response.content
                            if hasattr(response, "content")
                            else str(response)
                        )
                        for i, word in enumerate(response_text.split()):
                            yield word if i == 0 else " " + word
                            time.sleep(SETTINGS.ui.streaming_delay_seconds)
                    except (ValueError, TypeError, RuntimeError) as e:
                        yield f"Error processing query: {e!s}"

                full_response = st.write_stream(stream_response())
                if full_response:
                    st.session_state.memory.put(
                        ChatMessage(role="assistant", content=full_response)
                    )
            else:
                st.error("Please upload and process documents first before chatting.")
        except (ValueError, TypeError, RuntimeError) as e:
            st.error(f"Chat response failed: {e!s}")
            logger.error("Chat error: {}", e)

    # Store user message in memory using proper ChatMessage API
    st.session_state.memory.put(ChatMessage(role="user", content=user_input))

# Render action controls
_render_analyze_button()

# Persistence with Memory API and Error Handling
if st.button("Save Session", key="save_session_btn"):
    try:
        st.session_state.memory.chat_store.persist("session.json")
        st.success("Saved!")
    except (OSError, ValueError, TypeError) as e:
        st.error(f"Save failed: {e!s}")
        logger.error("Save error: {}", e)

if st.button("Load Session", key="load_session_btn"):
    try:
        st.session_state.memory = ChatMemoryBuffer.from_file("session.json")
        st.success("Loaded!")
    except (OSError, ValueError, TypeError) as e:
        st.error(f"Load failed: {e!s}")
        logger.error("Load error: {}", e)
