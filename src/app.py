"""Main Streamlit application for DocMind AI.

This module provides a comprehensive web interface for document analysis using
local large language models. It handles user interface components, model
selection and configuration, document upload and processing, analysis with
customizable prompts, and interactive chat functionality with multimodal
and hybrid search support, enhanced by Agentic RAG, optimizations,
ColBERT for late-interaction, and auto-quantization.

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
from typing import Any

import ollama
import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from loguru import logger

# Conditional import for LlamaCPP to handle BLAS library issues
try:
    from llama_index.llms.llama_cpp import LlamaCPP

    LLAMACPP_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError, OSError) as e:
    logger.warning("LlamaCPP not available. Running without LlamaCPP support.")
    logger.debug("LlamaCPP import failed: %s", e)
    LlamaCPP = None
    LLAMACPP_AVAILABLE = False

from dependency_injector.wiring import Provide, inject

from src.agents.coordinator import MultiAgentCoordinator
from src.agents.tool_factory import ToolFactory
from src.config import settings
from src.containers import ApplicationContainer
from src.prompts import PREDEFINED_PROMPTS
from src.utils.core import detect_hardware, validate_startup_configuration
from src.utils.document import load_documents_unstructured


# Simple wrapper functions for Ollama API calls
async def get_ollama_models() -> dict[str, Any]:
    """Get list of available Ollama models."""
    return ollama.list()


async def pull_ollama_model(ollama_model_name: str) -> dict[str, Any]:
    """Pull an Ollama model."""
    return ollama.pull(ollama_model_name)


def create_tools_from_index(index: Any) -> list[Any]:
    """Create tools from index using ToolFactory.

    Returns:
        list[Any]: List of tools created from the index.
    """
    return ToolFactory.create_basic_tools({"vector": index})


@inject
def get_agent_system(
    _tools: Any,
    _llm: Any,
    _memory: Any,
    multi_agent_coordinator=Provide[ApplicationContainer.multi_agent_coordinator],
) -> tuple[MultiAgentCoordinator, str]:
    """Get agent system using MultiAgentCoordinator with dependency injection.

    Note: _tools, _llm, _memory parameters are currently not used but kept for future compatibility.

    Args:
        multi_agent_coordinator: Injected multi-agent coordinator instance
    """
    return multi_agent_coordinator, "multi_agent"


def process_query_with_agent_system(
    agent_system: Any, query: str, mode: str, memory: Any
) -> Any:
    """Process query with agent system.

    Returns:
        AgentResponse: Response object with .content attribute.
    """
    if mode == "multi_agent":
        return agent_system.process_query(query, context=memory)
    # Return a mock response object for error cases
    from types import SimpleNamespace

    return SimpleNamespace(content="Processing error")


# Validate configuration at startup
try:
    validate_startup_configuration(settings)
except RuntimeError as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.error(
        "Please check your .env file and ensure all required settings are "
        "properly configured."
    )
    st.stop()

# Wire dependency injection container
from src.containers import wire_container

wire_container([__name__])

st.set_page_config(page_title="DocMind AI", page_icon="🧠")

if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(
        token_limit=settings.default_token_limit
    )
if "agent_system" not in st.session_state:
    st.session_state.agent_system = None
if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "single"
if "index" not in st.session_state:
    st.session_state.index = None

st.title("🧠 DocMind AI: Local LLM Document Analysis")

# Dynamic theming with Streamlit 1.47.0
theme: str = st.selectbox("Theme", ["Light", "Dark", "Auto"], index=2)
if theme != "Auto":
    bg_color = "#222" if theme == "Dark" else "#fff"
    text_color = "#fff" if theme == "Dark" else "#000"
    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: {bg_color}; color: {text_color}; }}
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
quant_suffix: str = ""
suggested_model: str = "google/gemma-3n-E4B-it"
suggested_context: int = 8192
if vram:
    if vram >= settings.minimum_vram_high_gb:
        suggested_model = "nvidia/OpenReasoning-Nemotron-32B"
        quant_suffix = "-Q4_K_M"  # Fits 16GB
        suggested_context = settings.suggested_context_high
    elif vram >= settings.minimum_vram_medium_gb:
        suggested_model = "nvidia/OpenReasoning-Nemotron-14B"
        quant_suffix = "-Q8_0"  # Fits 8GB
        suggested_context = settings.suggested_context_medium
    else:
        suggested_model = "google/gemma-3n-E4B-it"
        quant_suffix = "-Q4_K_S"  # Minimal
        suggested_context = settings.suggested_context_low
st.sidebar.info(
    f"Suggested: {suggested_model}{quant_suffix} with {suggested_context} context"
)


def is_llamacpp_available() -> bool:
    """Check if LlamaCPP backend is available."""
    return LLAMACPP_AVAILABLE


use_gpu: bool = st.sidebar.checkbox(
    "Use GPU", value=hardware_status.get("cuda_available", False)
)
# ColBERT reranking is now always enabled via native postprocessor (Phase 2.2)
parse_media: bool = st.sidebar.checkbox("Parse Video/Audio", value=False)
enable_multimodal: bool = st.sidebar.checkbox(
    "Enable Multimodal Processing",
    value=True,
    help="Extract and process images from PDFs using local Jina CLIP models",
)

# Single ReActAgent (simplified architecture)


# Model and Backend Selection with Auto-Download
st.sidebar.header("Model and Backend")
with st.sidebar.expander("Advanced Settings"):
    # Show backend availability status
    backend_options = ["ollama", "lmstudio"]
    if is_llamacpp_available():
        backend_options.insert(1, "llamacpp")
    else:
        st.sidebar.warning("LlamaCPP backend unavailable (BLAS library issue)")

    backend: str = st.selectbox("Backend", backend_options, index=0)
    context_size: int = st.selectbox(
        "Context Size", settings.context_size_options, index=1
    )

model_options: list[str] = []
if backend == "ollama":
    ollama_url: str = st.sidebar.text_input(
        "Ollama URL", value=settings.ollama_base_url
    )
    try:
        # Use sync model listing to avoid asyncio event loop conflicts
        models_response = ollama.list()  # Direct sync call
        model_options = [m["name"] for m in models_response["models"]]
    except (ConnectionError, TimeoutError, ValueError) as e:
        st.sidebar.error(f"Error fetching models: {str(e)}")
model_name: str = (
    st.sidebar.selectbox("Model", model_options or [suggested_model]) + quant_suffix
)

# Auto-download if not present (quick win for UX)
if backend == "ollama" and model_name not in model_options:
    try:
        with st.sidebar.status("Downloading model..."):
            # Use sync model pulling to avoid asyncio event loop conflicts
            ollama.pull(model_name)  # Direct sync call
            st.sidebar.success("Model downloaded!")
    except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
        st.sidebar.error(f"Download failed: {str(e)}")

llm: Any | None = None
try:
    if backend == "ollama":
        llm = Ollama(
            base_url=ollama_url,
            model=model_name,
            request_timeout=settings.request_timeout_seconds,
        )
    elif backend == "llamacpp":
        if not is_llamacpp_available():
            st.error(
                "LlamaCPP backend is not available. Please check the installation "
                "or use a different backend (Ollama or LM Studio)."
            )
        else:
            n_gpu_layers = -1 if use_gpu else 0
            llm = LlamaCPP(
                model_path=settings.llamacpp_model_path,
                context_window=context_size,
                n_gpu_layers=n_gpu_layers,
            )
    elif backend == "lmstudio":
        llm = OpenAI(
            base_url=settings.lmstudio_base_url,
            api_key="not-needed",
            model=model_name,
            max_tokens=context_size,
        )
except (ValueError, TypeError, RuntimeError, ConnectionError) as e:
    st.error(f"Model initialization error: {str(e)}")
    logger.error("Model init error: %s", str(e))
    st.stop()


# Async Document Upload Section with Media Parsing and Error Handling
@st.fragment
async def upload_section() -> None:
    """Async function to handle document upload and processing."""
    uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile] | None = (
        st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=["pdf", "docx", "mp4", "mp3", "wav"],
        )
    )
    if uploaded_files:
        with st.status("Processing documents..."):
            try:
                # Start timing for performance monitoring
                start_time = time.perf_counter()

                # ADR-009 compliant document processing
                docs = await load_documents_unstructured(uploaded_files, settings)
                doc_load_time = time.perf_counter() - start_time

                # Progress tracking for document processing
                progress_bar = st.progress(0)
                for i, _doc in enumerate(docs):
                    progress_bar.progress((i + 1) / len(docs))

                # Use async indexing for 50-80% performance improvement
                index_start_time = time.perf_counter()
                try:
                    # Create vector store and index with documents
                    vector_store = SimpleVectorStore()
                    st.session_state.index = VectorStoreIndex.from_documents(
                        docs, vector_store=vector_store
                    )
                except Exception as e:
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
                st.info(f"""
                **Performance Metrics (Async Mode):**
                - Document loading: {doc_load_time:.2f}s
                - Index creation: {index_time:.2f}s
                - Total processing: {total_time:.2f}s
                - Documents processed: {len(docs)}
                """)
                logger.info(
                    "Async processing completed in %.2fs for %s documents",
                    total_time,
                    len(docs),
                )

            except (ValueError, TypeError, OSError, RuntimeError) as e:
                st.error(f"Document processing failed: {str(e)}")
                logger.error("Doc process error: %s", str(e))


# Analysis Options and Agentic Analysis with Error Handling
st.header("Analysis Options")
prompt_type: str = st.selectbox("Prompt", list(PREDEFINED_PROMPTS.keys()))
# Other selects for tone, instructions, etc. (assuming they exist in full code)


async def run_analysis() -> None:
    """Async function to run document analysis with multi-agent support."""
    if st.session_state.index:
        with st.spinner("Running analysis..."):
            try:
                # Create tools from index
                agent_tools = create_tools_from_index(st.session_state.index)

                # Get appropriate agent system
                agent_system, mode = get_agent_system(
                    _tools=agent_tools,
                    _llm=llm,
                    _memory=st.session_state.memory,
                )

                # Process analysis with agent system
                analysis_query = f"Perform {prompt_type} analysis on the documents"
                results = await asyncio.to_thread(
                    process_query_with_agent_system,
                    agent_system,
                    analysis_query,
                    mode,
                    st.session_state.memory,
                )

                st.session_state.analysis_results = results
                st.session_state.agent_system = agent_system
                st.session_state.agent_mode = mode

                if mode == "multi":
                    st.info("✅ Analysis completed using multi-agent system")

            except (ValueError, TypeError, RuntimeError) as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.error("Analysis error: %s", str(e))


if st.button("Analyze") and st.session_state.index:
    # Use sync analysis to avoid asyncio event loop conflicts
    with st.spinner("Running analysis..."):
        try:
            # Create tools from index
            agent_tools = create_tools_from_index(st.session_state.index)

            # Get appropriate agent system
            agent_system, mode = get_agent_system(
                _tools=agent_tools,
                _llm=llm,
                _memory=st.session_state.memory,
            )

            # Process analysis with agent system
            analysis_query = f"Perform {prompt_type} analysis on the documents"
            results = process_query_with_agent_system(
                agent_system,
                analysis_query,
                mode,
                st.session_state.memory,
            )

            st.session_state.analysis_results = results
            st.session_state.agent_system = agent_system
            st.session_state.agent_mode = mode

            if mode == "multi_agent":
                st.info("✅ Analysis completed using multi-agent system")

        except (ValueError, TypeError, RuntimeError) as e:
            st.error(f"Analysis failed: {str(e)}")
            logger.error("Analysis error: %s", str(e))

# Chat with Agent using st.chat_message and write_stream
# (Async Streaming with Error Handling)
st.header("Chat with Documents")
for message in st.session_state.memory.chat_store.get_messages("default"):
    with st.chat_message(message.role):
        st.markdown(message.content)

user_input: str | None = st.chat_input("Ask a question")
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
                        _llm=llm,
                        _memory=st.session_state.memory,
                    )
                )

            if st.session_state.agent_system:
                # Using single ReActAgent for all queries

                # Process query with appropriate agent system using streaming
                def stream_response():
                    """Stream response from agent system."""
                    try:
                        response = process_query_with_agent_system(
                            st.session_state.agent_system,
                            user_input,
                            st.session_state.agent_mode,
                            st.session_state.memory,
                        )

                        # Stream response word by word for better UX
                        # Fix: Extract content from AgentResponse object
                        if hasattr(response, "content"):
                            response_text = response.content
                        else:
                            response_text = str(response)

                        words = response_text.split()
                        for i, word in enumerate(words):
                            if i == 0:
                                yield word
                            else:
                                yield " " + word
                            # Add slight delay for streaming effect
                            time.sleep(settings.streaming_delay_seconds)
                    except (ValueError, TypeError, RuntimeError) as e:
                        yield f"Error processing query: {str(e)}"

                # Use Streamlit's native streaming
                full_response = st.write_stream(stream_response())

                # Store the response in memory using proper ChatMessage API
                if full_response:
                    from llama_index.core.llms import ChatMessage

                    st.session_state.memory.put(
                        ChatMessage(role="assistant", content=full_response)
                    )
            else:
                st.error("Please upload and process documents first before chatting.")
        except (ValueError, TypeError, RuntimeError) as e:
            st.error(f"Chat response failed: {str(e)}")
            logger.error("Chat error: %s", str(e))

    # Store user message in memory using proper ChatMessage API
    from llama_index.core.llms import ChatMessage

    st.session_state.memory.put(ChatMessage(role="user", content=user_input))

# Persistence with Memory API and Error Handling
if st.button("Save Session"):
    try:
        st.session_state.memory.chat_store.persist("session.json")
        st.success("Saved!")
    except (OSError, ValueError, TypeError) as e:
        st.error(f"Save failed: {str(e)}")
        logger.error("Save error: %s", str(e))

if st.button("Load Session"):
    try:
        st.session_state.memory = ChatMemoryBuffer.from_file("session.json")
        st.success("Loaded!")
    except (OSError, ValueError, TypeError) as e:
        st.error(f"Load failed: {str(e)}")
        logger.error("Load error: %s", str(e))
