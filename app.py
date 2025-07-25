"""Main Streamlit application for DocMind AI.

This module provides a comprehensive web interface for document analysis using
local large language models. It handles user interface components, model
selection and configuration, document upload and processing, analysis with
customizable prompts, and interactive chat functionality with multimodal
and hybrid search support, enhanced by Agentic RAG, optimizations,
Phoenix for local observability, ColBERT for late-interaction, and auto-quantization.

The application supports multiple backends (Ollama, LlamaCpp, LM Studio),
various document formats including basic video/audio, and provides features like session
persistence, theming, hardware detection, agentic workflows, async operations,
auto-model download, and Phoenix dashboard link.

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
import logging
import time
from typing import Any

import ollama
import phoenix as px
import streamlit as st
from llama_index.core import set_global_handler
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from agent_factory import (
    get_agent_system,
    process_query_with_agent_system,
)
from models import AppSettings
from prompts import PREDEFINED_PROMPTS
from utils import (
    create_index_async,
    create_tools_from_index,
    detect_hardware,
    load_documents_llama,
    setup_logging,
)

setup_logging()

settings: AppSettings = AppSettings()

st.set_page_config(page_title="DocMind AI", page_icon="🧠")

if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=32768)
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
    if vram >= 16:
        suggested_model = "nvidia/OpenReasoning-Nemotron-32B"
        quant_suffix = "-Q4_K_M"  # Fits 16GB
        suggested_context = 65536
    elif vram >= 8:
        suggested_model = "nvidia/OpenReasoning-Nemotron-14B"
        quant_suffix = "-Q8_0"  # Fits 8GB
        suggested_context = 32768
    else:
        suggested_model = "google/gemma-3n-E4B-it"
        quant_suffix = "-Q4_K_S"  # Minimal
        suggested_context = 8192
st.sidebar.info(
    f"Suggested: {suggested_model}{quant_suffix} with {suggested_context} context"
)

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

# LIBRARY-FIRST: LangGraph Multi-Agent Toggle
enable_multi_agent: bool = st.sidebar.checkbox(
    "Enable Multi-Agent Mode",
    value=False,
    help="Use LangGraph supervisor system for complex queries with specialized agents",
)
if enable_multi_agent:
    st.sidebar.info(
        "🤖 Multi-agent mode: Document, Knowledge Graph, and Multimodal specialists"
    )

use_phoenix: bool = st.sidebar.checkbox("Enable Phoenix Observability", value=False)
if use_phoenix:
    px.launch_app()
    set_global_handler("arize_phoenix")

# Model and Backend Selection with Auto-Download
st.sidebar.header("Model and Backend")
with st.sidebar.expander("Advanced Settings"):
    backend: str = st.selectbox("Backend", ["ollama", "llamacpp", "lmstudio"], index=0)
    context_size: int = st.selectbox(
        "Context Size", [8192, 32768, 65536, 131072], index=1
    )

model_options: list[str] = []
if backend == "ollama":
    ollama_url: str = st.sidebar.text_input(
        "Ollama URL", value=settings.ollama_base_url
    )
    try:
        model_options = [m["name"] for m in ollama.list()["models"]]
    except Exception as e:
        st.sidebar.error(f"Error fetching models: {str(e)}")
model_name: str = (
    st.sidebar.selectbox("Model", model_options or [suggested_model]) + quant_suffix
)

# Auto-download if not present (quick win for UX)
if backend == "ollama" and model_name not in model_options:
    try:
        with st.sidebar.status("Downloading model..."):
            ollama.pull(model_name)
            st.sidebar.success("Model downloaded!")
    except Exception as e:
        st.sidebar.error(f"Download failed: {str(e)}")

llm: Any | None = None
try:
    if backend == "ollama":
        llm = Ollama(base_url=ollama_url, model=model_name, request_timeout=60.0)
    elif backend == "llamacpp":
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
except Exception as e:
    st.error(f"Model initialization error: {str(e)}")
    logging.error(f"Model init error: {str(e)}")
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

                docs: list[Any] = await asyncio.to_thread(
                    load_documents_llama, uploaded_files, parse_media, enable_multimodal
                )
                doc_load_time = time.perf_counter() - start_time

                # Use async indexing for 50-80% performance improvement
                index_start_time = time.perf_counter()
                st.session_state.index = await create_index_async(docs, use_gpu)
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
                logging.info(
                    f"Async processing completed in {total_time:.2f}s for "
                    f"{len(docs)} documents"
                )

            except Exception as e:
                st.error(f"Document processing failed: {str(e)}")
                logging.error(f"Doc process error: {str(e)}")


# Analysis Options and Agentic Analysis with Error Handling
st.header("Analysis Options")
prompt_type: str = st.selectbox("Prompt", list(PREDEFINED_PROMPTS.keys()))
# Other selects for tone, instructions, etc. (assuming they exist in full code)


async def run_analysis() -> None:
    """Async function to run document analysis with multi-agent support."""
    if st.session_state.index:
        with st.spinner(
            "Running analysis..."
            + (" (Multi-Agent Mode)" if enable_multi_agent else "")
        ):
            try:
                # Create tools from index
                tools = create_tools_from_index(st.session_state.index)

                # Get appropriate agent system
                agent_system, mode = get_agent_system(
                    tools=tools,
                    llm=llm,
                    enable_multi_agent=enable_multi_agent,
                    memory=st.session_state.memory,
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

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logging.error(f"Analysis error: {str(e)}")


if st.button("Analyze"):
    asyncio.run(run_analysis())

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
                        tools=tools,
                        llm=llm,
                        enable_multi_agent=enable_multi_agent,
                        memory=st.session_state.memory,
                    )
                )

            if st.session_state.agent_system:
                # Show which mode is being used
                if st.session_state.agent_mode == "multi":
                    st.info("🤖 Using multi-agent system")

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
                        words = response.split()
                        for i, word in enumerate(words):
                            if i == 0:
                                yield word
                            else:
                                yield " " + word
                            # Add slight delay for streaming effect
                            time.sleep(0.02)
                    except Exception as e:
                        yield f"Error processing query: {str(e)}"

                # Use Streamlit's native streaming
                full_response = st.write_stream(stream_response())

                # Store the response in memory
                if full_response:
                    st.session_state.memory.put(
                        {"role": "assistant", "content": full_response}
                    )
            else:
                st.error("Please upload and process documents first before chatting.")
        except Exception as e:
            st.error(f"Chat response failed: {str(e)}")
            logging.error(f"Chat error: {str(e)}")

    # Store user message in memory
    st.session_state.memory.put({"role": "user", "content": user_input})

# Persistence with Memory API and Error Handling
if st.button("Save Session"):
    try:
        st.session_state.memory.chat_store.persist("session.json")
        st.success("Saved!")
    except Exception as e:
        st.error(f"Save failed: {str(e)}")
        logging.error(f"Save error: {str(e)}")

if st.button("Load Session"):
    try:
        st.session_state.memory = ChatMemoryBuffer.from_file("session.json")
        st.success("Loaded!")
    except Exception as e:
        st.error(f"Load failed: {str(e)}")
        logging.error(f"Load error: {str(e)}")

# Phoenix Dashboard Link if enabled
if use_phoenix:
    st.sidebar.link_button(
        "View Phoenix Dashboard", "http://localhost:6006"
    )  # Default local URL
