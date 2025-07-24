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
from collections.abc import AsyncGenerator
from typing import Any

import ollama
import phoenix as px
import streamlit as st
import torch
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.llms.ollama import Ollama
from langchain_openai import OpenAI
from llama_index.core import set_global_handler
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

from models import Settings
from prompts import PREDEFINED_PROMPTS
from utils import (
    analyze_documents_agentic,
    chat_with_agent,
    create_index,
    detect_hardware,
    load_documents_llama,
    setup_logging,
)

setup_logging()

settings: Settings = Settings()

st.set_page_config(page_title="DocMind AI", page_icon="🧠")

if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=32768)
if "agent" not in st.session_state:
    st.session_state.agent = None
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
hardware_info, vram = detect_hardware()
st.sidebar.info(f"Detected: {hardware_info}, VRAM: {vram}GB" if vram else hardware_info)
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

use_gpu: bool = st.sidebar.checkbox("Use GPU", value=torch.cuda.is_available())
use_colbert: bool = st.sidebar.checkbox("Use ColBERT Late-Interaction", value=False)
parse_media: bool = st.sidebar.checkbox("Parse Video/Audio", value=False)
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
        llm = Ollama(base_url=ollama_url, model=model_name, num_ctx=context_size)
    elif backend == "llamacpp":
        n_gpu_layers = -1 if use_gpu else 0
        llm = LlamaCpp(
            model_path=settings.llamacpp_model_path,
            n_ctx=context_size,
            n_gpu_layers=n_gpu_layers,
        )
    elif backend == "lmstudio":
        llm = OpenAI(
            base_url=settings.lmstudio_base_url, api_key="not-needed", model=model_name
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
                docs: list[Any] = await asyncio.to_thread(
                    load_documents_llama, uploaded_files, parse_media
                )
                st.session_state.index = await asyncio.to_thread(
                    create_index, docs, use_gpu, use_colbert
                )
                st.success("Documents indexed!")
            except Exception as e:
                st.error(f"Document processing failed: {str(e)}")
                logging.error(f"Doc process error: {str(e)}")


# Analysis Options and Agentic Analysis with Error Handling
st.header("Analysis Options")
prompt_type: str = st.selectbox("Prompt", list(PREDEFINED_PROMPTS.keys()))
# Other selects for tone, instructions, etc. (assuming they exist in full code)


async def run_analysis() -> None:
    """Async function to run document analysis."""
    if st.session_state.index:
        with st.spinner("Agentic Analysis..."):
            try:
                results: Any = await asyncio.to_thread(
                    analyze_documents_agentic,
                    st.session_state.agent or ReActAgent.from_tools([]),  # Init if None
                    st.session_state.index,
                    prompt_type,  # Params
                )
                st.session_state.analysis_results = results
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

            async def stream_response() -> AsyncGenerator[str, None]:
                """Stream response from chat agent."""
                async for chunk in chat_with_agent(
                    st.session_state.agent, user_input, st.session_state.memory
                ):
                    yield chunk

            st.write_stream(asyncio.run(stream_response()))
        except Exception as e:
            st.error(f"Chat response failed: {str(e)}")
            logging.error(f"Chat error: {str(e)}")
    st.session_state.memory.put({"role": "user", "content": user_input})
    # Put assistant response after streaming

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
