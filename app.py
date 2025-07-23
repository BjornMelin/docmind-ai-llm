"""Main Streamlit application for DocMind AI.

This module provides a comprehensive web interface for document analysis using
local large language models. It handles user interface components, model
selection and configuration, document upload and processing, analysis with
customizable prompts, and interactive chat functionality with multimodal
and hybrid search support.

The application supports multiple backends (Ollama, LlamaCpp, LM Studio),
various document formats (PDF, DOCX, TXT, etc.), and provides features like
session persistence, theming, and hardware detection for optimal model
suggestions.

Example:
    Run the application using Streamlit::

        $ streamlit run app.py

Attributes:
    settings: Global settings instance loaded from environment variables.
    st.session_state.analysis_results: Cached analysis results from documents.
    st.session_state.chat_history: Chat conversation history with the AI.
    st.session_state.vectorstore: Qdrant vector database for document search.
    st.session_state.documents_text: Combined text content from uploaded documents.

"""

import concurrent.futures
import json
import logging
import os
import pickle

import ollama
import streamlit as st
from langchain_community.llms import LlamaCpp, Ollama
from langchain_openai import OpenAI

from models import AnalysisOutput, Settings
from prompts import INSTRUCTIONS, LENGTHS, PREDEFINED_PROMPTS, TONES
from utils import (
    analyze_documents,
    chat_with_context,
    create_vectorstore,
    detect_hardware,
    estimate_tokens,
    load_documents,
    setup_logging,
)

setup_logging()

settings = Settings()

st.set_page_config(page_title="DocMind AI", page_icon="🧠")

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_text" not in st.session_state:
    st.session_state.documents_text = ""

st.title("🧠 DocMind AI: Local LLM Document Analysis")

# Dynamic theming
theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], index=2)
if theme == "Dark":
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] { background-color: #333; color: #fff; }
            .stApp { background-color: #222; color: #fff; }
            .stButton > button { 
                background-color: #444; color: #fff; border: 1px solid #555; 
            }
            .stTextInput > div > input { background-color: #333; color: #fff; }
            .stSelectbox > div > select { background-color: #333; color: #fff; }
            h1, h2, h3, h4, h5, h6 { color: #fff; }
        </style>
    """,
        unsafe_allow_html=True,
    )
elif theme == "Light":
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] { background-color: #fff; color: #000; }
            .stApp { background-color: #fff; color: #000; }
            .stButton > button { 
                background-color: #eee; color: #000; border: 1px solid #ddd; 
            }
            .stTextInput > div > input { background-color: #fff; color: #000; }
            .stSelectbox > div > select { background-color: #fff; color: #000; }
            h1, h2, h3, h4, h5, h6 { color: #000; }
        </style>
    """,
        unsafe_allow_html=True,
    )

# Hardware Detection and Model Suggestion
hardware_info = detect_hardware()
st.sidebar.info(f"Detected Hardware: {hardware_info}")
suggested_model = "Qwen/Qwen3-8B" if "CPU" in hardware_info else "Qwen/Qwen3-72B"
st.sidebar.info(f"Suggested Model: {suggested_model}")

# Model and Backend Selection
st.sidebar.header("Model and Backend Selection")
with st.sidebar.expander("Advanced Settings"):
    backend = st.selectbox("Backend", ["ollama", "llamacpp", "lmstudio"], index=0)
    context_size = st.selectbox(
        "Context Size", [2048, 4096, 8192, 32768, 131072], index=1
    )

model_options = []
if backend == "ollama":
    ollama_url = st.sidebar.text_input(
        "Ollama Base URL", value=settings.ollama_base_url
    )
    try:
        available_models = [m["name"] for m in ollama.list()["models"]]
        model_options = available_models or ["No models found"]
    except Exception as e:
        st.sidebar.error(f"Error fetching models: {str(e)}")
        model_options = ["Error fetching models"]
    pull_model = st.sidebar.text_input("Model to Pull")
    if st.sidebar.button("Pull Model") and pull_model:
        try:
            ollama.pull(pull_model)
            st.sidebar.success(f"Pulled {pull_model}")
        except Exception as e:
            st.sidebar.error(f"Pull failed: {str(e)}")
elif backend == "lmstudio":
    lmstudio_url = st.sidebar.text_input(
        "LM Studio Base URL", value=settings.lmstudio_base_url
    )
elif backend == "llamacpp":
    llamacpp_path = st.sidebar.text_input(
        "Llama.cpp Model Path (.gguf)", value=settings.llamacpp_model_path
    )

model_name = (
    st.sidebar.selectbox("Model Name", model_options, index=0)
    if model_options
    else st.sidebar.text_input("Model Name", value=suggested_model)
)

llm = None
try:
    if backend == "ollama":
        llm = Ollama(base_url=ollama_url, model=model_name, num_ctx=context_size)
    elif backend == "llamacpp":
        llm = LlamaCpp(model_path=llamacpp_path, n_ctx=context_size)
    elif backend == "lmstudio":
        llm = OpenAI(base_url=lmstudio_url, api_key="not-needed", model=model_name)
    llm.invoke("Test prompt")
except ConnectionError as e:
    st.error(f"Backend connection failed: {str(e)}. Ensure it's running.")
    st.stop()
except ValueError as e:
    st.error(f"Invalid configuration: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Model initialization error: {str(e)}")
    logging.error(f"Model init error: {str(e)}")
    st.stop()

# Document Upload
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Choose files",
    accept_multiple_files=True,
    type=[
        "pdf",
        "docx",
        "txt",
        "xlsx",
        "md",
        "json",
        "xml",
        "rtf",
        "csv",
        "msg",
        "pptx",
        "odt",
        "epub",
        "py",
        "js",
        "java",
        "ts",
        "tsx",
        "c",
        "cpp",
        "h",
    ],
)

# Preview with multimodal for PDF
if uploaded_files:
    for file in uploaded_files:
        with st.expander(f"Preview: {file.name}"):
            bytes_data = file.getvalue()
            snippet = bytes_data.decode("utf-8", errors="ignore")[:500] + "..."
            st.text(snippet)
            if file.type == "application/pdf":
                try:
                    import fitz

                    doc = fitz.open(stream=bytes_data, filetype="pdf")
                    page = doc.load_page(0)
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, caption="First Page Preview")
                except Exception as e:
                    logging.error(f"PDF preview error: {str(e)}")
                    st.warning("Unable to generate image preview.")

# Persistence
if st.button("Save Session"):
    try:
        with open("session.pkl", "wb") as f:
            pickle.dump(
                {
                    "vectorstore": st.session_state.vectorstore,
                    "chat_history": st.session_state.chat_history,
                    "analysis_results": st.session_state.analysis_results,
                },
                f,
            )
        st.success("Session saved")
    except Exception as e:
        st.error(f"Save failed: {str(e)}")
        logging.error(f"Save session error: {str(e)}")

if st.button("Load Session"):
    if os.path.exists("session.pkl"):
        try:
            with open("session.pkl", "rb") as f:
                data = pickle.load(f)  # noqa: S301
                st.session_state.vectorstore = data["vectorstore"]
                st.session_state.chat_history = data["chat_history"]
                st.session_state.analysis_results = data["analysis_results"]
            st.success("Session loaded")
        except Exception as e:
            st.error(f"Load failed: {str(e)}")
            logging.error(f"Load session error: {str(e)}")
    else:
        st.error("No saved session found")

# Analysis Options
st.header("Analysis Options")
prompt_type = st.selectbox("Choose Prompt", list(PREDEFINED_PROMPTS.keys()))
custom_prompt = (
    st.text_area("Enter Custom Prompt") if prompt_type == "Custom Prompt" else None
)

tone = st.selectbox("Select Tone", list(TONES.keys()))
instruction = st.selectbox("Select Instructions", list(INSTRUCTIONS.keys()))
custom_instruction = (
    st.text_area("Enter Custom Instructions")
    if instruction == "Custom Instructions"
    else None
)

length_detail = st.selectbox("Select Length/Detail", list(LENGTHS.keys()))

analysis_mode = st.radio(
    "Analysis Mode",
    ["Analyze each document separately", "Combine analysis for all documents"],
)

chunked_analysis = st.checkbox("Enable Chunked Analysis for Large Docs")
late_chunking = st.checkbox("Enable Late Chunking (recommended for accuracy)")
multi_vector = st.checkbox("Enable Multi-Vector Embeddings")

if st.button("Extract and Analyze") and uploaded_files:
    progress = st.progress(0)
    with st.spinner("Loading and indexing documents..."):
        try:
            docs = load_documents(uploaded_files, late_chunking=late_chunking)
            progress.progress(0.3)
            st.session_state.vectorstore = create_vectorstore(
                docs, multi_vector=multi_vector
            )
            texts = [doc.page_content for doc in docs]
            st.session_state.documents_text = " ".join(texts)
            progress.progress(0.5)

            total_tokens = estimate_tokens(st.session_state.documents_text)
            if total_tokens > context_size * 0.8:
                max_tokens = context_size * 0.8
                st.warning(
                    f"Document content may exceed context window "
                    f"({total_tokens} est. tokens > {max_tokens}). "
                    f"Using retrieval for chat."
                )
        except Exception as e:
            st.error(f"Document loading failed: {str(e)}")
            logging.error(f"Doc load error: {str(e)}")
            st.stop()

    def run_analysis():
        """Run document analysis based on selected mode and parameters.

        Processes documents either individually or combined based on the
        analysis_mode setting. Handles both chunked and regular analysis
        approaches depending on document size and user preferences.

        Returns:
            List of analysis results, either AnalysisOutput objects or
            raw string outputs in case of parsing failures.

        Raises:
            RuntimeError: If analysis encounters a critical error during
                processing that prevents completion.

        """
        try:
            if analysis_mode == "Combine analysis for all documents":
                combined_text = " ".join(texts)
                return [
                    analyze_documents(
                        llm,
                        [combined_text],
                        prompt_type,
                        custom_prompt,
                        tone,
                        instruction,
                        custom_instruction,
                        length_detail,
                        context_size,
                        chunked_analysis,
                    )
                ]
            else:
                results = []
                for i, text in enumerate(texts):
                    result = analyze_documents(
                        llm,
                        [text],
                        prompt_type,
                        custom_prompt,
                        tone,
                        instruction,
                        custom_instruction,
                        length_detail,
                        context_size,
                        chunked_analysis,
                    )
                    results.append(result)
                    progress.progress(0.5 + (0.5 * (i + 1) / len(texts)))
                return results
        except Exception as e:
            logging.error(f"Analysis error: {str(e)}")
            raise RuntimeError(f"Analysis error: {str(e)}") from e

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_analysis)
        try:
            st.session_state.analysis_results = future.result()
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            logging.error(f"Unexpected analysis error: {str(e)}")

# Display Results
if st.session_state.analysis_results:
    st.header("Analysis Results")
    for idx, result in enumerate(st.session_state.analysis_results):
        if isinstance(result, AnalysisOutput):
            st.subheader(
                f"Document {idx + 1}"
                if len(st.session_state.analysis_results) > 1
                else "Combined Analysis"
            )
            st.write("**Summary:**", result.summary)
            st.write("**Key Insights:**", ", ".join(result.key_insights))
            st.write("**Action Items:**", ", ".join(result.action_items))
            st.write("**Open Questions:**", ", ".join(result.open_questions))
        else:
            st.write("Raw Output:", result)

    if st.button("Export as JSON"):
        json_data = [
            r.model_dump() if isinstance(r, AnalysisOutput) else {"raw": str(r)}
            for r in st.session_state.analysis_results
        ]
        st.download_button(
            "Download JSON",
            json.dumps(json_data, indent=4),
            "analysis.json",
            "application/json",
        )

    if st.button("Export as Markdown"):
        md_content = ""
        for idx, result in enumerate(st.session_state.analysis_results):
            md_content += (
                f"## Document {idx + 1}\n"
                if len(st.session_state.analysis_results) > 1
                else "## Combined Analysis\n"
            )
            if isinstance(result, AnalysisOutput):
                md_content += f"**Summary:** {result.summary}\n\n"
                md_content += (
                    "**Key Insights:**\n"
                    + "\n".join(f"- {i}" for i in result.key_insights)
                    + "\n\n"
                )
                md_content += (
                    "**Action Items:**\n"
                    + "\n".join(f"- {i}" for i in result.action_items)
                    + "\n\n"
                )
                md_content += (
                    "**Open Questions:**\n"
                    + "\n".join(f"- {q}" for q in result.open_questions)
                    + "\n\n"
                )
            else:
                md_content += f"Raw Output: {result}\n\n"
        st.download_button("Download MD", md_content, "analysis.md", "text/markdown")

# Interactive Chat
st.header("Chat with Documents")
user_input = st.text_input("Ask a question about the documents:")
if st.button("Send") and user_input and st.session_state.vectorstore:
    with st.spinner("Thinking..."):
        try:
            response_stream = chat_with_context(
                llm,
                st.session_state.vectorstore,
                user_input,
                st.session_state.chat_history,
            )
            response = st.write_stream(response_stream)
            st.session_state.chat_history.append(
                {"user": user_input, "assistant": response}
            )
        except Exception as e:
            st.error(f"Chat error: {str(e)}")
            logging.error(f"Chat error: {str(e)}")

for msg in st.session_state.chat_history:
    st.write(f"**User:** {msg['user']}")
    st.write(f"**Assistant:** {msg['assistant']}")
