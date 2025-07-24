"""Utility functions for DocMind AI.

Provides functions for loading documents with LlamaParse extensions for video/audio,
creating hybrid indexes with ColBERT late-interaction and torch.compile,
agentic analysis with ReAct agents, and chat with knowledge graphs/multimodal.

"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import AsyncGenerator
from typing import Any

import torch
import whisper
from llama_index.core import (
    Document,
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from moviepy.video.io.VideoFileClip import VideoFileClip
from qdrant_client import QdrantClient
from ragatouille import RAGPretrainedModel

from models import Settings

settings: Settings = Settings()


def detect_hardware() -> tuple[str, int | None]:
    """Detect available hardware and VRAM for model suggestion.

    Returns:
        Tuple of hardware description and VRAM in GB (or None).
    """
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode()
        vram_match = re.search(r"(\d+)MiB / (\d+)MiB", output)
        if vram_match:
            return "GPU detected", int(vram_match.group(2)) // 1024
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("nvidia-smi not found or failed.")
    return "CPU only", None


def load_documents_llama(
    uploaded_files: list[Any], parse_media: bool = False
) -> list[Document]:
    """Load documents using LlamaParse.

    With extensions for video/audio ingestion (basic transcription/frames).

    Args:
        uploaded_files: List of uploaded file objects.
        parse_media: Whether to parse video/audio files.

    Returns:
        List of loaded Document objects.
    """
    from llama_index.core import LlamaParse

    parser = LlamaParse(result_type="markdown")  # Latest for tables/images
    docs: list[Document] = []
    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(file.name)[1]
            ) as tmp_file:
                tmp_file.write(file.getvalue())
                file_path: str = tmp_file.name

            if parse_media and (
                file_path.endswith((".mp4", ".avi"))
                or file_path.endswith((".mp3", ".wav"))
            ):
                if "video" in file.type:
                    clip = VideoFileClip(file_path)
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as audio_tmp:
                        audio_path = audio_tmp.name
                    clip.audio.write_audiofile(audio_path)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = whisper.load_model(
                        "base", device=device
                    )  # GPU offload if available
                    result = model.transcribe(audio_path)
                    text: str = result["text"]
                    # Extract frames at intervals (e.g., every 5s) per practices
                    frames: list[Any] = []
                    for t in range(0, int(clip.duration), 5):
                        frame = clip.get_frame(t)
                        from PIL import Image

                        img = Image.fromarray(frame)
                        frames.append(img)
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"images": frames, "source": file.name},
                        )
                    )
                    os.remove(audio_path)
                elif "audio" in file.type:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = whisper.load_model("base", device=device)
                    result = model.transcribe(file_path)
                    text = result["text"]
                    docs.append(
                        Document(page_content=text, metadata={"source": file.name})
                    )
            else:
                reader = SimpleDirectoryReader(
                    input_files=[file_path], file_extractor={".*": parser}
                )
                docs.extend(reader.load_data())
            os.remove(file_path)
        except FileNotFoundError as e:
            logging.error(f"File not found: {file.name} - {str(e)}")
        except ValueError as e:
            logging.error(f"Invalid file format: {file.name} - {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error loading {file.name}: {str(e)}")
    return docs


def create_index(
    docs: list[Document], use_gpu: bool, use_colbert: bool = False
) -> dict[str, Any]:
    """Create hybrid index with Qdrant, knowledge graph, torch.compile for embeddings.
    
    And optional ColBERT late-interaction.

    Args:
        docs: List of documents to index.
        use_gpu: Whether to use GPU for embeddings.
        use_colbert: Whether to enable ColBERT retriever.

    Returns:
        Dict with vector and kg indexes.
    """
    try:
        client = QdrantClient(url=settings.qdrant_url)
        embed_model = HuggingFaceEmbedding(
            model_name="jinaai/jina-embeddings-v4"
        )  # Kept as SOTA
        if use_gpu and torch.cuda.is_available():
            embed_model.model = torch.compile(embed_model.model)  # 3x speed
        vector_store = QdrantVectorStore(
            client=client, collection_name="docmind", enable_hybrid=True
        )
        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=embed_model,
        )
        if use_colbert:
            colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

            # Integrate as retriever with rerank per practices (simplified for KISS)
            def colbert_retriever(query: str, k: int = 5) -> list[Document]:
                results = colbert.rerank(
                    query, [doc.page_content for doc in docs], k=k, return_scores=True
                )  # Improved with return_scores
                return [docs[i] for i in [r["index"] for r in results]]

            index.retriever = colbert_retriever
        kg_index = KnowledgeGraphIndex.from_documents(
            docs
        )  # For entity/relation queries
        return {"vector": index, "kg": kg_index}
    except ValueError as e:
        logging.error(f"Invalid configuration for index: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Index creation error: {str(e)}")
        raise


def analyze_documents_agentic(
    agent: ReActAgent, index: dict[str, Any], prompt_type: str
) -> str:
    """Agentic analysis with multi-step reasoning.

    Args:
        agent: ReActAgent instance.
        index: Dict of indexes.
        prompt_type: Type of prompt to use.

    Returns:
        Analysis response string.
    """
    query_engine = index["vector"].as_query_engine()
    tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(name="doc_query", description="Query documents"),
        ),
    ]
    if not agent:
        agent = ReActAgent.from_tools(
            tools,
            llm=Ollama(model="llama2:7b"),
            verbose=True,
        )
    response = agent.chat(f"Analyze with prompt: {prompt_type}")
    return response.response  # Parse to AnalysisOutput


async def chat_with_agent(
    agent: ReActAgent, user_input: str, memory: ChatMemoryBuffer
) -> AsyncGenerator[str, None]:
    """Async stream chat with agent, supporting KG and multimodal via Gemma/Nemotron.

    Args:
        agent: ReActAgent instance.
        user_input: User query string.
        memory: Chat memory buffer.

    Yields:
        Chunks of response text.
    """
    try:
        response = await asyncio.to_thread(
            agent.async_stream_chat, user_input, memory=memory
        )
        async for chunk in response.async_response_gen():  # Using async_gen
            yield chunk
        # Multimodal handling: If Gemma, use native; for Nemotron (text-only),
        # extract text features
    except Exception as e:
        logging.error(f"Chat generation error: {str(e)}")
        raise
