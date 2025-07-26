"""Document loading utilities for DocMind AI.

This module handles document loading with support for various formats,
multimodal processing (text/images), and media parsing (video/audio).

Functions:
    extract_images_from_pdf: Extract images from PDF files.
    create_native_multimodal_embeddings: Create multimodal embeddings using FastEmbed.
    load_documents_llama: Load documents with multimodal and media support.
"""

import base64
import io
import logging
import os
import tempfile
from typing import Any

import torch
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_parse import LlamaParse
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from whisper import load_model as whisper_load

from models import AppSettings
from utils.model_manager import ModelManager

settings = AppSettings()


def extract_images_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """Extract images from PDF for multimodal processing.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of image dictionaries with content and metadata.
    """
    images = []
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("ppm")
                    img_pil = Image.open(io.BytesIO(img_data))

                    # Convert to base64 for storage
                    buffer = io.BytesIO()
                    img_pil.save(buffer, format="PNG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()

                    images.append(
                        {
                            "image_data": img_base64,
                            "page_number": page_num + 1,
                            "image_index": img_index,
                            "format": "PNG",
                            "size": img_pil.size,
                        }
                    )

                pix = None

        doc.close()
        logging.info(f"Extracted {len(images)} images from PDF")

    except Exception as e:
        logging.error(f"PDF image extraction failed: {e}")

    return images


def create_native_multimodal_embeddings(
    text: str,
    images: list[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Use FastEmbed native LateInteractionMultimodalEmbedding.

    Creates multimodal embeddings using FastEmbed's native implementation
    for optimal performance and memory efficiency.

    Args:
        text: Text content to embed.
        images: List of image dictionaries with base64 data. Defaults to None.

    Returns:
        Dictionary containing native multimodal embeddings with metadata.
    """
    embeddings = {
        "text_embedding": None,
        "image_embeddings": [],
        "combined_embedding": None,
        "provider_used": None,
    }

    try:
        # Use FastEmbed native LateInteractionMultimodalEmbedding
        try:
            # Use singleton pattern to prevent redundant model initializations
            model = ModelManager.get_multimodal_embedding_model()

            # Process text embedding with native FastEmbed
            text_embedding = list(model.embed_text([text]))[0]
            embeddings["text_embedding"] = text_embedding.flatten().tolist()

            # Process image embeddings with native FastEmbed
            if images:
                # Save base64 images to temporary files for FastEmbed
                image_paths = []
                for i, img_data in enumerate(images):
                    try:
                        img_bytes = base64.b64decode(img_data["image_data"])
                        temp_path = f"{tempfile.gettempdir()}/multimodal_img_{i}.png"
                        with open(temp_path, "wb") as f:
                            f.write(img_bytes)
                        image_paths.append(temp_path)
                    except Exception as e:
                        logging.warning(f"Failed to save image {i}: {e}")

                if image_paths:
                    # Native FastEmbed image embeddings
                    image_embeddings = list(model.embed_image(image_paths))

                    for i, img_emb in enumerate(image_embeddings):
                        embeddings["image_embeddings"].append(
                            {
                                "embedding": img_emb.flatten().tolist(),
                                "metadata": images[i] if i < len(images) else {},
                            }
                        )

                    # Clean up temporary files
                    import contextlib

                    for path in image_paths:
                        with contextlib.suppress(Exception):
                            os.unlink(path)

            # FastEmbed handles combined embeddings natively
            embeddings["combined_embedding"] = embeddings["text_embedding"]
            embeddings["provider_used"] = "fastembed_native_multimodal"

            logging.info("Using FastEmbed native LateInteractionMultimodalEmbedding")

        except ImportError:
            logging.warning(
                "FastEmbed LateInteractionMultimodalEmbedding not available, "
                "using text-only"
            )
            # Fallback to FastEmbed text-only
            fastembed_model = FastEmbedEmbedding(
                model_name=settings.dense_embedding_model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                cache_dir="./embeddings_cache",
            )
            embeddings["text_embedding"] = fastembed_model.get_text_embedding(text)
            embeddings["combined_embedding"] = embeddings["text_embedding"]
            embeddings["provider_used"] = "fastembed_text_only"

    except Exception as e:
        logging.error(f"Native multimodal embedding creation failed: {e}")
        # Ultimate fallback to FastEmbed text-only
        try:
            fastembed_model = FastEmbedEmbedding(
                model_name=settings.dense_embedding_model,
                cache_dir="./embeddings_cache",
            )
            embeddings["text_embedding"] = fastembed_model.get_text_embedding(text)
            embeddings["combined_embedding"] = embeddings["text_embedding"]
            embeddings["provider_used"] = "fastembed_fallback"
        except Exception as fallback_e:
            logging.error(f"All embedding methods failed: {fallback_e}")
            embeddings["provider_used"] = "failed"

    return embeddings


def load_documents_llama(
    uploaded_files: list[Any], parse_media: bool = False, enable_multimodal: bool = True
) -> list[Document]:
    """Load documents using LlamaParse with multimodal support.

    Supports standard document formats plus video/audio ingestion and PDF image
    extraction for multimodal processing with enhanced error handling.

    Args:
        uploaded_files: List of uploaded file objects.
        parse_media: Whether to parse video/audio files. Defaults to False.
        enable_multimodal: Whether to enable multimodal processing for PDFs.
            Defaults to True.

    Returns:
        List of loaded Document objects with multimodal embeddings where applicable.

    Raises:
        Exception: If critical document loading operations fail.
    """
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
                    model = whisper_load(
                        "base", device=device
                    )  # GPU offload if available
                    result = model.transcribe(audio_path)
                    text: str = result["text"]
                    # Extract frames at intervals (e.g., every 5s) per practices
                    frames: list[Any] = []
                    for t in range(0, int(clip.duration), 5):
                        frame = clip.get_frame(t)

                        img = Image.fromarray(frame)
                        frames.append(img)
                    docs.append(
                        Document(
                            text=text,
                            metadata={
                                "images": frames,
                                "source": file.name,
                                "type": "video",
                            },
                        )
                    )
                    os.remove(audio_path)
                elif "audio" in file.type:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = whisper_load("base", device=device)
                    result = model.transcribe(file_path)
                    text = result["text"]
                    docs.append(
                        Document(
                            text=text, metadata={"source": file.name, "type": "audio"}
                        )
                    )
            else:
                # Standard document processing with multimodal support
                reader = SimpleDirectoryReader(
                    input_files=[file_path], file_extractor={".*": parser}
                )
                loaded_docs = reader.load_data()

                # Enhanced multimodal processing for PDFs
                if enable_multimodal and file_path.lower().endswith(".pdf"):
                    try:
                        # Extract images from PDF
                        pdf_images = extract_images_from_pdf(file_path)

                        # Process each document with multimodal embeddings
                        for doc in loaded_docs:
                            if pdf_images:
                                # Create local multimodal embeddings (offline)
                                multimodal_embeddings = (
                                    create_native_multimodal_embeddings(
                                        text=doc.text,
                                        images=pdf_images,
                                    )
                                )

                                # Update document with multimodal data
                                doc.metadata.update(
                                    {
                                        "source": file.name,
                                        "type": "pdf_multimodal",
                                        "image_count": len(pdf_images),
                                        "has_images": True,
                                        "multimodal_embeddings": multimodal_embeddings,
                                    }
                                )

                                logging.info(
                                    f"Created multimodal embeddings for {file.name} "
                                    f"with {len(pdf_images)} images"
                                )
                            else:
                                # PDF without images - standard text processing
                                doc.metadata.update(
                                    {
                                        "source": file.name,
                                        "type": "pdf_text_only",
                                        "has_images": False,
                                    }
                                )

                    except Exception as e:
                        logging.warning(
                            f"Multimodal processing failed for {file.name}: {e}"
                        )
                        # Fallback to standard processing
                        for doc in loaded_docs:
                            doc.metadata.update(
                                {
                                    "source": file.name,
                                    "type": "pdf_fallback",
                                    "has_images": False,
                                }
                            )
                else:
                    # Standard processing for non-PDF files
                    for doc in loaded_docs:
                        doc.metadata.update(
                            {
                                "source": file.name,
                                "type": "standard_document",
                                "has_images": False,
                            }
                        )

                docs.extend(loaded_docs)

            os.remove(file_path)

        except FileNotFoundError as e:
            logging.error(f"File not found: {file.name} - {str(e)}")
        except ValueError as e:
            logging.error(f"Invalid file format: {file.name} - {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error loading {file.name}: {str(e)}")

    logging.info(
        f"Loaded {len(docs)} documents, multimodal processing enabled: "
        f"{enable_multimodal}"
    )
    return docs
