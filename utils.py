"""Utility functions for document loading, vectorstore, analysis, and chat.

This module provides comprehensive utilities for the DocMind AI application,
including document processing, vector database operations, text analysis,
and chat functionality. It supports multiple document formats and provides
advanced features like late chunking, multi-vector embeddings, and hybrid
search capabilities.

Key functionalities:
- Document loading from various formats (PDF, DOCX, CSV, etc.)
- Vector store creation and management with Qdrant
- Document analysis with customizable prompts and chunking
- Chat interface with context retrieval and reranking
- Hardware detection for optimal model suggestions
- Token estimation and text processing utilities

Example:
    Basic document processing workflow::

        docs = load_documents(uploaded_files, late_chunking=True)
        vectorstore = create_vectorstore(docs, multi_vector=True)
        results = analyze_documents(llm, texts, prompt_type, ...)

"""

import logging
import os
import subprocess
import tempfile
from collections.abc import Generator

import extract_msg
import fitz
import nltk
import polars as pl
import tiktoken
from docx import Document as DocxDocument
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.output_parsers import PydanticOutputParser
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from torch import mean
from transformers import AutoModel, AutoTokenizer

from models import AnalysisOutput, Settings
from prompts import INSTRUCTIONS, LENGTHS, PREDEFINED_PROMPTS, TONES

nltk.download("punkt", quiet=True)


def setup_logging():
    """Set up basic logging configuration to file.

    Initializes the logging system with file output using settings from
    the environment. Creates the log directory if it doesn't exist and
    configures the logging format with timestamps and log levels.

    The log file location is determined by the Settings.log_path configuration.
    """
    settings = Settings()
    os.makedirs(os.path.dirname(settings.log_path), exist_ok=True)
    logging.basicConfig(
        filename=settings.log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def detect_hardware():
    """Detect available hardware for optimal model suggestion.

    Attempts to detect NVIDIA GPU availability by checking for the nvidia-smi
    command. This helps suggest appropriate models based on available hardware
    capabilities.

    Returns:
        A string indicating the detected hardware type, either "GPU detected"
        for systems with NVIDIA GPUs or "CPU only" for systems without GPU
        acceleration capabilities.

    """
    try:
        subprocess.check_output(  # noqa: S603
            ["nvidia-smi"],  # noqa: S607
            stderr=subprocess.DEVNULL,
        )
        return "GPU detected"
    except (subprocess.CalledProcessError, OSError):
        return "CPU only"


def load_pdf_or_epub(file_path: str) -> list[Document]:
    """Load text and images from PDF or EPUB files using PyMuPDF.

    Extracts both textual content and images from PDF or EPUB documents.
    Images are converted to PNG format and stored in the document metadata.

    Args:
        file_path: Path to the PDF or EPUB file to process.

    Returns:
        List containing a single Document object with extracted text content
        and images stored in metadata under the 'images' key.

    """
    doc = fitz.open(file_path)
    text = ""
    images = []
    for page in doc:
        text += page.get_text()
        pix = page.get_pixmap()
        images.append(pix.tobytes("png"))
    return [
        Document(page_content=text, metadata={"source": file_path, "images": images})
    ]


def load_docx_or_odt_or_rtf_or_pptx(file_path: str) -> list[Document]:
    """Load text from DOCX, ODT, RTF, PPTX using python-docx.

    Extracts text content from Microsoft Office documents and similar formats
    by reading paragraph content from the document structure.

    Args:
        file_path: Path to the document file to process.

    Returns:
        List containing a single Document object with extracted text content.

    """
    doc = DocxDocument(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return [Document(page_content=text, metadata={"source": file_path})]


def load_xlsx_or_csv_or_xml_or_json_or_md(file_path: str) -> list[Document]:
    """Load data from structured files and markdown using Polars or text read.

    Handles multiple structured data formats by using appropriate Polars readers
    for tabular data (XLSX, CSV, XML, JSON) and direct text reading for markdown.
    Tabular data is converted to string representation for processing.

    Args:
        file_path: Path to the data file to process.

    Returns:
        List containing a single Document object with extracted content.

    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".csv", ".xml", ".json"]:
        if ext == ".xlsx":
            df = pl.read_excel(file_path)
        elif ext == ".csv":
            df = pl.read_csv(file_path)
        elif ext == ".xml":
            df = pl.read_xml(file_path)
        elif ext == ".json":
            df = pl.read_json(file_path)
        text = df.to_pandas().to_string()
    elif ext == ".md":
        with open(file_path) as f:
            text = f.read()
    return [Document(page_content=text, metadata={"source": file_path})]


def load_msg(file_path: str) -> list[Document]:
    """Load text from MSG email files using extract-msg.

    Extracts the body content from Microsoft Outlook MSG email files.

    Args:
        file_path: Path to the MSG email file to process.

    Returns:
        List containing a single Document object with email body content.

    """
    msg = extract_msg.Message(file_path)
    text = msg.body
    return [Document(page_content=text, metadata={"source": file_path})]


def late_chunking(text: str, token_embeddings) -> list:
    """Perform late chunking using NLTK for spans and mean pooling.

    Implements late chunking by segmenting text into sentences using NLTK,
    then computing mean pooled embeddings for each sentence span from the
    token-level embeddings.

    Args:
        text: Input text to chunk into sentences.
        token_embeddings: Token-level embeddings tensor from transformer model.

    Returns:
        List of mean-pooled embeddings, one for each sentence chunk.

    """
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    spans = []
    start = 0
    for sent in sentences:
        end = start + len(sent)
        spans.append((start, end))
        start = end + 1  # for space
    pooled = []
    for start, end in spans:
        chunk_emb = mean(token_embeddings[start:end], dim=0)
        pooled.append(chunk_emb)
    return pooled


def load_documents(uploaded_files, late_chunking=False) -> list[Document]:
    """Load and split documents from uploaded files with optional late chunking.

    Processes uploaded files by temporarily saving them to disk, loading content
    based on file type, optionally applying late chunking embeddings, and then
    splitting the documents into manageable chunks for processing.

    Args:
        uploaded_files: List of uploaded file objects from Streamlit.
        late_chunking: Whether to apply late chunking with sentence-level
            embeddings using NLTK tokenization and mean pooling.

    Returns:
        List of Document objects after text splitting, with optional chunk
        embeddings stored in metadata.

    Raises:
        ValueError: If an unsupported file type is encountered.

    """
    docs = []
    settings = Settings()
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.name)[1]
        ) as tmp_file:
            tmp_file.write(file.getvalue())
            file_path = tmp_file.name
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".pdf", ".epub"]:
            docs.extend(load_pdf_or_epub(file_path))
        elif ext in [".docx", ".odt", ".rtf", ".pptx"]:
            docs.extend(load_docx_or_odt_or_rtf_or_pptx(file_path))
        elif ext in [".xlsx", ".csv", ".xml", ".json", ".md"]:
            docs.extend(load_xlsx_or_csv_or_xml_or_json_or_md(file_path))
        elif ext == ".msg":
            docs.extend(load_msg(file_path))
        elif ext in [".txt", ".py", ".js", ".java", ".ts", ".tsx", ".c", ".cpp", ".h"]:
            loader = TextLoader(file_path)
            docs.extend(loader.load())
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        os.remove(file_path)
    if late_chunking:
        tokenizer = AutoTokenizer.from_pretrained(
            settings.default_embedding_model, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            settings.default_embedding_model, trust_remote_code=True
        )
        for doc in docs:
            inputs = tokenizer(doc.page_content, return_tensors="pt")
            outputs = model(**inputs)
            token_emb = outputs.last_hidden_state[0]
            chunk_emb = late_chunking(doc.page_content, token_emb)
            doc.metadata["chunk_embeddings"] = chunk_emb
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def create_vectorstore(docs: list[Document], multi_vector=False) -> Qdrant:
    """Create Qdrant vectorstore from documents with hybrid search capabilities.

    Initializes a Qdrant vector database with the provided documents, using
    HuggingFace embeddings and enabling hybrid search for both semantic and
    keyword-based retrieval.

    Args:
        docs: List of Document objects to index in the vectorstore.
        multi_vector: Whether to enable multi-vector embeddings for enhanced
            representation capabilities.

    Returns:
        Configured Qdrant vectorstore instance ready for search operations.

    """
    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url)
    embeddings = HuggingFaceEmbeddings(model_name=settings.default_embedding_model)
    if multi_vector:
        return Qdrant.from_documents(
            docs,
            embeddings,
            client=client,
            collection_name="docmind",
            hybrid=True,
            multi_vector=True,
        )
    return Qdrant.from_documents(
        docs, embeddings, client=client, collection_name="docmind", hybrid=True
    )


def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken encoding.

    Provides an approximation of how many tokens the input text will consume
    when processed by language models, using the cl100k_base encoding.

    Args:
        text: Input text to analyze for token count.

    Returns:
        Estimated number of tokens in the text.

    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def aggregate_results(
    results: list[AnalysisOutput | str],
) -> AnalysisOutput | str:
    """Aggregate chunked analysis results into a single output.

    Combines multiple analysis results from document chunks into a unified
    result by concatenating summaries and merging lists of insights, action
    items, and questions. Falls back to string concatenation for raw outputs.

    Args:
        results: List of analysis results, either structured AnalysisOutput
            objects or raw string outputs from failed parsing attempts.

    Returns:
        Aggregated AnalysisOutput object if valid results exist, otherwise
        a concatenated string of all raw outputs.

    """
    valid_results = [r for r in results if isinstance(r, AnalysisOutput)]
    if not valid_results:
        return " ".join([str(r) for r in results])
    summary = " ".join([r.summary for r in valid_results])
    key_insights = [insight for r in valid_results for insight in r.key_insights]
    action_items = [item for r in valid_results for item in r.action_items]
    open_questions = [q for r in valid_results for q in r.open_questions]
    return AnalysisOutput(
        summary=summary,
        key_insights=key_insights,
        action_items=action_items,
        open_questions=open_questions,
    )


def analyze_documents(
    llm,
    texts: list[str],
    prompt_type: str,
    custom_prompt: str,
    tone: str,
    instruction: str,
    custom_instruction: str,
    length_detail: str,
    context_size: int,
    chunked: bool = False,
) -> list[AnalysisOutput | str]:
    """Analyze documents with customizable prompts and optional chunking.

    Processes document texts using a large language model with configurable
    analysis parameters. Handles both regular and chunked analysis approaches
    based on document size and user preferences.

    Args:
        llm: Language model instance for text processing.
        texts: List of document texts to analyze.
        prompt_type: Type of analysis prompt to use from predefined options.
        custom_prompt: Custom prompt text if prompt_type is "Custom Prompt".
        tone: Desired tone for the analysis (professional, academic, etc.).
        instruction: Role-based instructions for the analysis approach.
        custom_instruction: Custom instruction text if needed.
        length_detail: Desired length and detail level for the output.
        context_size: Maximum context window size for the language model.
        chunked: Whether to force chunked analysis for large documents.

    Returns:
        List of analysis results, either structured AnalysisOutput objects
        or raw string outputs in case of parsing failures. Returns single
        result if only one text is provided.

    """
    prompt_text = (
        custom_prompt
        if prompt_type == "Custom Prompt"
        else PREDEFINED_PROMPTS[prompt_type]
    )
    tone_text = TONES[tone]
    instr_text = (
        custom_instruction
        if instruction == "Custom Instructions"
        else INSTRUCTIONS[instruction]
    )
    length_text = LENGTHS[length_detail]

    full_prompt = (
        f"{instr_text} {tone_text} {length_text} {prompt_text} Document: {{text}}"
    )

    parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
    prompt = PromptTemplate(
        template=full_prompt + "\n{format_instructions}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    results = []
    for text in texts:
        if chunked:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=context_size // 2, chunk_overlap=100
            )
            chunks = splitter.split_text(text)
            chunk_results = []
            for chunk in chunks:
                try:
                    output = chain.run(text=chunk)
                    parsed = parser.parse(output)
                    chunk_results.append(parsed)
                except Exception as e:
                    chunk_results.append(f"Error: {str(e)} - Raw output: {output}")
            aggregated = aggregate_results(chunk_results)
            results.append(aggregated)
        else:
            tokens = estimate_tokens(text)
            if tokens > context_size * 0.8:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=context_size // 2, chunk_overlap=100
                )
                chunks = splitter.create_documents([text])
                sum_chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = sum_chain.run(chunks)
                text = summary if isinstance(summary, str) else summary[0].page_content
            try:
                output = chain.run(text=text)
                parsed = parser.parse(output)
                results.append(parsed)
            except Exception as e:
                results.append(f"Error: {str(e)} - Raw output: {output}")
    return results if len(results) > 1 else results[0]


class JinaRerankCompressor:
    """Custom document compressor using local Jina reranker model.

    Implements contextual document compression by reranking retrieved documents
    based on their relevance to the query using a cross-encoder model. This
    improves the quality of context provided to the language model by filtering
    and ranking the most relevant documents.

    Attributes:
        model: CrossEncoder instance for document reranking.
        top_n: Number of top documents to return after reranking.

    """

    def __init__(self, top_n=5):
        """Initialize the Jina reranker compressor.

        Args:
            top_n: Maximum number of documents to return after reranking.
                Defaults to 5.

        """
        self.model = CrossEncoder(Settings.default_reranker_model)
        self.top_n = top_n

    def compress_documents(self, documents, query):
        """Compress documents by reranking based on query relevance.

        Uses the Jina cross-encoder model to score document relevance against
        the query, then returns the top-k most relevant documents in order
        of relevance score.

        Args:
            documents: List of Document objects to rerank.
            query: Query string to use for relevance scoring.

        Returns:
            List of top-k most relevant documents, sorted by relevance score
            in descending order.

        """
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        sorted_docs = [
            doc for _, doc in sorted(zip(scores, documents, strict=False), reverse=True)
        ]
        return sorted_docs[: self.top_n]


def chat_with_context(
    llm, vectorstore: Qdrant, user_input: str, history: list[dict]
) -> Generator[str, None, None]:
    """Stream chat response with custom reranker and hybrid retrieval.

    Provides an interactive chat interface that combines the user's query with
    conversation history and retrieves relevant context from the vectorstore.
    Uses hybrid search and document reranking to provide high-quality context
    for response generation.

    Args:
        llm: Language model instance for generating responses.
        vectorstore: Qdrant vectorstore containing indexed documents.
        user_input: Current user query or message.
        history: List of previous conversation exchanges, each containing
            'user' and 'assistant' keys with message content.

    Yields:
        Streaming response tokens from the language model, allowing for
        real-time display of the generated response.

    """
    compressor = JinaRerankCompressor()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(
            search_type="hybrid", search_kwargs={"k": 10}
        ),
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=compression_retriever
    )
    history_str = "\n".join(
        [f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history]
    )
    input_text = f"History: {history_str}\nUser: {user_input}"
    yield from qa_chain.llm.stream(input_text)
