# LangChain Usage in DocMind AI

LangChain is the core orchestration framework in DocMind AI, handling LLM interactions, document processing, retrieval, and chaining. This guide explains its integration, key components, and usage in the codebase.

## Overview

- **Version:** langchain>=0.3.26, langchain-community==0.3.27.

- **Purpose:** Simplifies building LLM applications with modular components for chains, retrievers, splitters, embeddings, and vector stores.

- **Integration Points:** Primarily in `src/utils/` for processing and `src/app.py` for invoking analysis/chat.

## Key Components and Usage

1. **Chains:**
   - **LLMChain:** Used in `analyze_documents()` for prompt-based analysis with custom tones/instructions.
     Example:

     ```python
     chain = LLMChain(llm=llm, prompt=prompt)
     output = chain.run(text=text)
     ```

   - **RetrievalQA:** In `chat_with_context()` for RAG-based chat responses.
     Example:

     ```python
     qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever)
     ```

   - **load_summarize_chain:** For map-reduce summarization of large texts in `analyze_documents()`.
     Example:

     ```python
     sum_chain = load_summarize_chain(llm, chain_type="map_reduce")
     summary = sum_chain.run(chunks)
     ```

2. **Retrievers:**
   - **ContextualCompressionRetriever:** Wraps base retrievers with custom compressors (e.g., JinaRerankCompressor) for reranking in chat.
     Example:

     ```python
     compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vectorstore.as_retriever(search_type="hybrid", search_kwargs={"k": 10}))
     ```

3. **Text Splitters:**
   - **RecursiveCharacterTextSplitter:** Splits documents in `load_documents()` and for chunking in analysis.
     Example:

     ```python
     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
     return splitter.split_documents(docs)
     ```

4. **Embeddings:**
   - **HuggingFaceEmbeddings:** For dense embeddings (Jina v4) in vectorstore creation.
     Example:

     ```python
     dense_embeddings = HuggingFaceEmbeddings(model_name=settings.default_embedding_model, model_kwargs={"device_map": "auto"})
     ```

   - **FastEmbedSparse (community):** For sparse embeddings in hybrid search.
     Example:

     ```python
     sparse_embeddings = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1", providers=["CUDAExecutionProvider"] if device == 'cuda' else None)
     ```

5. **Vector Stores:**
   - **QdrantVectorStore (community):** Creates hybrid vectorstores in `create_vectorstore()`.
     Example:

     ```python
     return Qdrant.from_documents(docs, dense_embeddings, sparse_embedding=sparse_embeddings, client=client, collection_name="docmind", hybrid=True)
     ```

6. **Output Parsers:**
   - **PydanticOutputParser:** Structures analysis outputs in `analyze_documents()`.
     Example:

     ```python
     parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
     parsed = parser.parse(output)
     ```

## Best Practices in DocMind AI

- **RAG Pipelines:** Combine retrievers with chains for context-aware chat; use hybrid search for better recall.

- **Document Analysis:** Chain splitters → embeddings → chains for scalable processing.

- **Error Handling:** Fallback to raw outputs if parsing fails; log errors.

- **Extensibility:** LangChain's modular design allows easy swaps (e.g., different embeddings/retrievers).

## References

- LangChain Docs: [Introduction](https://python.langchain.com/docs/get_started/introduction), [Qdrant Integration](https://python.langchain.com/docs/integrations/vectorstores/qdrant).

- Codebase: See `src/utils/` for implementations.

For updates as of July 2025, LangChain continues to support Pydantic 2 and hybrid retrievals without major breaks in our usage.
