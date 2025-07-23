# DocMind AI Architecture Overview

## High-Level Components

- **Frontend:** Streamlit UI for uploads, configs, results, chat.
- **Backend:** Ollama/LlamaCpp/LM Studio for LLM inference.
- **Orchestration:** LangChain for chains, retrievers, embeddings.
- **Storage:** Qdrant for hybrid vector search.
- **Processing:** Utils for loading, chunking, analysis.

## Data Flow

1. User uploads docs → Loaded/split in utils.py.
2. Indexed in Qdrant with hybrid embeddings (Jina v4 dense, FastEmbed sparse).
3. Analysis: LLMChain processes with prompts → Structured via Pydantic.
4. Chat: RetrievalQA with reranking (Jina v2 + submodular opt).
5. GPU: torch.cuda for embeddings/reranking if enabled.

## Key Technologies

- Embeddings: HuggingFace (Jina v4), FastEmbed (SPLADE++).
- Optimization: PEFT for efficiency, late chunking with NLTK.

See ADRs in [../adrs/](../adrs/) for decisions.
