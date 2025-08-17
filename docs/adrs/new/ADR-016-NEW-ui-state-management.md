# ADR-016-NEW: Streamlit Native State Management

## Title

Streamlit Native State with LangGraph Memory Integration

## Version/Date

4.0 / 2025-08-17

## Status

Accepted

## Description

Use Streamlit's native `st.session_state` and `st.cache_data` directly without custom abstraction layers. Streamlit's built-in features are sufficient for a local RAG application.

## Context

The original ADR-016 created complex custom state management with:

- Custom SessionStateManager class
- Custom CacheManager wrapper
- Custom real-time synchronization
- Custom performance monitoring

This is over-engineering. Streamlit already provides excellent state management out-of-the-box.

## Decision

We will use **Streamlit native state with LangGraph memory** for conversation persistence:

### Conversation Memory with LangGraph

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from llama_index.memory import ChatMemoryBuffer
import streamlit as st
from typing import List, Dict, Any

class StreamlitChatMemory(BaseChatMessageHistory):
    """Bridge between Streamlit session state and LangGraph memory."""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.key = f"chat_history_{session_id}"
        
        # Initialize in Streamlit session state
        if self.key not in st.session_state:
            st.session_state[self.key] = []
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from Streamlit state."""
        return st.session_state[self.key]
    
    def add_message(self, message: BaseMessage) -> None:
        """Add message to Streamlit state."""
        st.session_state[self.key].append(message)
    
    def clear(self) -> None:
        """Clear conversation history."""
        st.session_state[self.key] = []

# LlamaIndex ChatMemoryBuffer Integration
@st.cache_resource
def get_chat_memory():
    """Get LlamaIndex chat memory with token limit."""
    from llama_index.memory import ChatMemoryBuffer
    
    return ChatMemoryBuffer.from_defaults(
        token_limit=4000,  # Keep last 4k tokens
        tokenizer_fn=lambda text: len(text.split())  # Simple tokenizer
    )

# LangGraph InMemoryStore for Long-term Memory
@st.cache_resource
def get_long_term_memory():
    """Get LangGraph memory store for cross-session persistence."""
    from langgraph.store import InMemoryStore
    
    # This persists across conversations
    store = InMemoryStore()
    
    # Optional: Add Redis backend for true persistence
    # from redis import Redis
    # store = RedisStore(Redis.from_url("redis://localhost:6379"))
    
    return store

# Usage in Chat Interface
def chat_with_memory():
    """Chat interface with integrated memory."""
    
    # Get memory components
    chat_memory = StreamlitChatMemory(st.session_state.get("session_id", "default"))
    llama_memory = get_chat_memory()
    long_term_store = get_long_term_memory()
    
    # Display conversation history
    for message in chat_memory.messages:
        with st.chat_message(message.type):
            st.write(message.content)
    
    # Handle new input
    if prompt := st.chat_input("Ask a question"):
        # Add to memory
        chat_memory.add_message(HumanMessage(content=prompt))
        
        # Get response with memory context
        response = generate_with_memory(
            prompt,
            chat_memory.messages,
            long_term_store
        )
        
        # Add response to memory
        chat_memory.add_message(AIMessage(content=response))
        
        # Update LlamaIndex memory
        llama_memory.put(HumanMessage(content=prompt))
        llama_memory.put(AIMessage(content=response))

# Persist Important Information
def save_to_long_term_memory(key: str, value: Any):
    """Save important facts to long-term memory."""
    store = get_long_term_memory()
    
    # Store with namespace for organization
    namespace = ("user_facts", st.session_state.get("user_id", "default"))
    store.put(namespace, key, value)

def retrieve_from_long_term(key: str) -> Any:
    """Retrieve from long-term memory."""
    store = get_long_term_memory()
    namespace = ("user_facts", st.session_state.get("user_id", "default"))
    
    return store.get(namespace, key)
```

### Simple Session State

```python
import streamlit as st
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class AppState:
    """Simple state container with memory integration."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_page: str = "chat"
    uploaded_docs: List[str] = field(default_factory=list)
    model_config: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_enabled: bool = True
    max_memory_tokens: int = 4000

# Initialize state (in app.py)
if 'app_state' not in st.session_state:
    st.session_state.app_state = AppState()

# Access state anywhere
state = st.session_state.app_state

# Update state
state.messages.append({"role": "user", "content": user_input})
state.current_page = "documents"
```

### Simple Caching

```python
import streamlit as st
from datetime import timedelta

# Cache expensive computations
@st.cache_data(ttl=timedelta(hours=1))
def get_document_embeddings(doc_id: str):
    """Cache document embeddings for 1 hour."""
    return compute_embeddings(doc_id)

@st.cache_data(ttl=timedelta(minutes=5))
def search_documents(query: str, filters: Dict):
    """Cache search results for 5 minutes."""
    return vector_db.search(query, filters)

# Cache resources (singleton objects)
@st.cache_resource
def get_llm_model():
    """Cache LLM model instance."""
    return load_model("Qwen/Qwen3-14B-Instruct")

@st.cache_resource
def get_vector_db():
    """Cache vector database connection."""
    return QdrantClient(path="./data/qdrant")

@st.cache_resource
def get_document_processor():
    """Cache document processor."""
    from unstructured.partition.auto import partition
    return partition
```

### Page Navigation with State Persistence

```python
# app.py - Main entry point
import streamlit as st
from pages import chat, documents, analytics, settings

# Configure app
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple page routing
pages = {
    "Chat": chat.show,
    "Documents": documents.show,
    "Analytics": analytics.show,
    "Settings": settings.show
}

# Sidebar navigation
with st.sidebar:
    selected_page = st.selectbox(
        "Navigate",
        options=list(pages.keys()),
        index=0
    )

# Render selected page
pages[selected_page]()
```

### Real-time Updates with Fragments

```python
import streamlit as st
import time

# Use fragments for partial updates (Streamlit 1.33+)
@st.fragment(run_every=5)  # Update every 5 seconds
def metrics_display():
    """Auto-updating metrics without full page reload."""
    metrics = get_current_metrics()  # Cached function
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", metrics['doc_count'])
    with col2:
        st.metric("Queries Today", metrics['queries'])
    with col3:
        st.metric("Avg Response Time", f"{metrics['avg_time']:.2f}s")

# In your page
metrics_display()
```

### Persistent Settings with JSON

```python
import json
from pathlib import Path

SETTINGS_FILE = Path("./data/settings.json")

def load_settings():
    """Load settings from disk."""
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {
        "model": "Qwen/Qwen3-14B-Instruct",
        "context_length": 131072,  # 128K native context
        "temperature": 0.7,
        "top_k": 10
    }

def save_settings(settings: dict):
    """Save settings to disk."""
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

# Initialize settings in session state
if 'settings' not in st.session_state:
    st.session_state.settings = load_settings()

# Settings page
def show_settings():
    st.header("Settings")
    
    settings = st.session_state.settings
    
    settings['model'] = st.selectbox(
        "Model",
        ["Qwen/Qwen3-14B-Instruct", "Qwen/Qwen3-7B-Instruct", "mistralai/Mistral-7B-Instruct"],
        index=0 if settings['model'].startswith("Qwen3-14B") else (1 if settings['model'].startswith("Qwen3-7B") else 2)
    )
    
    settings['temperature'] = st.slider(
        "Temperature",
        0.0, 1.0, settings['temperature']
    )
    
    if st.button("Save Settings"):
        save_settings(settings)
        st.success("Settings saved!")
        st.rerun()
```

### Chat History Management

```python
# Simple chat history without complex state managers
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add new message
if prompt := st.chat_input("Ask a question"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response (cached)
    response = generate_response(prompt)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()
```

## Benefits of Native Streamlit

- **Zero Abstraction**: Use Streamlit directly as designed
- **Well-Documented**: Extensive Streamlit documentation and examples
- **Community Support**: Large community, many examples
- **Automatic Optimization**: Streamlit handles state efficiently
- **Built-in Features**: Fragments, caching, session state all work together

## What We Removed

- ❌ Custom SessionStateManager class (unnecessary wrapper)
- ❌ Custom CacheManager (st.cache_data is sufficient)
- ❌ Custom real-time synchronization (use fragments)
- ❌ Complex state schemas (simple dataclasses work)
- ❌ Performance monitoring (Streamlit provides this)
- ❌ 500+ lines of state management code

## Performance Tips

1. **Use `st.cache_data` for data**: Automatically handles serialization
2. **Use `st.cache_resource` for objects**: Singletons like models, connections
3. **Use `st.fragment` for partial updates**: Avoid full page reloads
4. **Clear cache selectively**: `st.cache_data.clear()` when needed
5. **Monitor with built-in tools**: `streamlit run app.py --server.enableXsrfProtection=false --logger.level=debug`

## Dependencies

```toml
[project.dependencies]
streamlit = ">=1.36.0"  # Latest version with fragments
```

## Monitoring

Streamlit provides built-in monitoring:

- Session state size: `len(str(st.session_state))`
- Cache statistics: Available in Streamlit Cloud metrics
- Performance: Browser DevTools Network tab
- Memory: `streamlit run app.py --logger.level=debug`

## Changelog

- **3.0 (2025-08-17)**: FINALIZED - Updated with Qwen3 models and 128K context, accepted status
- **2.0 (2025-08-17)**: SIMPLIFIED - Use native Streamlit features only
- **1.0 (2025-08-17)**: Original over-engineered custom state management
