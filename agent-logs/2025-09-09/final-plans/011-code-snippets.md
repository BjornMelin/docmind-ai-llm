# Title: DocMind AI — Canonical Code Snippets (GPT‑5 PRO + Research)

**Date:** 2025-09-09

**Note:** These snippets are lifted and normalized from agent-logs/2025-09-09/reviews/reference/005.1-gpt-5-pro-final-plans.md and our research. Replace UNVERIFIED adapters with the thinner wrappers described in the implementation plans when wiring into the current codebase.

1) src/app.py (programmatic navigation)

```python
import streamlit as st

st.set_page_config(page_title="DocMind AI", layout="wide")

chat = st.Page("src/pages/01_chat.py", title="Chat", icon=":material/chat:")
docs = st.Page("src/pages/02_documents.py", title="Documents", icon=":material/description:")
analytics = st.Page("src/pages/03_analytics.py", title="Analytics", icon=":material/insights:")
settings = st.Page("src/pages/04_settings.py", title="Settings", icon=":material/settings:")

st.navigation([chat, docs, analytics, settings]).run()
```

2) src/pages/01_chat.py (chat with native streaming + optional router override)

```python
from __future__ import annotations
import streamlit as st

st.title("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    from src.agents.coordinator import MultiAgentCoordinator
    coord = MultiAgentCoordinator()
    overrides = {"router_engine": st.session_state["router_engine"]} if "router_engine" in st.session_state else None
    resp = coord.process_query(query=prompt, context=None, settings_override=overrides)
    answer = getattr(resp, "content", str(resp))
    def _stream():
        # naive chunking fallback
        for i in range(0, len(answer), 48):
            yield answer[i:i+48]
    with st.chat_message("assistant"):
        final_text = st.write_stream(_stream)
    st.session_state.messages.append({"role": "assistant", "content": final_text})
```

3) src/pages/02_documents.py (form + status/toast + router setup)

```python
from __future__ import annotations
import streamlit as st

st.title("Documents")

with st.form("ingest_form", clear_on_submit=False):
    files = st.file_uploader("Add files", type=None, accept_multiple_files=True)
    use_graphrag = st.checkbox("Enable GraphRAG", value=False)
    submitted = st.form_submit_button("Ingest")

if submitted:
    if not files:
        st.warning("No files selected.")
    else:
        from src.ui.ingest_adapter import ingest_files

        with st.status("Ingesting…", expanded=True) as status:
            try:
                count = ingest_files(files, enable_graphrag=use_graphrag)
                # Build router engine after indexing
                from src.utils.storage import create_vector_store
                from llama_index.core import VectorStoreIndex
                from src.retrieval.query_engine import ServerHybridRetriever, _HybridParams, create_adaptive_router_engine
                vs = create_vector_store(settings.database.qdrant_collection, enable_hybrid=True)
                vector_index = VectorStoreIndex.from_vector_store(vs)
                retriever = ServerHybridRetriever(_HybridParams(collection=settings.database.qdrant_collection))
                st.session_state.router_engine = create_adaptive_router_engine(
                    vector_index=vector_index, hybrid_retriever=retriever
                )
                st.write(f"Ingested {count} documents.")
                status.update(label="Done", state="complete")
                st.toast("Ingestion complete", icon="✅")
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"Ingestion failed: {e}")
```

4) src/core/analytics.py (DuckDB analytics manager)

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Optional, Any

import duckdb

@dataclass(frozen=True)
class AnalyticsConfig:
    enabled: bool
    db_path: Path
    retention_days: int = 30

class AnalyticsManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, cfg: AnalyticsConfig):
        self.cfg = cfg
        self._q: "Queue[tuple[str, tuple[Any, ...]]]" = Queue()
        self._worker: Optional[threading.Thread] = None
        self._last_prune: datetime = datetime.min.replace(tzinfo=timezone.utc)
        if self.cfg.enabled:
            self._ensure_dirs()
            self._ensure_schema()

    @classmethod
    def instance(cls, cfg: AnalyticsConfig):
        with cls._lock:
            if cls._instance is None or cls._instance.cfg != cfg:
                cls._instance = cls(cfg)
            return cls._instance

    def _ensure_dirs(self) -> None:
        os.makedirs(self.cfg.db_path.parent, exist_ok=True)

    def _conn(self):
        return duckdb.connect(str(self.cfg.db_path))

    def _ensure_schema(self) -> None:
        with self._conn() as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS query_metrics(
                ts TIMESTAMP,
                query_type TEXT,
                latency_ms DOUBLE,
                result_count INTEGER,
                retrieval_strategy TEXT,
                success BOOLEAN
            );
            """)
            con.execute("""
            CREATE TABLE IF NOT EXISTS embedding_metrics(
                ts TIMESTAMP, model TEXT, items INT, latency_ms DOUBLE
            );
            """)
            con.execute("""
            CREATE TABLE IF NOT EXISTS reranking_metrics(
                ts TIMESTAMP, model TEXT, items INT, latency_ms DOUBLE
            );
            """)
            con.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics(
                ts TIMESTAMP, key TEXT, value DOUBLE
            );
            """)

    def _start_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                sql, params = self._q.get(timeout=2.0)
            except Empty:
                now = datetime.now(timezone.utc)
                if (now - self._last_prune).total_seconds() > 3600:
                    self.prune_old_records()
                    self._last_prune = now
                continue
            try:
                with self._conn() as con:
                    con.execute(sql, params)
            except Exception:
                pass

    def log_query(self, *, query_type: str, latency_ms: float, result_count: int,
                  retrieval_strategy: str, success: bool) -> None:
        if not self.cfg.enabled:
            return
        self._start_worker()
        self._q.put((
            "INSERT INTO query_metrics VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc), query_type, latency_ms, result_count, retrieval_strategy, success),
        ))

    def log_embedding(self, *, model: str, items: int, latency_ms: float) -> None:
        if not self.cfg.enabled:
            return
        self._start_worker()
        self._q.put((
            "INSERT INTO embedding_metrics VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc), model, items, latency_ms),
        ))

    def log_reranking(self, *, model: str, items: int, latency_ms: float) -> None:
        if not self.cfg.enabled:
            return
        self._start_worker()
        self._q.put((
            "INSERT INTO reranking_metrics VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc), model, items, latency_ms),
        ))

    def prune_old_records(self) -> None:
        if not self.cfg.enabled:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.cfg.retention_days)
        with self._conn() as con:
            for table in ("query_metrics", "embedding_metrics", "reranking_metrics", "system_metrics"):
                con.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
```

5) src/pages/03_analytics.py (charts)

```python
from __future__ import annotations
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb

from src.config.settings import settings

st.title("Analytics")

if not getattr(settings, "analytics_enabled", False):
    st.info("Analytics disabled. Enable DOCMIND_ANALYTICS__ENABLED=true and retry.")
    st.stop()

db_path = Path(settings.data_dir) / "analytics" / "analytics.duckdb"
if not db_path.exists():
    st.warning("No analytics DB yet.")
    st.stop()

con = duckdb.connect(str(db_path))

st.subheader("Query volumes by strategy")
df_strategy = con.execute("""
    SELECT retrieval_strategy, COUNT(*) AS n
    FROM query_metrics
    GROUP BY retrieval_strategy
    ORDER BY n DESC
""").df()
st.plotly_chart(px.bar(df_strategy, x="retrieval_strategy", y="n"), use_container_width=True)

st.subheader("Latency over time (avg ms)")
df_latency = con.execute("""
    SELECT date_trunc('day', ts) AS day, AVG(latency_ms) AS avg_ms
    FROM query_metrics
    GROUP BY 1
    ORDER BY 1
""").df()
st.plotly_chart(px.line(df_latency, x="day", y="avg_ms"), use_container_width=True)

st.subheader("Success rate")
df_success = con.execute("""
    SELECT success, COUNT(*) AS n
    FROM query_metrics
    GROUP BY success
""").df()
st.plotly_chart(px.bar(df_success, x="success", y="n"), use_container_width=True)
```

6) tools/eval/run_beir.py and 7) tools/eval/run_ragas.py — see full listings in reference 005.1; implement per the snippets above.

8) Test snippets — analytics, models_pull, and ragas CLI smoke test are included above.

10) Streaming adapters (sync/async variants)
```python
# A) Sync pipeline returning final text; chunked generator
def chunked_stream(text: str, chunk_size: int = 48):
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]

def render_answer_sync(answer_text: str):
    final_text = st.write_stream(chunked_stream(answer_text))
    return final_text

# B) Async pipeline returning final text; wrap with asyncio.run and chunk
import asyncio

async def get_answer_async(coordinator, prompt: str) -> str:
    resp = await coordinator.process_query(prompt)
    return getattr(resp, "content", str(resp))

def render_answer_async(coordinator, prompt: str):
    text = asyncio.run(get_answer_async(coordinator, prompt))
    return st.write_stream(chunked_stream(text))

# C) LlamaIndex/Agent native streaming (if exposed)
def render_agent_stream(agent, prompt: str, history):
    def _stream():
        yield from agent.stream(prompt, history=history)
    return st.write_stream(_stream)
```

11) Ingestion adapter skeleton (ui/ingest_adapter.py)
```python
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Sequence
from src.config.settings import settings
from src.processing.document_processor import DocumentProcessor

def _save_uploaded_file(file) -> Path:
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.name).name.replace("..", "_")
    path = upload_dir / safe_name
    with path.open("wb") as f:
        f.write(file.getbuffer())
    return path

def ingest_files(files: Sequence, enable_graphrag: bool = False) -> int:
    proc = DocumentProcessor(settings=settings)
    count = 0
    for file in files:
        path = _save_uploaded_file(file)
        asyncio.run(proc.process_document_async(path))
        count += 1
    # Optional GraphRAG post-process here if enable_graphrag
    return count
```

12) Ensure Qdrant collection (named vectors) helper
```python
from qdrant_client import QdrantClient
from qdrant_client import models as qm

def ensure_collection(client: QdrantClient, name: str):
    try:
        client.get_collection(name)
        return
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors={
            "text-dense": qm.VectorParams(size=1024, distance=qm.Distance.COSINE),
        },
        sparse_vectors={
            "text-sparse": qm.SparseVectorParams(),
        },
    )
```

13) BEIR indexing builder (doc_id payload mapping)
```python
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

def build_index_from_corpus(corpus: dict, collection: str, qdrant_client) -> VectorStoreIndex:
    docs = []
    for doc_id, data in corpus.items():
        docs.append(Document(text=data.get("text", ""), metadata={"doc_id": doc_id, "title": data.get("title", "")}))
    vs = QdrantVectorStore(client=qdrant_client, collection_name=collection)
    storage = StorageContext.from_defaults(vector_store=vs)
    return VectorStoreIndex.from_documents(docs, storage_context=storage)
```

14) GraphRAG export helpers (JSONL fallback)
```python
import json
from pathlib import Path

def export_graph_jsonl(index, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pseudocode: fetch triplets from the property graph store
    triplets = [("A", "USES", "B"), ("Model", "OPTIMIZED_FOR", "GPU")]
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in triplets:
            f.write(json.dumps({"head": h, "relation": r, "tail": t}) + "\n")
```

15) Telemetry event example (retrieval)
```python
from src.utils.telemetry import log_jsonl

log_jsonl({
    "retrieval.fusion_mode": "rrf",
    "retrieval.prefetch_dense_limit": 200,
    "retrieval.prefetch_sparse_limit": 400,
    "retrieval.fused_limit": 60,
    "retrieval.return_count": 10,
    "retrieval.latency_ms": 145,
    "dedup.before": 120,
    "dedup.after": 60,
    "dedup.dropped": 60,
    "dedup.key": "page_id",
})
```
16) BEIR retrieval mapping variant using .retrieve()
```python
# For each query id/text from BEIR dataset
results = {}
for qid, qtext in queries.items():
    nodes = retriever.retrieve(qtext)  # returns list[NodeWithScore]
    doc_scores = {}
    for nws in nodes:
        doc_id = (nws.node.metadata or {}).get("doc_id")
        if doc_id:
            doc_scores[str(doc_id)] = float(nws.score)
    results[qid] = doc_scores
```

17) Chat page variant including provider badge
```python
import streamlit as st
from src.ui.components.provider_badge import provider_badge
from src.config.settings import settings

st.title("Chat")
provider_badge(settings)
# ... render history and input as in Section 2
```
