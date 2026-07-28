"""
app/sessions/vector_backends/faiss_backend.py

Default VectorStoreBackend implementation: one FAISS index per session,
stored on disk under a session-specific directory. Zero extra
infrastructure required -- matches the "local FAISS per-session" choice.

This is a completely separate FAISS index from the one in
app/retrieval/vector_store.py (which stays untouched and keeps serving the
original global /chat pipeline). Each session gets its own directory under
SESSION_VECTOR_DB_ROOT, so sessions never share or leak documents.

Metadata filtering: LangChain's FAISS wrapper only supports a callable
filter function (not a query language), so build_filter()'s dict output
(app/chunking/metadata_builder.py) is translated into a predicate function
here.
"""

from __future__ import annotations

import os
import pickle
import threading

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.sessions.vector_backends.base import VectorStoreBackend

SESSION_VECTOR_DB_ROOT = os.path.join("vector_db", "sessions")

try:
    # Reuse the exact embedding model already configured for the existing
    # pipeline, if that function exists.
    from app.retrieval.vector_store import get_embedding_model as _get_embedding_model
except ImportError:
    _get_embedding_model = None

_fallback_embeddings = None


def _resolve_embedding_model():
    if _get_embedding_model is not None:
        return _get_embedding_model()

    global _fallback_embeddings
    if _fallback_embeddings is None:
        from langchain_openai import OpenAIEmbeddings
        _fallback_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _fallback_embeddings


def _matches_filter(metadata: dict, metadata_filter: dict | None) -> bool:
    if not metadata_filter:
        return True
    for key, condition in metadata_filter.items():
        value = metadata.get(key)
        if isinstance(condition, dict) and "$in" in condition:
            if value not in condition["$in"]:
                return False
        else:
            if value != condition:
                return False
    return True

def _distance_to_similarity(distance: float) -> float:
    """FAISS returns L2 distance (lower = better, unbounded). Convert to a
    0-1 similarity score (higher = better) for use by callers like
    retrieval_validation.py and chat_stream_routes.py.
    """
    return 1.0 / (1.0 + max(distance, 0.0))


class FaissSessionBackend(VectorStoreBackend):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._dir = os.path.join(SESSION_VECTOR_DB_ROOT, session_id)
        self._index_path = os.path.join(self._dir, "faiss_index")
        self._docs_path = os.path.join(self._dir, "documents.pkl")
        self._lock = threading.Lock()

        os.makedirs(self._dir, exist_ok=True)
        self._embeddings = _resolve_embedding_model()
        self._store: FAISS | None = self._load()
        self._documents: list[Document] = self._load_documents()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> FAISS | None:
        if os.path.exists(self._index_path):
            try:
                return FAISS.load_local(
                    self._index_path, self._embeddings, allow_dangerous_deserialization=True
                )
            except Exception:
                return None
        return None

    def _load_documents(self) -> list[Document]:
        if os.path.exists(self._docs_path):
            try:
                with open(self._docs_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return []
        return []

    def _persist(self) -> None:
        if self._store is not None:
            self._store.save_local(self._index_path)
        with open(self._docs_path, "wb") as f:
            pickle.dump(self._documents, f)

    # ------------------------------------------------------------------
    # VectorStoreBackend interface
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        with self._lock:
            if self._store is None:
                self._store = FAISS.from_documents(documents, self._embeddings)
            else:
                self._store.add_documents(documents)
            self._documents.extend(documents)
            self._persist()

    def similarity_search(
        self, query: str, k: int = 5, metadata_filter: dict | None = None
    ) -> list[Document]:
        if self._store is None:
            return []

        with self._lock:
            # Over-fetch when filtering, since FAISS filters post-search.
            fetch_k = k * 5 if metadata_filter else k
            results = self._store.similarity_search(query, k=fetch_k)

        if metadata_filter:
            results = [r for r in results if _matches_filter(r.metadata, metadata_filter)]

        return results[:k]

    def similarity_search_with_scores(
        self, query: str, k: int = 5, metadata_filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        if self._store is None:
            return []

        with self._lock:
            fetch_k = k * 5 if metadata_filter else k
            results = self._store.similarity_search_with_score(query, k=fetch_k)

        scored = [
            (doc, _distance_to_similarity(distance))
            for doc, distance in results
            if _matches_filter(doc.metadata, metadata_filter)
        ]

        return scored[:k]

    def get_all_documents(self, metadata_filter: dict | None = None) -> list[Document]:
        with self._lock:
            docs = list(self._documents)
        if metadata_filter:
            docs = [d for d in docs if _matches_filter(d.metadata, metadata_filter)]
        return docs

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            keep = [d for d in self._documents if d.metadata.get("doc_id") != doc_id]
            removed = len(self._documents) - len(keep)
            if removed == 0:
                return 0

            self._documents = keep
            if keep:
                self._store = FAISS.from_documents(keep, self._embeddings)
            else:
                self._store = None
                if os.path.exists(self._index_path):
                    import shutil
                    shutil.rmtree(self._index_path, ignore_errors=True)
            self._persist()
            return removed

    def clear(self) -> None:
        with self._lock:
            self._documents = []
            self._store = None
            if os.path.exists(self._index_path):
                import shutil
                shutil.rmtree(self._index_path, ignore_errors=True)
            if os.path.exists(self._docs_path):
                os.remove(self._docs_path)

    def document_count(self) -> int:
        with self._lock:
            return len({d.metadata.get("doc_id") for d in self._documents})

    def chunk_count(self) -> int:
        with self._lock:
            return len(self._documents)