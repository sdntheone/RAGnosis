"""
app/sessions/vector_backends/base.py

VectorStoreBackend interface.

Nothing in api_v2/* or sessions/session_manager.py talks to FAISS (or any
other vector store) directly -- everything goes through this interface.
Today's default implementation (faiss_backend.py, next file) is one FAISS
index per session, which needs zero extra infrastructure and matches what
was chosen for this project. If usage ever outgrows local FAISS files, a
QdrantBackend implementing this same interface can be dropped in with a
one-line change in session_manager.py -- no other file changes.

This does not touch app/retrieval/vector_store.py, which remains exactly as
it is and continues to power the original /chat pipeline against the
global book index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class VectorStoreBackend(ABC):
    """One backend instance corresponds to one session's document
    collection. session_manager.py is responsible for creating/caching one
    backend instance per session_id.
    """

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Embed and index the given chunks."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 5, metadata_filter: dict | None = None
    ) -> list[Document]:
        """Return the top-k most similar chunks, optionally restricted by
        a metadata filter (see app/chunking/metadata_builder.py::build_filter
        for the filter dict shape).
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search_with_scores(
        self, query: str, k: int = 5, metadata_filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Same as similarity_search, but also returns a similarity score
        per document, normalized to 0-1 where higher = better match. Each
        backend is responsible for converting its own native score/distance
        metric to this common scale.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_documents(self, metadata_filter: dict | None = None) -> list[Document]:
        """Return every indexed chunk (optionally filtered) -- used to
        build a session-scoped BM25 retriever and for document listing.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to one doc_id. Returns the number of
        chunks removed.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove everything indexed for this session."""
        raise NotImplementedError

    @abstractmethod
    def document_count(self) -> int:
        """Number of distinct doc_ids currently indexed."""
        raise NotImplementedError

    @abstractmethod
    def chunk_count(self) -> int:
        """Number of chunks currently indexed."""
        raise NotImplementedError