from app.retrieval.vector_store import (
    get_retriever,
    get_documents
)

from app.retrieval.bm25_retriever import (
    get_bm25_retriever
)

from app.retrieval.rrf import (
    reciprocal_rank_fusion
)

from app.retrieval.reranker import (
    CrossEncoderReranker
)


class HybridRetriever:

    def __init__(
        self,
        dense_k=10,
        sparse_k=10,
        rerank_top_k=5
    ):
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rerank_top_k = rerank_top_k

        # FAISS Retriever
        self.faiss_retriever = get_retriever(
            k=self.dense_k
        )

        # BM25 Retriever
        documents = get_documents()

        self.bm25_retriever = (
            get_bm25_retriever(
                documents=documents,
                k=self.sparse_k
            )
        )

        # Cross Encoder Reranker
        self.reranker = (
            CrossEncoderReranker()
        )

    def dense_retrieve(
        self,
        query
    ):
        return self.faiss_retriever.invoke(
            query
        )

    def sparse_retrieve(
        self,
        query
    ):
        return self.bm25_retriever.invoke(
            query
        )

    def retrieve(
        self,
        query
    ):

        # Dense Retrieval
        faiss_docs = (
            self.dense_retrieve(
                query
            )
        )

        # Sparse Retrieval
        bm25_docs = (
            self.sparse_retrieve(
                query
            )
        )

        # RRF Fusion
        fused_docs = (
            reciprocal_rank_fusion(
                faiss_docs,
                bm25_docs
            )
        )

        return fused_docs

    def rerank(
        self,
        query,
        documents
    ):
        return self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=self.rerank_top_k
        )

    def invoke(
        self,
        query
    ):

        # Hybrid Retrieval
        retrieved_docs = (
            self.retrieve(
                query
            )
        )

        # Cross Encoder Re-ranking
        reranked_docs = (
            self.rerank(
                query=query,
                documents=retrieved_docs
            )
        )

        return reranked_docs


if __name__ == "__main__":

    retriever = HybridRetriever(
        dense_k=10,
        sparse_k=10,
        rerank_top_k=5
    )

    query = "What is supervised learning?"

    results = retriever.invoke(
        query
    )

    print(
        f"\nRetrieved {len(results)} documents\n"
    )

    for idx, doc in enumerate(
        results,
        start=1
    ):
        print(
            f"\n===== Result {idx} =====\n"
        )

        print(
            doc.page_content[:500]
        )