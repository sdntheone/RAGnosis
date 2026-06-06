from app.retrieval.hybrid_retriever import (
    HybridRetriever
)

retriever = HybridRetriever(
    dense_k=10,
    sparse_k=10,
    rerank_top_k=5
)

docs = retriever.invoke(
    "What is supervised learning?"
)

for i, doc in enumerate(docs, start=1):

    print(f"\n===== Result {i} =====\n")

    print(
        doc.page_content[:500]
    )