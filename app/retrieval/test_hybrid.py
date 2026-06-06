from app.retrieval.hybrid_retriever import (
    HybridRetriever
)

retriever = HybridRetriever()

docs = retriever.invoke(
    "What is supervised learning?"
)

for i, doc in enumerate(docs):

    print(
        f"\nResult {i+1}"
    )

    print(
        doc.page_content[:300]
    )