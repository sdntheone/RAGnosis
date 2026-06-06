from collections import defaultdict


def reciprocal_rank_fusion(
    faiss_docs,
    bm25_docs,
    k=60
):
    scores = defaultdict(float)

    for rank, doc in enumerate(faiss_docs):
        doc_id = doc.page_content

        scores[doc_id] += 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.page_content

        scores[doc_id] += 1 / (k + rank + 1)

    all_docs = {
        doc.page_content: doc
        for doc in faiss_docs + bm25_docs
    }

    ranked_docs = sorted(
        all_docs.values(),
        key=lambda doc: scores[doc.page_content],
        reverse=True
    )

    return ranked_docs