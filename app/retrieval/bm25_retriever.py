from langchain_community.retrievers import BM25Retriever


def get_bm25_retriever(documents, k=10):
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever