import os
from threading import Lock

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.utils.logger import get_logger
from app.retrieval.vector_store import get_retriever
from app.llm.prompt import get_prompt

logger = get_logger(__name__)

# ===== Cache + Lock =====
_cache = {}
_lock = Lock()


# ===== Format Docs =====
def format_docs(docs):
    try:
        return "\n\n".join(doc.page_content[:200] for doc in docs)
    except Exception as e:
        logger.error(f"Error formatting docs: {e}")
        return ""


# ===== Build + Cache RAG Chain =====
def get_rag_chain(mode: str = "default", k: int = 2):
    key = (mode, k)

    # Fast path
    if key in _cache:
        return _cache[key]

    # Thread-safe initialization
    with _lock:
        if key in _cache:
            return _cache[key]

        try:
            logger.info(f"Initializing RAG chain | mode={mode}, k={k}")

            llm_model = "gpt-4o-mini"
            temperature = 0

            retriever = get_retriever(k=k)
            prompt = get_prompt(mode=mode)

            llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                streaming=False
            )

            chain = (
                {
                    "context": retriever | format_docs,
                    "question": lambda x: x
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            _cache[key] = chain
            return chain

        except Exception as e:
            logger.error(f"Error building RAG chain: {e}")
            raise


# ===== Main (Testing Only) =====
def main():
    logger.info("Starting RAG test")

    try:
        rag_chain = get_rag_chain(mode="default", k=3)

        query = "What is unsupervised learning?"
        response = rag_chain.invoke(query)

        logger.info("Response generated successfully")
        print("\nResponse:\n", response)

    except Exception as e:
        logger.error(f"Error during RAG execution: {e}")


if __name__ == "__main__":
    main()