from threading import Lock

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.utils.logger import get_logger
from app.retrieval.hybrid_retriever import HybridRetriever
from app.llm.prompt import get_prompt

logger = get_logger(__name__)

# ===== Cache + Lock =====

_cache = {}
_lock = Lock()


# ===== Format Docs =====

def format_docs(docs):
    try:
        return "\n\n".join(
            doc.page_content[:200]
            for doc in docs
        )

    except Exception as e:
        logger.error(
            f"Error formatting docs: {e}"
        )

        return ""


# ===== Build + Cache RAG Chain =====

def get_rag_chain(
    mode: str = "default",
    k: int = 5
):

    key = (mode, k)

    if key in _cache:
        return _cache[key]

    with _lock:

        if key in _cache:
            return _cache[key]

        try:

            logger.info(
                f"Initializing Advanced RAG Chain | mode={mode}"
            )

            prompt = get_prompt(
                mode=mode
            )

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                streaming=False
            )

            # ===== Advanced Retriever =====

            retriever = HybridRetriever(
                dense_k=10,
                sparse_k=10,
                rerank_top_k=k
            )

            chain = (
                {
                    "context": lambda query: format_docs(
                        retriever.invoke(query)
                    ),
                    "question": lambda query: query
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            _cache[key] = chain

            logger.info(
                "Advanced RAG Chain initialized successfully"
            )

            return chain

        except Exception as e:

            logger.error(
                f"Error building RAG chain: {e}"
            )

            raise


# ===== Main (Testing Only) =====

def main():

    logger.info(
        "Starting Advanced RAG Test"
    )

    try:

        rag_chain = get_rag_chain(
            mode="default",
            k=5
        )

        query = (
            "What is supervised learning?"
        )

        response = rag_chain.invoke(
            query
        )

        print("\nResponse:\n")
        print(response)

        logger.info(
            "Response generated successfully"
        )

    except Exception as e:

        logger.error(
            f"Error during RAG execution: {e}"
        )


if __name__ == "__main__":
    main()