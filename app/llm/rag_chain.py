import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.utils.logger import get_logger
from app.retrieval.vector_store import get_retriever
from app.llm.prompt import get_prompt

load_dotenv()

logger = get_logger(__name__)


# ===== LLM =====
def get_llm():
    try:
        logger.info("Initializing LLM")

        return ChatOpenAI(
            model="gpt-4",
            temperature=0
        )

    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise


# ===== Format Docs =====
def format_docs(docs):
    try:
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.error(f"Error formatting docs: {e}")
        return ""


# ===== Build RAG Chain =====
def get_rag_chain(mode: str = "default", k: int = 5):
    try:
        logger.info(f"Building RAG chain | mode={mode}, k={k}")

        retriever = get_retriever(k=k)
        prompt = get_prompt(mode=mode)
        llm = get_llm()

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": lambda x: x
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain

    except Exception as e:
        logger.error(f"Error building RAG chain: {e}")
        raise


# ===== Main (Testing) =====
def main():
    logger.info("Starting RAG test")

    try:
        rag_chain = get_rag_chain(mode="default", k=5)

        query = "What is unsupervised learning?"
        response = rag_chain.invoke(query)

        logger.info("Response generated successfully")
        print("\nResponse:\n", response)

    except Exception as e:
        logger.error(f"Error during RAG execution: {e}")


if __name__ == "__main__":
    main()