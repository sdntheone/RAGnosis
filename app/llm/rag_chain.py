import os
import mlflow
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.utils.logger import get_logger
from app.retrieval.vector_store import get_retriever
from app.llm.prompt import get_prompt

load_dotenv()

logger = get_logger(__name__)

# ===== Global Cache =====
_rag_chain = None


# ===== Format Docs (trimmed for speed) =====
def format_docs(docs):
    try:
        return "\n\n".join(doc.page_content[:200] for doc in docs)
    except Exception as e:
        logger.error(f"Error formatting docs: {e}")
        return ""


# ===== Build RAG Chain (ONLY ONCE) =====
def get_rag_chain(mode: str = "default", k: int = 2):
    llm_model="gpt-4o-mini"
    temperature=0
    global _rag_chain

    if _rag_chain is None:
        try:
            logger.info(f"Initializing RAG chain (once) | mode={mode}, k={k}")

            retriever = get_retriever(k=k)
            prompt = get_prompt(mode=mode)

            llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                streaming=True
            )

            _rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": lambda x: x
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            mlflow.log_param("llm_model",llm_model)
            mlflow.log_param("temperature",temperature)

        except Exception as e:
            logger.error(f"Error building RAG chain: {e}")
            raise

    return _rag_chain


# ===== Main (Testing) =====
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