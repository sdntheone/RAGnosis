import os
import mlflow
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# ===== Paths =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db", "faiss_index")

# ===== Global Cache =====
_embedding_model = None
_vectorstore = None


# ===== Embedding Model (Cached) =====
def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        try:
            logger.info("Initializing embedding model (once)")
            _embedding_model = OpenAIEmbeddings(model="text-embedding-3-small",chunk_size=1000)
            mlflow.log_param("embedding_model","text-embedding-3-small")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    return _embedding_model


# ===== Create + Save Vector Store =====
def create_vectorstore(chunks):
    try:
        logger.info("Creating vector store from chunks")

        embedding_model = get_embedding_model()
        vectorstore = FAISS.from_documents(chunks, embedding_model)

        os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
        vectorstore.save_local(VECTOR_DB_PATH)

        logger.info(f"Vector store saved at: {VECTOR_DB_PATH}")
        return vectorstore

    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise


# ===== Load Vector Store (Cached) =====
def load_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        try:
            logger.info("Loading vector store (once)")

            embedding_model = get_embedding_model()

            _vectorstore = FAISS.load_local(
                VECTOR_DB_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )

            logger.info("Vector store loaded successfully")

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    return _vectorstore


# ===== Get Retriever =====
def get_retriever(k=5):
    try:
        logger.info(f"Creating retriever with k={k}")
        mlflow.log_param("retirival_k",k)

        vectorstore = load_vectorstore()
        return vectorstore.as_retriever(search_kwargs={"k": k})

    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise


# ===== Main (Testing) =====
def main():
    logger.info("Starting vector store module test")

    try:
        retriever = get_retriever(k=3)

        query = "What is overfitting?"
        results = retriever.invoke(query)

        logger.info(f"Retrieved {len(results)} documents")

        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}: {doc.metadata.get('source')}")
        mlflow.log_metric("num_docs_retrieved",len(results))

    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()