import os
import mlflow
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from app.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# =========================
# Paths
# =========================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

VECTOR_DB_PATH = os.path.join(
    BASE_DIR,
    "vector_db",
    "faiss_index"
)

# =========================
# Global Cache
# =========================

_embedding_model = None
_vectorstore = None
_documents = None

# =========================
# Embedding Model
# =========================

def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        try:
            logger.info(
                "Initializing embedding model"
            )

            _embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                chunk_size=1000
            )

            mlflow.log_param(
                "embedding_model",
                "text-embedding-3-small"
            )

        except Exception as e:
            logger.error(
                f"Embedding initialization failed: {e}"
            )
            raise

    return _embedding_model


# =========================
# Create Vector Store
# =========================

def create_vectorstore(chunks):
    global _documents

    try:
        logger.info(
            "Creating vector store"
        )

        embedding_model = get_embedding_model()

        vectorstore = FAISS.from_documents(
            chunks,
            embedding_model
        )

        os.makedirs(
            os.path.dirname(VECTOR_DB_PATH),
            exist_ok=True
        )

        vectorstore.save_local(
            VECTOR_DB_PATH
        )

        _documents = chunks

        logger.info(
            f"Vector store saved at {VECTOR_DB_PATH}"
        )

        return vectorstore

    except Exception as e:
        logger.error(
            f"Vector store creation failed: {e}"
        )
        raise


# =========================
# Load Vector Store
# =========================

def load_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        try:
            logger.info(
                "Loading vector store"
            )

            embedding_model = (
                get_embedding_model()
            )

            _vectorstore = FAISS.load_local(
                VECTOR_DB_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )

            logger.info(
                "Vector store loaded successfully"
            )

        except Exception as e:
            logger.error(
                f"Failed to load vector store: {e}"
            )
            raise

    return _vectorstore


# =========================
# Get Documents
# Needed for BM25
# =========================

def get_documents():
    global _documents

    if _documents is None:

        vectorstore = load_vectorstore()

        _documents = list(
            vectorstore.docstore._dict.values()
        )

    return _documents


# =========================
# Dense Retriever (FAISS)
# =========================

def get_retriever(k=5):
    try:
        logger.info(
            f"Creating FAISS retriever k={k}"
        )

        mlflow.log_param(
            "retrieval_k",
            k
        )

        vectorstore = load_vectorstore()

        return vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    except Exception as e:
        logger.error(
            f"Retriever creation failed: {e}"
        )
        raise


# =========================
# Similarity Search
# Useful for debugging
# =========================

def similarity_search(
    query,
    k=5
):
    vectorstore = load_vectorstore()

    return vectorstore.similarity_search(
        query,
        k=k
    )


# =========================
# Test
# =========================

if __name__ == "__main__":

    retriever = get_retriever(k=3)

    docs = retriever.invoke(
        "What is overfitting?"
    )

    print(f"Retrieved {len(docs)} docs")

    for i, doc in enumerate(docs, start=1):
        print(
            f"\n----- Doc {i} -----"
        )
        print(
            doc.page_content[:300]
        )