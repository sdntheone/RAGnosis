import os
import time
import mlflow
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logger import get_logger
from app.retrieval.vector_store import create_vectorstore
mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "http://localhost:5000"))
mlflow.set_experiment("RAG pipeline")

load_dotenv()
logger = get_logger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")


def load_books():
    start_time = time.time()
    all_docs = []

    try:
        files = os.listdir(DATA_PATH)
    except Exception as e:
        logger.error(f"Failed to access data path: {DATA_PATH} | Error: {e}")
        return []

    for file in files:
        file_path = os.path.join(DATA_PATH, file)

        if os.path.isfile(file_path) and file.endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = file
                    doc.metadata["type"] = "book"

                all_docs.extend(docs)

            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

    mlflow.log_metric("total_documents", len(all_docs))
    mlflow.log_metric("loading_time", time.time() - start_time)

    return all_docs


def split_chunks(docs):
    start_time = time.time()

    chunk_size = 500
    chunk_overlap = 100

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    mlflow.log_param("chunk_size", chunk_size)
    mlflow.log_param("chunk_overlap", chunk_overlap)

    chunks = splitter.split_documents(docs)

    mlflow.log_metric("total_chunks", len(chunks))
    mlflow.log_metric("chunking_time", time.time() - start_time)

    return chunks


def main():
    with mlflow.start_run(run_name="ingestion_pipeline"):
        mlflow.set_tag("stage","ingestion")

        docs = load_books()
        if not docs:
            mlflow.log_metric("pipeline_success", 0)
            return

        chunks = split_chunks(docs)
        if not chunks:
            mlflow.log_metric("pipeline_success", 0)
            return

        start_time = time.time()

        create_vectorstore(chunks)

        mlflow.log_metric("indexing_time", time.time() - start_time)
        mlflow.log_metric("pipeline_success", 1)


if __name__ == "__main__":
    main()