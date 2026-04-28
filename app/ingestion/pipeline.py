from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from app.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")


def load_books():
    all_docs = []

    try:
        files = os.listdir(DATA_PATH)
    except Exception as e:
        logger.error(f"Failed to access data path: {DATA_PATH} | Error: {e}")
        return []

    for file in files:
        file_path = os.path.join(DATA_PATH, file)

        if os.path.isfile(file_path) and file.endswith(".pdf"):
            logger.info(f"Loading file: {file}")

            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = file
                    doc.metadata["type"] = "book"

                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {file}")

            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def split_chunks(docs):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = splitter.split_documents(docs)
        logger.info(f"Total chunks created: {len(chunks)}")

        return chunks

    except Exception as e:
        logger.error(f"Error during chunking: {e}")
        return []


def main():
    logger.info("Starting ingestion pipeline")

    docs = load_books()
    if not docs:
        logger.warning("No documents loaded. Exiting pipeline.")
        return

    chunks = split_chunks(docs)
    if not chunks:
        logger.warning("Chunking failed. Exiting pipeline.")
        return

    logger.info("Ingestion pipeline completed successfully")


if __name__ == "__main__":
    main()