from fastapi import APIRouter
from pydantic import BaseModel
import time
import os

from app.utils.logger import get_logger
from app.llm.rag_chain import get_rag_chain

# ===== Safe MLflow Setup =====

MLFLOW_URI = os.getenv("MLFLOW_URI")

try:
    import mlflow

    if MLFLOW_URI:
        mlflow.set_tracking_uri(
            MLFLOW_URI
        )

        mlflow.set_experiment(
            "RAG pipeline"
        )

        MLFLOW_ENABLED = True

    else:
        MLFLOW_ENABLED = False

except Exception:
    MLFLOW_ENABLED = False


logger = get_logger(__name__)

router = APIRouter()


# ===== Request Schema =====

class ChatRequest(BaseModel):
    query: str
    mode: str = "default"
    k: int = 5


# ===== Response Schema =====

class ChatResponse(BaseModel):
    answer: str
    latency: float


# ===== Chat Endpoint =====

@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    start_time = time.time()

    try:

        logger.info(
            f"Received query: {request.query}"
        )

        logger.info(
            f"Mode={request.mode} | K={request.k}"
        )

        chain = get_rag_chain(
            mode=request.mode,
            k=request.k
        )

        response = chain.invoke(
            request.query
        )

        latency = round(
            time.time() - start_time,
            2
        )

        logger.info(
            f"Response generated in {latency}s"
        )

        if MLFLOW_ENABLED:

            with mlflow.start_run(
                run_name="rag_inference"
            ):

                mlflow.set_tag(
                    "stage",
                    "inference"
                )

                mlflow.log_param(
                    "mode",
                    request.mode
                )

                mlflow.log_param(
                    "retrieval_k",
                    request.k
                )

                mlflow.log_metric(
                    "latency",
                    latency
                )

                mlflow.log_metric(
                    "request_success",
                    1
                )

        return {
            "answer": response,
            "latency": latency
        }

    except Exception as e:

        logger.error(
            f"Error in /chat endpoint: {e}"
        )

        if MLFLOW_ENABLED:

            with mlflow.start_run(
                run_name="rag_inference_error"
            ):

                mlflow.log_metric(
                    "request_success",
                    0
                )

        return {
            "answer": f"Error: {str(e)}",
            "latency": 0.0
        }


# ===== Health Check =====

@router.get("/health")
def health():

    return {
        "status": "ok"
    }