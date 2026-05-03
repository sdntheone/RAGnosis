from fastapi import APIRouter
from pydantic import BaseModel
import time
import mlflow

from app.utils.logger import get_logger
from app.llm.rag_chain import get_rag_chain
import os
mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "http://localhost:5000"))
mlflow.set_experiment("RAG pipeline")

logger = get_logger(__name__)

router = APIRouter()

# ===== Initialize ONCE =====
mode="default"
retrieval_k=2
rag_chain = get_rag_chain(mode=mode, k=retrieval_k)



# ===== Request Schema =====
class ChatRequest(BaseModel):
    query: str


# ===== Response Schema =====
class ChatResponse(BaseModel):
    answer: str
    latency: float


# ===== Chat Endpoint =====
@router.post("/chat", response_model=ChatResponse)

def chat(request: ChatRequest):
    with mlflow.start_run(run_name="rag_inference"):
        mlflow.set_tag("stage", "inference")
        mlflow.log_param("mode",mode)
        mlflow.log_param("retrieval_k",retrieval_k)
        start_time = time.time()

        try:
            logger.info(f"Received query: {request.query}")

            response = rag_chain.invoke(request.query)

            latency = round(time.time() - start_time, 2)

            logger.info(f"Response generated in {latency}s")
            mlflow.log_metric("latency",latency)
            mlflow.log_metric("request_success",1)


            return {
                "answer": response,
                "latency": latency
            }

            

        except Exception as e:
            logger.error(f"Error in /chat endpoint: {e}")
            mlflow.log_metric("request_success",0)

            return {
                "answer": "Something went wrong. Please try again.",
                "latency": 0.0
            }


# ===== Health Check =====
@router.get("/health")
def health():
    return {"status": "ok"}