from fastapi import APIRouter
from pydantic import BaseModel
import time

from app.utils.logger import get_logger
from app.llm.rag_chain import get_rag_chain

logger = get_logger(__name__)

router = APIRouter()

# ===== Initialize ONCE =====
rag_chain = get_rag_chain(mode="default", k=2)


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
    start_time = time.time()

    try:
        logger.info(f"Received query: {request.query}")

        response = rag_chain.invoke(request.query)

        latency = round(time.time() - start_time, 2)

        logger.info(f"Response generated in {latency}s")

        return {
            "answer": response,
            "latency": latency
        }

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")

        return {
            "answer": "Something went wrong. Please try again.",
            "latency": 0.0
        }


# ===== Health Check =====
@router.get("/health")
def health():
    return {"status": "ok"}