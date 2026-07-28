from fastapi import FastAPI
from app.api.routes import router
from app.api_v2.upload_routes import router as upload_router
from app.api_v2.document_routes import router as document_router
from app.api_v2.chat_stream_routes import router as chat_stream_router
from app.api_v2.observability_routes import router as observability_router

app = FastAPI(
    title="RAGnosis API",
    version="1.0"
)

app.include_router(router)
app.include_router(upload_router)
app.include_router(document_router)
app.include_router(chat_stream_router)
app.include_router(observability_router)