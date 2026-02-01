"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure Chroma data dir exists."""
    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    logger.info("Chroma persist dir: %s", settings.chroma_path)
    yield
    # Shutdown: nothing to clean up
    pass


app = FastAPI(
    title="Chat With Your Docs",
    description="RAG API: upload documents, ask questions. Hybrid search (semantic + BM25), Ollama LLM.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health")
def health():
    """Health check for containers."""
    return {"status": "ok"}
