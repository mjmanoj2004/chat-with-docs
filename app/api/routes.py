"""FastAPI routes: upload, ask, collections."""

import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.api.models import (
    AskRequest,
    AskResponse,
    CollectionsResponse,
    UploadBatchResponse,
    UploadResponse,
)
from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.ingestion import ingest_file
from app.services.llm import stream_rag_response
from app.services.vectorstore import (
    add_documents_to_vectorstore,
    delete_collection,
    list_collections,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["rag"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".doc", ".docx"}


def _trace_id(request: Request) -> str:
    return request.headers.get("x-trace-id") or str(uuid.uuid4())[:8]


async def _process_one_file(
    file: UploadFile,
    collection_name: str,
) -> UploadResponse:
    """Ingest one uploaded file into Chroma; return upload result."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for {file.filename}: allowed {', '.join(ALLOWED_EXTENSIONS)}",
        )
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        chunks = ingest_file(tmp_path, file.filename or "unknown")
        if not chunks:
            raise HTTPException(status_code=422, detail=f"No text extracted from {file.filename}.")
        add_documents_to_vectorstore(collection_name, chunks)
        return UploadResponse(
            filename=file.filename or "unknown",
            collection_name=collection_name,
            chunks_added=len(chunks),
            message=f"Indexed {len(chunks)} chunks.",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/upload", response_model=UploadResponse | UploadBatchResponse)
async def upload_document(
    request: Request,
    files: list[UploadFile] = File(..., description="One or more files to index"),
):
    """Upload one or more PDF/TXT/DOC files; chunk and index into Chroma (default collection)."""
    trace_id = _trace_id(request)
    log = get_logger(__name__)
    start = time.perf_counter()
    settings = get_settings()
    collection_name = settings.default_collection
    if not files:
        raise HTTPException(status_code=400, detail="Provide at least one file.")
    uploads = files

    results: list[UploadResponse] = []
    for u in uploads:
        try:
            r = await _process_one_file(u, collection_name)
            results.append(r)
        except HTTPException:
            raise
        except Exception as e:
            log.exception("upload failed for %s", u.filename)
            raise HTTPException(status_code=500, detail=f"{u.filename}: {e}")

    elapsed = time.perf_counter() - start
    total_chunks = sum(r.chunks_added for r in results)
    log.info("upload ok | trace_id=%s | files=%d | total_chunks=%d | elapsed_ms=%.0f", trace_id, len(results), total_chunks, elapsed * 1000)

    if len(results) == 1:
        return results[0]
    return UploadBatchResponse(results=results, total_chunks=total_chunks)


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest):
    """Ask a question; return answer and source refs (non-streaming)."""
    trace_id = _trace_id(request)
    log = get_logger(__name__)
    start = time.perf_counter()
    settings = get_settings()
    collection_name = body.collection_name or settings.default_collection

    from app.services.llm import build_rag_chain
    from app.services.retriever import get_hybrid_retriever

    try:
        retriever = get_hybrid_retriever(collection_name)
        docs = retriever.invoke(body.question)
        chain = build_rag_chain(collection_name)
        answer = chain(body.question)
        sources = [
            {"filename": d.metadata.get("filename"), "page": d.metadata.get("page"), "snippet": d.page_content[:200]}
            for d in docs
        ]
        elapsed = time.perf_counter() - start
        log.info("ask ok | trace_id=%s | collection=%s | elapsed_ms=%.0f", trace_id, collection_name, elapsed * 1000)
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        log.exception("ask failed | trace_id=%s", trace_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ask/stream")
async def ask_stream(
    request: Request,
    question: str,
    collection_name: str | None = None,
):
    """Stream answer tokens (SSE-style)."""
    if not question or len(question) > 2000:
        raise HTTPException(status_code=400, detail="Invalid question length.")
    settings = get_settings()
    col = collection_name or settings.default_collection

    async def generate():
        async for chunk in stream_rag_response(col, question):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/plain",
    )


@router.get("/collections", response_model=CollectionsResponse)
async def get_collections():
    """List Chroma collection names."""
    try:
        names = list_collections()
        return CollectionsResponse(collections=names)
    except Exception as e:
        logger.exception("list collections failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{name}")
async def delete_collection_route(name: str):
    """Delete a Chroma collection (reset)."""
    try:
        delete_collection(name)
        return {"status": "deleted", "collection": name}
    except Exception as e:
        logger.exception("delete collection failed")
        raise HTTPException(status_code=500, detail=str(e))
