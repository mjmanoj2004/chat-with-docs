"""Pydantic models for API request/response."""

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=1, max_length=2000)
    collection_name: Optional[str] = None


class AskResponse(BaseModel):
    """Response for POST /ask (non-streaming)."""

    answer: str
    sources: List[dict] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Response for a single file in POST /upload."""

    filename: str
    collection_name: str
    chunks_added: int
    message: str


class UploadBatchResponse(BaseModel):
    """Response for POST /upload when multiple files are sent."""

    results: list[UploadResponse]
    total_chunks: int


class CollectionsResponse(BaseModel):
    """Response for GET /collections."""

    collections: List[str]
