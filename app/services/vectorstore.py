"""ChromaDB vector store and BM25 index for hybrid search."""

from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def get_embeddings() -> Embeddings:
    """Sentence-transformers embeddings (free, no API key)."""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
    )


def get_chroma_client() -> chromadb.PersistentClient:
    """Persistent Chroma client."""
    settings = get_settings()
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_vectorstore(
    collection_name: str,
    embeddings: Optional[Embeddings] = None,
) -> Chroma:
    """Get or create a LangChain Chroma vectorstore."""
    client = get_chroma_client()
    emb = embeddings or get_embeddings()
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=emb,
    )


def build_bm25_index(documents: List[Document]) -> BM25Okapi:
    """Build BM25 index from document texts (tokenized by whitespace)."""
    tokenized = [doc.page_content.lower().split() for doc in documents]
    return BM25Okapi(tokenized)


def add_documents_to_vectorstore(
    collection_name: str,
    documents: List[Document],
    embeddings: Optional[Embeddings] = None,
) -> None:
    """Add chunked documents to Chroma."""
    if not documents:
        return
    vs = get_vectorstore(collection_name, embeddings)
    vs.add_documents(documents)
    logger.info("Added %d chunks to collection %s", len(documents), collection_name)


def list_collections() -> List[str]:
    """List all Chroma collection names."""
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]


def delete_collection(name: str) -> None:
    """Delete a Chroma collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(name)
        logger.info("Deleted collection %s", name)
    except Exception as e:
        logger.warning("Delete collection %s: %s", name, e)
        raise


def get_collection_documents(collection_name: str) -> List[Document]:
    """Fetch all documents from a Chroma collection for BM25 indexing."""
    client = get_chroma_client()
    try:
        coll = client.get_collection(name=collection_name)
    except Exception:
        return []
    data = coll.get(include=["documents", "metadatas"])
    docs: List[Document] = []
    for i, content in enumerate(data["documents"] or []):
        meta = (data["metadatas"] or [{}])[i] if data["metadatas"] else {}
        docs.append(Document(page_content=content or "", metadata=dict(meta or {})))
    return docs
