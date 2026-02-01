"""Hybrid retriever: semantic (Chroma) + BM25, merged with RRF; returns top 5."""

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from app.services.vectorstore import (
    build_bm25_index,
    get_collection_documents,
    get_vectorstore,
)

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# RRF constant (Reciprocal Rank Fusion)
RRF_K = 60


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = RRF_K,
) -> List[Document]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    Each document's score = sum(1 / (k + rank)); dedupe by content, take top 5.
    """
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}

    for rank, doc in enumerate(ranked_lists[0] if ranked_lists else []):
        key = (doc.page_content[:200], doc.metadata.get("filename", ""))
        key_str = str(key)
        scores[key_str] = scores.get(key_str, 0) + 1 / (k + rank + 1)
        doc_by_key[key_str] = doc

    for lst in ranked_lists[1:]:
        for rank, doc in enumerate(lst):
            key = (doc.page_content[:200], doc.metadata.get("filename", ""))
            key_str = str(key)
            scores[key_str] = scores.get(key_str, 0) + 1 / (k + rank + 1)
            if key_str not in doc_by_key:
                doc_by_key[key_str] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: -scores[x])
    top_k = get_settings().top_k
    return [doc_by_key[key] for key in sorted_keys[:top_k]]


class HybridRetriever(BaseRetriever):
    """Retriever that combines Chroma (semantic) and BM25 (keyword); returns top 5."""

    vectorstore: VectorStore
    collection_name: str
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        # Semantic search
        semantic_docs = self.vectorstore.similarity_search(query, k=self.top_k)

        # BM25 search: build index from collection and run BM25
        all_docs = get_collection_documents(self.collection_name)
        if not all_docs:
            return semantic_docs[: self.top_k]

        bm25 = build_bm25_index(all_docs)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: -bm25_scores[i],
        )[: self.top_k]
        bm25_docs = [all_docs[i] for i in top_indices if bm25_scores[i] > 0]

        if not bm25_docs:
            return semantic_docs[: self.top_k]

        # RRF merge
        merged = _reciprocal_rank_fusion([semantic_docs, bm25_docs])
        logger.debug("Hybrid retrieval: %d semantic, %d BM25, %d merged", len(semantic_docs), len(bm25_docs), len(merged))
        return merged[: self.top_k]


def get_hybrid_retriever(collection_name: str) -> HybridRetriever:
    """Build hybrid retriever for the given collection."""
    vs = get_vectorstore(collection_name)
    top_k = get_settings().top_k
    return HybridRetriever(
        vectorstore=vs,
        collection_name=collection_name,
        top_k=top_k,
    )
