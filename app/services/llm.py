"""Ollama LLM and RAG chain with guardrails."""

import re
from typing import Any, AsyncIterator, List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.retriever import get_hybrid_retriever

logger = get_logger(__name__)

# Simple prompt-injection deny list (guardrail)
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+instructions",
    r"disregard\s+(previous|all)",
    r"system\s*:\s*you\s+are",
    r"you\s+are\s+now\s+(a|in)",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if",
    r"new\s+instructions?\s*:",
]


def get_llm() -> BaseChatModel:
    """Ollama chat model (free, local)."""
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.2,
        num_predict=settings.max_answer_tokens,
    )


def check_prompt_injection(text: str) -> bool:
    """Return True if likely prompt injection (should reject)."""
    lower = text.lower().strip()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True
    return False


def format_context(docs: List[Document], max_chars: int) -> str:
    """Format retrieved docs into context string, truncating to max_chars."""
    parts: List[str] = []
    total = 0
    for d in docs:
        block = f"[Source: {d.metadata.get('filename', 'unknown')}]\n{d.page_content}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts) if parts else "No relevant context found."


SYSTEM_INSTRUCTION = """You are a helpful assistant that answers questions based only on the provided context from uploaded documents.
If the answer cannot be found in the context, say so clearly. Do not make up information.
Keep answers concise and cite the source (filename) when possible."""


def build_rag_chain(collection_name: str):
    """Build RAG chain: retriever -> format context -> LLM -> string."""
    settings = get_settings()
    retriever = get_hybrid_retriever(collection_name)
    llm = get_llm()

    def run(query: str) -> str:
        if check_prompt_injection(query):
            logger.warning("Prompt injection attempt detected")
            return "I can only answer questions about the uploaded documents. Please ask a clear question about the document content."
        docs = retriever.invoke(query)
        context = format_context(docs, settings.max_context_chars)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_INSTRUCTION + "\n\nContext:\n{context}"),
            ("human", "{question}"),
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

    return run


async def stream_rag_response(
    collection_name: str,
    question: str,
) -> AsyncIterator[str]:
    """Stream RAG response chunks (async)."""
    settings = get_settings()
    if check_prompt_injection(question):
        yield "I can only answer questions about the uploaded documents."
        return

    retriever = get_hybrid_retriever(collection_name)
    llm = get_llm()
    docs = retriever.invoke(question)
    context = format_context(docs, settings.max_context_chars)

    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTION + "\n\nContext:\n" + context),
        HumanMessage(content=question),
    ]
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content
