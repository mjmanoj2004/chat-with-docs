"""Application configuration from environment."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load from .env; no hard-coded secrets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama (free LLM)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2:latest"

    # Embeddings (sentence-transformers - free)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    default_collection: str = "documents"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 5
    max_context_chars: int = 4000
    max_answer_tokens: int = 512

    # App
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)


def get_settings() -> Settings:
    """Return app settings (singleton-style usage)."""
    return Settings()
