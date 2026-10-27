from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # Stage 2: embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Stage 1: chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Stage 3: retrieval
    top_k: int = 5

    # Stage 4: generation
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"

    # Stage 6: second vector store
    vector_backend: str = "pgvector"  # "pgvector" | "qdrant"
    qdrant_url: str = "http://localhost:6333"


@lru_cache
def get_settings() -> Settings:
    return Settings()
