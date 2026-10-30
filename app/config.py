from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения из переменных окружения и `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # Этап 2: эмбеддинги. Модель мультиязычная: вопросы на русском, документы на английском.
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    # Этап 1: корпус и нарезка на куски
    data_dir: Path = Path("data")
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Этап 3: поиск. Сколько кандидатов берём из каждого канала до слияния и переранжирования.
    top_k: int = 5
    candidates: int = 30
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Этап 4: генерация
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"

    # Этап 6: вторая векторная база
    vector_backend: str = "pgvector"  # "pgvector" | "qdrant"
    qdrant_url: str = "http://localhost:6333"


@lru_cache
def get_settings() -> Settings:
    return Settings()
