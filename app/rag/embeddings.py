"""Этап 2: превращение текста в векторы."""

from functools import lru_cache
from typing import Protocol

from app.config import get_settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """sentence-transformers на CPU. Модели семейства e5 обучены с префиксами:
    `passage: ` для кусков документов и `query: ` для вопросов, без них качество падает."""

    def __init__(self, model_name: str, expected_dim: int) -> None:
        # импорт здесь: torch грузится секунды, а /health он не нужен
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        # Этап 10: один и тот же вопрос не считается дважды. Кэш на экземпляре, а не на методе
        # (lru_cache на методе держал бы self и не освобождал модель). Кортеж — чтобы не мутировали.
        self._cached_query = lru_cache(maxsize=2048)(self._encode_query)
        self.dim = self._model.get_sentence_embedding_dimension()
        if self.dim != expected_dim:
            raise ValueError(
                f"модель {model_name} даёт {self.dim} измерений, в настройках {expected_dim}"
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            ["passage: " + t for t in texts], batch_size=32, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return list(self._cached_query(text))

    def _encode_query(self, text: str) -> tuple[float, ...]:
        return tuple(self._model.encode("query: " + text, normalize_embeddings=True).tolist())


@lru_cache
def get_embedder() -> Embedder:
    """Одна модель на процесс: загрузка весов занимает секунды и сотни МБ памяти."""
    settings = get_settings()
    return LocalEmbedder(settings.embedding_model, settings.embedding_dim)
