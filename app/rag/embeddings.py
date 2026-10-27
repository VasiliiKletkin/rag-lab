"""Этап 2: превращение текста в векторы."""

from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# TODO(этап 2): LocalEmbedder на sentence-transformers (settings.embedding_model).
#   У e5 к кускам документов добавляется префикс "passage: ", к вопросам — "query: ".
#   По желанию: эмбеддер через API (например, Voyage AI) за тем же протоколом.
