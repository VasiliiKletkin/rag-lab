"""Этапы 2, 3 и 6: хранение кусков и векторов и поиск по ним."""

from typing import Protocol

from app.rag.chunking import Chunk
from app.schemas import SearchHit


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int) -> list[SearchHit]: ...


# TODO(этап 2): PgVectorStore — таблицы `documents` и `chunks` с колонкой
#   `embedding vector(N)` и индексом HNSW (vector_cosine_ops).
# TODO(этап 6): QdrantStore с тем же протоколом.
