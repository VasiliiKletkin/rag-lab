"""Stages 2, 3 and 6: persist chunks + vectors and query them."""

from typing import Protocol

from app.rag.chunking import Chunk
from app.schemas import SearchHit


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int) -> list[SearchHit]: ...


# TODO(stage 2): PgVectorStore - tables `documents` and `chunks` with `embedding vector(N)`
#   and an HNSW index (vector_cosine_ops).
# TODO(stage 6): QdrantStore implementing the same protocol.
