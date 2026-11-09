"""Этап 6: две векторные базы за одним протоколом.

Postgres остаётся источником правды: текст кусков, embedding_model и сами векторы живут
в таблице chunks. Qdrant — зеркало векторов с payload (документ, текст), чтобы поиск
возвращал готовые SearchHit без второго похода в Postgres. Переключение — VECTOR_BACKEND.
"""

import warnings
from dataclasses import dataclass
from typing import Protocol

import psycopg

from app.config import get_settings
from app.schemas import SearchHit


@dataclass(frozen=True)
class StoredVector:
    chunk_id: int
    document: str
    text: str
    vector: list[float]


class VectorStore(Protocol):
    name: str

    def upsert(self, items: list[StoredVector]) -> None: ...

    def delete_documents(self, paths: list[str]) -> None: ...

    def count(self) -> int: ...

    def search(self, vector: list[float], k: int, document: str | None = None) -> list[SearchHit]:
        """Ближайшие k кусков; document — путь, если искать только в одном документе."""
        ...


# <=> — косинусное расстояние; 1 - расстояние = похожесть, оценка читается «чем больше, тем лучше».
# %(doc)s IS NULL в WHERE: без фильтра условие всегда истинно и планировщик его выбрасывает.
PGVECTOR_SQL = """
SELECT c.id, d.path, c.content, 1 - (c.embedding <=> %(q)s::vector) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.embedding IS NOT NULL AND (%(doc)s::text IS NULL OR d.path = %(doc)s)
ORDER BY c.embedding <=> %(q)s::vector
LIMIT %(k)s
"""


class PgVectorStore:
    """Векторы уже лежат в chunks.embedding (их пишет embed_missing), поэтому
    upsert и delete здесь ничего не делают: удаление идёт через ON DELETE CASCADE."""

    name = "pgvector"

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def upsert(self, items: list[StoredVector]) -> None:
        return None

    def delete_documents(self, paths: list[str]) -> None:
        return None

    def count(self) -> int:
        return self._conn.execute(
            "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]

    def search(self, vector: list[float], k: int, document: str | None = None) -> list[SearchHit]:
        rows = self._conn.execute(PGVECTOR_SQL, {"q": vector, "k": k, "doc": document}).fetchall()
        return [
            SearchHit(chunk_id=str(i), document=p, text=t, score=float(s)) for i, p, t, s in rows
        ]


class QdrantStore:
    """Коллекция на модель: сменили модель — сменилась коллекция, старые векторы не смешаются."""

    name = "qdrant"

    def __init__(self, client, collection: str, dim: int) -> None:
        from qdrant_client import models

        self._client = client
        self._collection = collection
        self._models = models
        if not client.collection_exists(collection):
            client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            # индекс по payload: фильтр по документу иначе перебирает все точки
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # в :memory:-режиме индексы не действуют
                client.create_payload_index(
                    collection, "document", field_schema=models.PayloadSchemaType.KEYWORD
                )

    def _by_document(self, paths: list[str]):
        m = self._models
        return m.Filter(must=[m.FieldCondition(key="document", match=m.MatchAny(any=paths))])

    def upsert(self, items: list[StoredVector]) -> None:
        if not items:
            return
        self._client.upsert(
            self._collection,
            points=[
                self._models.PointStruct(
                    id=it.chunk_id,
                    vector=it.vector,
                    payload={"document": it.document, "text": it.text},
                )
                for it in items
            ],
        )

    def delete_documents(self, paths: list[str]) -> None:
        if paths:
            self._client.delete(
                self._collection,
                points_selector=self._models.FilterSelector(filter=self._by_document(paths)),
            )

    def count(self) -> int:
        return self._client.count(self._collection).count

    def search(self, vector: list[float], k: int, document: str | None = None) -> list[SearchHit]:
        result = self._client.query_points(
            self._collection,
            query=vector,
            limit=k,
            query_filter=self._by_document([document]) if document else None,
        )
        return [
            SearchHit(
                chunk_id=str(p.id),
                document=p.payload["document"],
                text=p.payload["text"],
                score=float(p.score),
            )
            for p in result.points
        ]


def get_store(conn: psycopg.Connection) -> VectorStore:
    settings = get_settings()
    if settings.vector_backend == "pgvector":
        return PgVectorStore(conn)
    if settings.vector_backend == "qdrant":
        from qdrant_client import QdrantClient

        collection = "chunks_" + settings.embedding_model.replace("/", "_").replace("-", "_")
        return QdrantStore(
            QdrantClient(url=settings.qdrant_url), collection, settings.embedding_dim
        )
    raise ValueError(f"неизвестный VECTOR_BACKEND: {settings.vector_backend}")
