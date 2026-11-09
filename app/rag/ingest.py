"""Этап 1: идемпотентная загрузка документов и их кусков в Postgres."""

from dataclasses import dataclass

import psycopg

from app.rag.chunking import split_text
from app.rag.embeddings import Embedder
from app.rag.loader import SourceDocument
from app.rag.store import StoredVector, VectorStore

# Вставить новый документ или обновить изменившийся. Если хеш и настройки нарезки
# совпадают, WHERE не пропускает UPDATE и RETURNING возвращает ноль строк.
UPSERT_DOCUMENT = """
INSERT INTO documents (path, title, url, content_hash, chunker)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (path) DO UPDATE
SET title = EXCLUDED.title, url = EXCLUDED.url,
    content_hash = EXCLUDED.content_hash, chunker = EXCLUDED.chunker,
    ingested_at = now()
WHERE documents.content_hash IS DISTINCT FROM EXCLUDED.content_hash
   OR documents.chunker IS DISTINCT FROM EXCLUDED.chunker
RETURNING id
"""

INSERT_CHUNK = """
INSERT INTO chunks (document_id, chunk_index, content, char_start, char_end)
VALUES (%s, %s, %s, %s, %s)
"""


@dataclass
class IngestStats:
    written: int = 0  # новые и изменённые документы
    unchanged: int = 0
    deleted: int = 0
    chunks: int = 0  # сколько кусков записано за этот запуск


def ingest_documents(
    conn: psycopg.Connection,
    documents: list[SourceDocument],
    size: int,
    overlap: int,
    store: VectorStore | None = None,
) -> IngestStats:
    """Синхронизировать базу с корпусом: записать новое и изменённое, удалить пропавшее.

    Каждый документ пишется в своей транзакции, поэтому падение посередине не оставит
    документ с половиной кусков. store (этап 6) — внешняя векторная база, из которой
    надо убрать векторы перезаписанных и удалённых документов; для pgvector это no-op.
    """
    if not documents:
        # пустой manifest — скорее ошибка, чем желание удалить весь корпус
        raise ValueError("нет документов для загрузки")

    chunker = f"fixed-{size}-{overlap}"
    stats = IngestStats()
    for doc in documents:
        with conn.transaction():
            row = conn.execute(
                UPSERT_DOCUMENT, (doc.path, doc.title, doc.url, doc.content_hash, chunker)
            ).fetchone()
            if row is None:
                stats.unchanged += 1
                continue
            document_id = row[0]
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            if store:
                store.delete_documents([doc.path])
            chunks = split_text(doc.path, doc.text, size, overlap)
            with conn.cursor() as cur:
                cur.executemany(
                    INSERT_CHUNK, [(document_id, c.index, c.text, c.start, c.end) for c in chunks]
                )
            stats.written += 1
            stats.chunks += len(chunks)

    with conn.transaction():
        # куски удалённых документов уходят сами через ON DELETE CASCADE
        deleted = conn.execute(
            "DELETE FROM documents WHERE NOT (path = ANY(%s)) RETURNING path",
            ([doc.path for doc in documents],),
        ).fetchall()
    stats.deleted = len(deleted)
    if store and deleted:
        store.delete_documents([path for (path,) in deleted])
    return stats


SELECT_UNEMBEDDED = """
SELECT c.id, d.path, c.content FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.embedding IS NULL OR c.embedding_model IS DISTINCT FROM %s
ORDER BY c.id
"""

UPDATE_EMBEDDING = "UPDATE chunks SET embedding = %s, embedding_model = %s WHERE id = %s"


def embed_missing(
    conn: psycopg.Connection,
    embedder: Embedder,
    batch_size: int = 64,
    store: VectorStore | None = None,
) -> int:
    """Посчитать векторы кускам, у которых их нет или они от другой модели.

    Пачка кусков считается и пишется в одной транзакции: после падения посередине
    уже посчитанные пачки остаются, повторный запуск продолжит с оставшихся.
    Сначала Postgres, потом store: если упадём между ними, mirror_vectors догонит.
    """
    rows = conn.execute(SELECT_UNEMBEDDED, (embedder.name,)).fetchall()
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        vectors = embedder.embed_documents([content for _, _, content in batch])
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                UPDATE_EMBEDDING,
                [
                    (vec, embedder.name, chunk_id)
                    for (chunk_id, _, _), vec in zip(batch, vectors, strict=True)
                ],
            )
        if store:
            store.upsert(
                [
                    StoredVector(chunk_id, path, content, list(vec))
                    for (chunk_id, path, content), vec in zip(batch, vectors, strict=True)
                ]
            )
    return len(rows)


SELECT_EMBEDDED = """
SELECT c.id, d.path, c.content, c.embedding FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.embedding IS NOT NULL ORDER BY c.id
"""


def mirror_vectors(conn: psycopg.Connection, store: VectorStore, batch_size: int = 256) -> int:
    """Довести внешнюю базу до состояния Postgres, если число векторов разошлось
    (первый запуск с VECTOR_BACKEND=qdrant, падение между записями). Возвращает, сколько
    точек перезаписано; 0 — базы совпадали."""
    total = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    if store.count() == total:
        return 0
    rows = conn.execute(SELECT_EMBEDDED).fetchall()
    for i in range(0, len(rows), batch_size):
        store.upsert(
            [
                StoredVector(chunk_id, path, content, [float(x) for x in vec])
                for chunk_id, path, content, vec in rows[i : i + batch_size]
            ]
        )
    return len(rows)
