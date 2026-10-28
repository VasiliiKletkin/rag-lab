"""Этап 1: идемпотентная загрузка документов и их кусков в Postgres."""

from dataclasses import dataclass

import psycopg

from app.rag.chunking import split_text
from app.rag.loader import SourceDocument

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
) -> IngestStats:
    """Синхронизировать базу с корпусом: записать новое и изменённое, удалить пропавшее.

    Каждый документ пишется в своей транзакции, поэтому падение посередине не оставит
    документ с половиной кусков.
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
    return stats
