import hashlib

import psycopg
import pytest

from app.rag.ingest import ingest_documents
from app.rag.loader import SourceDocument


def doc(path: str, text: str) -> SourceDocument:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return SourceDocument(path=path, title=path, url=None, text=text, content_hash=digest)


def count(db: psycopg.Connection, table: str) -> int:
    return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_first_ingest_writes_documents_and_chunks(db: psycopg.Connection) -> None:
    stats = ingest_documents(db, [doc("a.md", "a" * 25), doc("b.md", "b" * 5)], size=10, overlap=2)

    assert (stats.written, stats.unchanged, stats.deleted) == (2, 0, 0)
    assert stats.chunks == count(db, "chunks") == 4  # a: 0-10, 8-18, 16-25; b: 0-5
    assert count(db, "documents") == 2


def test_second_ingest_writes_nothing(db: psycopg.Connection) -> None:
    docs = [doc("a.md", "a" * 25), doc("b.md", "b" * 5)]
    ingest_documents(db, docs, size=10, overlap=2)
    chunk_ids = db.execute("SELECT array_agg(id ORDER BY id) FROM chunks").fetchone()[0]

    stats = ingest_documents(db, docs, size=10, overlap=2)

    assert (stats.written, stats.unchanged, stats.deleted, stats.chunks) == (0, 2, 0, 0)
    assert db.execute("SELECT array_agg(id ORDER BY id) FROM chunks").fetchone()[0] == chunk_ids


def test_changed_document_gets_new_chunks(db: psycopg.Connection) -> None:
    ingest_documents(db, [doc("a.md", "a" * 25), doc("b.md", "b" * 5)], size=10, overlap=2)

    stats = ingest_documents(db, [doc("a.md", "c" * 12), doc("b.md", "b" * 5)], size=10, overlap=2)

    assert (stats.written, stats.unchanged) == (1, 1)
    contents = db.execute(
        "SELECT content FROM chunks JOIN documents d ON d.id = document_id"
        " WHERE d.path = 'a.md' ORDER BY chunk_index"
    ).fetchall()
    assert [r[0] for r in contents] == ["c" * 10, "c" * 4]


def test_new_chunk_settings_rewrite_everything(db: psycopg.Connection) -> None:
    docs = [doc("a.md", "a" * 25), doc("b.md", "b" * 5)]
    ingest_documents(db, docs, size=10, overlap=2)

    stats = ingest_documents(db, docs, size=20, overlap=2)

    assert (stats.written, stats.unchanged) == (2, 0)


def test_removed_document_is_deleted_with_its_chunks(db: psycopg.Connection) -> None:
    ingest_documents(db, [doc("a.md", "a" * 25), doc("b.md", "b" * 5)], size=10, overlap=2)

    stats = ingest_documents(db, [doc("b.md", "b" * 5)], size=10, overlap=2)

    assert stats.deleted == 1
    assert count(db, "documents") == 1
    assert count(db, "chunks") == 1


def test_empty_corpus_is_rejected_and_nothing_deleted(db: psycopg.Connection) -> None:
    ingest_documents(db, [doc("a.md", "a" * 25)], size=10, overlap=2)

    with pytest.raises(ValueError):
        ingest_documents(db, [], size=10, overlap=2)
    assert count(db, "documents") == 1
