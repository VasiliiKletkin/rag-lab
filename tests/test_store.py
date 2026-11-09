"""Этап 6. Qdrant тестируется в памяти (QdrantClient(":memory:")) — без Docker;
pgvector — на тестовой базе; паритет — на обоих сразу."""

import psycopg
import pytest
from qdrant_client import QdrantClient

from app.rag.ingest import embed_missing, ingest_documents, mirror_vectors
from app.rag.retrieval import search
from app.rag.store import PgVectorStore, QdrantStore, StoredVector
from tests.test_embeddings import FakeEmbedder
from tests.test_ingest import doc
from tests.test_retrieval import LengthEmbedder


def qdrant() -> QdrantStore:
    return QdrantStore(QdrantClient(":memory:"), "chunks_test", dim=384)


def vec(x: float) -> list[float]:
    return [x, 1.0] + [0.0] * 382


def test_qdrant_search_filter_and_delete() -> None:
    store = qdrant()
    store.upsert(
        [
            StoredVector(1, "pg.md", "hnsw", vec(0.9)),
            StoredVector(2, "py.md", "stack", vec(0.5)),
            StoredVector(3, "pg.md", "ivfflat", vec(0.1)),
        ]
    )

    assert [h.chunk_id for h in store.search(vec(0.9), k=2)] == ["1", "2"]
    assert [h.chunk_id for h in store.search(vec(0.9), k=5, document="py.md")] == ["2"]
    assert store.search(vec(0.9), k=1)[0].text == "hnsw"

    store.delete_documents(["pg.md"])
    assert store.count() == 1


def test_embed_missing_mirrors_into_store(db: psycopg.Connection) -> None:
    store = qdrant()
    ingest_documents(db, [doc("a.md", "a" * 10), doc("b.md", "b" * 10)], size=50, overlap=0)

    embedded = embed_missing(db, FakeEmbedder(), store=store)

    assert embedded == store.count() == 2
    assert mirror_vectors(db, store) == 0  # базы совпадают — ничего не перезаписано


def test_rewritten_and_removed_documents_leave_the_store(db: psycopg.Connection) -> None:
    store = qdrant()
    ingest_documents(db, [doc("a.md", "a" * 10), doc("b.md", "b" * 10)], size=50, overlap=0)
    embed_missing(db, FakeEmbedder(), store=store)

    ingest_documents(db, [doc("a.md", "a" * 20)], size=50, overlap=0, store=store)

    assert store.count() == 0  # a.md перезаписан (старые векторы удалены), b.md удалён
    embed_missing(db, FakeEmbedder(), store=store)
    assert [h.document for h in store.search(vec(0.2), k=5)] == ["a.md"]


def test_mirror_catches_up_an_empty_store(db: psycopg.Connection) -> None:
    ingest_documents(db, [doc("a.md", "a" * 10)], size=50, overlap=0)
    embed_missing(db, FakeEmbedder())  # векторы только в Postgres, как до VECTOR_BACKEND=qdrant
    store = qdrant()

    assert mirror_vectors(db, store) == 1
    assert store.count() == 1


def test_both_backends_rank_the_same(db: psycopg.Connection) -> None:
    embedder = LengthEmbedder()
    ingest_documents(
        db,
        [doc("pg.md", "Supported index types are HNSW and IVFFlat. " * 3), doc("py.md", "x" * 48)],
        size=500,
        overlap=0,
    )
    store = qdrant()
    embed_missing(db, embedder, store=store)
    query = "x" * 48

    pg = search(db, embedder, query, k=2, mode="vector", store=PgVectorStore(db))
    qd = search(db, embedder, query, k=2, mode="vector", store=store)

    assert [h.chunk_id for h in pg] == [h.chunk_id for h in qd] == [h.chunk_id for h in qd]
    assert pg[0].document == "py.md"
    assert pg[0].score == pytest.approx(qd[0].score, abs=1e-5)  # косинус тот же, базы разные


def test_document_filter_in_pgvector_and_hybrid(db: psycopg.Connection) -> None:
    embedder = LengthEmbedder()
    ingest_documents(
        db, [doc("pg.md", "HNSW index " * 5), doc("py.md", "HNSW stack " * 5)], size=500, overlap=0
    )
    embed_missing(db, embedder)

    only_py = search(db, embedder, "HNSW", k=5, mode="hybrid", candidates=10, document="py.md")

    assert {h.document for h in only_py} == {"py.md"}
