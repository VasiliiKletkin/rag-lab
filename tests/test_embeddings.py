import psycopg
import pytest

from app.rag.embeddings import LocalEmbedder, get_embedder
from app.rag.ingest import embed_missing, ingest_documents
from tests.test_ingest import doc


class FakeEmbedder:
    """Детерминированные векторы без модели: тесты загрузки не должны качать веса."""

    name = "fake"
    dim = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] + [0.0] * 383 for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def test_embed_only_missing_chunks(db: psycopg.Connection) -> None:
    ingest_documents(db, [doc("a.md", "a" * 25)], size=10, overlap=2)

    assert embed_missing(db, FakeEmbedder()) == 3
    assert embed_missing(db, FakeEmbedder()) == 0
    assert db.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0] == 0


def test_changed_model_reembeds_everything(db: psycopg.Connection) -> None:
    ingest_documents(db, [doc("a.md", "a" * 25)], size=10, overlap=2)
    embed_missing(db, FakeEmbedder())
    other = FakeEmbedder()
    other.name = "other-model"

    assert embed_missing(db, other) == 3
    models = db.execute("SELECT DISTINCT embedding_model FROM chunks").fetchall()
    assert models == [("other-model",)]


def test_rewritten_document_loses_old_vectors(db: psycopg.Connection) -> None:
    ingest_documents(db, [doc("a.md", "a" * 25)], size=10, overlap=2)
    embed_missing(db, FakeEmbedder())

    ingest_documents(db, [doc("a.md", "b" * 25)], size=10, overlap=2)

    assert db.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0] == 3


@pytest.mark.slow
def test_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        LocalEmbedder("intfloat/multilingual-e5-small", expected_dim=768)


@pytest.mark.slow
def test_real_model_cross_language() -> None:
    embedder = get_embedder()
    query = embedder.embed_query("Какие типы индексов поддерживает pgvector?")
    good, bad = embedder.embed_documents(
        ["Supported index types are HNSW and IVFFlat.", "Use a list as a stack with append()."]
    )

    assert len(query) == 384
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert dot(query, good) > dot(query, bad)
