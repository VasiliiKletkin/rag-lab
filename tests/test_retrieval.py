import psycopg
import pytest

from app.rag.ingest import embed_missing, ingest_documents
from app.rag.retrieval import lexical_search, reciprocal_rank_fusion, search, vector_search
from app.schemas import SearchHit
from tests.test_embeddings import FakeEmbedder
from tests.test_ingest import doc


def hit(chunk_id: str, score: float = 0.0) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, document="d", text=chunk_id, score=score)


def test_rrf_prefers_chunk_found_by_both_channels() -> None:
    vector = [hit("a"), hit("b"), hit("c")]
    lexical = [hit("c"), hit("d")]

    fused = reciprocal_rank_fusion([vector, lexical], k=60)

    assert [h.chunk_id for h in fused] == ["c", "a", "b", "d"]
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_empty_channel_is_fine() -> None:
    assert [h.chunk_id for h in reciprocal_rank_fusion([[hit("a")], []])] == ["a"]


class LengthEmbedder(FakeEmbedder):
    """Вектор = длина текста в первой координате: похожесть — близость длин."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[len(t) / 100, 1.0] + [0.0] * 382 for t in texts]


@pytest.fixture
def corpus(db: psycopg.Connection) -> psycopg.Connection:
    ingest_documents(
        db,
        [
            doc("pg.md", "Supported index types are HNSW and IVFFlat. " * 3),
            doc("py.md", "Use append to push an item on top of the stack."),
        ],
        size=500,
        overlap=0,
    )
    embed_missing(db, LengthEmbedder())
    return db


def test_vector_search_orders_by_cosine(corpus: psycopg.Connection) -> None:
    hits = vector_search(corpus, LengthEmbedder().embed_query("x" * 48), k=2)

    assert [h.document for h in hits] == ["py.md", "pg.md"]
    assert hits[0].score > hits[1].score


def test_lexical_search_matches_words(corpus: psycopg.Connection) -> None:
    hits = lexical_search(corpus, "HNSW index", k=5)

    assert [h.document for h in hits] == ["pg.md"]
    assert lexical_search(corpus, "какие индексы", k=5) == []  # русских слов в корпусе нет


def test_hybrid_mode_merges_both_channels(corpus: psycopg.Connection) -> None:
    hits = search(corpus, LengthEmbedder(), "HNSW index", k=2, mode="hybrid", candidates=10)

    assert hits[0].document == "pg.md"  # нашли оба канала
    assert len(hits) == 2


class FakeReranker:
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        return sorted(hits, key=lambda h: query in h.text, reverse=True)


def test_rerank_mode_uses_reranker(corpus: psycopg.Connection) -> None:
    hits = search(
        corpus,
        LengthEmbedder(),
        "stack",
        k=1,
        mode="rerank",
        candidates=10,
        reranker=FakeReranker(),
    )

    assert hits[0].document == "py.md"
