"""Этап 3: векторный, лексический, гибридный поиск и переранжирование."""

from functools import lru_cache
from typing import Literal

import psycopg

from app.config import get_settings
from app.rag.embeddings import Embedder
from app.rag.store import PgVectorStore, VectorStore, get_store
from app.schemas import SearchHit
from app.telemetry import tracer

SearchMode = Literal["vector", "hybrid", "rerank"]

# websearch_to_tsquery соединяет слова через И: русский вопрос к английскому тексту даст 0 строк
LEXICAL_SQL = """
SELECT c.id, d.path, c.content, ts_rank_cd(c.tsv, q) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id,
     websearch_to_tsquery('english', %(q)s) q
WHERE c.tsv @@ q AND (%(doc)s::text IS NULL OR d.path = %(doc)s)
ORDER BY score DESC
LIMIT %(k)s
"""


def _hits(rows: list[tuple]) -> list[SearchHit]:
    return [SearchHit(chunk_id=str(i), document=p, text=t, score=float(s)) for i, p, t, s in rows]


def vector_search(
    conn: psycopg.Connection, query_vector: list[float], k: int, document: str | None = None
) -> list[SearchHit]:
    return PgVectorStore(conn).search(query_vector, k, document)


def lexical_search(
    conn: psycopg.Connection, query: str, k: int, document: str | None = None
) -> list[SearchHit]:
    return _hits(conn.execute(LEXICAL_SQL, {"q": query, "k": k, "doc": document}).fetchall())


def reciprocal_rank_fusion(rankings: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    """Слить несколько ранжированных списков в один: score = сумма 1 / (k + место).

    Складываем места, а не оценки: косинус (0..1) и ts_rank_cd (0..20+) несравнимы.
    Выигрывает кусок, который нашли оба канала.
    """
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (k + position)
            hits.setdefault(hit.chunk_id, hit)
    return [
        hits[chunk_id].model_copy(update={"score": score})
        for chunk_id, score in sorted(scores.items(), key=lambda item: -item[1])
    ]


class Reranker:
    """Cross-encoder читает вопрос и кусок вместе, поэтому точнее bi-encoder'а,
    но считается заново на каждую пару — только для 30–50 кандидатов, не для всей базы."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.text) for h in hits])
        rescored = [
            h.model_copy(update={"score": float(s)}) for h, s in zip(hits, scores, strict=True)
        ]
        return sorted(rescored, key=lambda h: -h.score)


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(get_settings().reranker_model)


def search(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    k: int,
    mode: SearchMode,
    candidates: int | None = None,
    reranker: Reranker | None = None,
    store: VectorStore | None = None,
    document: str | None = None,
) -> list[SearchHit]:
    """vector — косинус в векторной базе (pgvector или Qdrant, см. VECTOR_BACKEND);
    hybrid — вектор + полнотекстовый поиск Postgres через RRF;
    rerank — гибридные кандидаты, пересортированные cross-encoder'ом.
    document — искать только в одном документе (по пути)."""
    candidates = candidates or get_settings().candidates
    store = store or get_store(conn)
    with tracer.start_as_current_span("retrieve") as span:
        span.set_attributes({"rag.mode": mode, "rag.k": k, "rag.store": store.name})
        with tracer.start_as_current_span("embed_query"):
            query_vector = embedder.embed_query(query)
        if mode == "vector":
            hits = store.search(query_vector, k, document)
            span.set_attribute("rag.hits", len(hits))
            return hits
        fused = reciprocal_rank_fusion(
            [
                store.search(query_vector, candidates, document),
                lexical_search(conn, query, candidates, document),
            ]
        )
        span.set_attribute("rag.candidates", len(fused))
    if mode == "hybrid":
        return fused[:k]
    reranker = reranker or get_reranker()
    with tracer.start_as_current_span("rerank") as span:
        span.set_attribute("rag.candidates", min(len(fused), candidates))
        return reranker.rerank(query, fused[:candidates])[:k]
