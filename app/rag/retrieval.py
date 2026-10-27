"""Stage 3: vector, hybrid and reranked retrieval."""

from app.schemas import SearchHit


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge several ranked lists of chunk ids into one, best first.

    TODO(stage 3): score(id) = sum(1 / (k + rank)) over every list that contains the id.
    """
    raise NotImplementedError


def search(query: str, k: int, mode: str) -> list[SearchHit]:
    """TODO(stage 3): "vector" -> cosine over pgvector; "hybrid" -> Postgres full-text
    search + vector merged with RRF; "rerank" -> hybrid top-50 re-scored by a cross-encoder."""
    raise NotImplementedError
