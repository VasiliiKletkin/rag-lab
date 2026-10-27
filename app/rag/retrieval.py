"""Этап 3: векторный, гибридный поиск и поиск с переранжированием."""

from app.schemas import SearchHit


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Слить несколько ранжированных списков id кусков в один, лучшие первыми.

    TODO(этап 3): score(id) = сумма 1 / (k + rank) по всем спискам, где есть этот id.
    """
    raise NotImplementedError


def search(query: str, k: int, mode: str) -> list[SearchHit]:
    """TODO(этап 3): "vector" — косинус в pgvector; "hybrid" — полнотекстовый поиск
    Postgres + векторный, слитые через RRF; "rerank" — top-50 гибридного поиска,
    пересортированные мультиязычным cross-encoder."""
    raise NotImplementedError
