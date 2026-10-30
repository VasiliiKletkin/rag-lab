"""Метрики качества поиска. Чистые функции, без базы и моделей."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expected_documents: list[str]
    must_contain: (
        str | None
    )  # фраза, которая есть в «правильном» куске; None = ответа в корпусе нет
    reference_answer: str


def gold_rank(question: Question, hits: list[tuple[str, str]]) -> int | None:
    """Место (с 1) первого куска, который из нужного документа и содержит золотую фразу.

    `hits` — список (документ, текст) в порядке выдачи.
    """
    if question.must_contain is None:
        return None
    for position, (document, text) in enumerate(hits, start=1):
        if document in question.expected_documents and question.must_contain in text:
            return position
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Доля вопросов, у которых правильный кусок попал в первые k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank: среднее 1/место; не нашли — 0."""
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None) / len(ranks)
