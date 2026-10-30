import pytest

from evals.metrics import Question, gold_rank, mrr, recall_at_k
from evals.run import load_questions


def q(must_contain: str | None = "needle", docs: list[str] | None = None) -> Question:
    return Question("q", "?", docs if docs is not None else ["a.md"], must_contain, "")


def test_gold_rank_requires_document_and_phrase() -> None:
    hits = [("b.md", "needle here"), ("a.md", "nothing"), ("a.md", "the needle")]

    assert gold_rank(q(), hits) == 3


def test_gold_rank_none_when_missing_or_unanswerable() -> None:
    assert gold_rank(q(), [("a.md", "nothing")]) is None
    assert gold_rank(q(must_contain=None, docs=[]), [("a.md", "needle")]) is None


def test_recall_and_mrr() -> None:
    ranks = [1, 1, 3, 6, None]

    assert recall_at_k(ranks, 5) == pytest.approx(3 / 5)
    assert mrr(ranks) == pytest.approx((1 + 1 + 1 / 3 + 1 / 6) / 5)
    assert recall_at_k([], 5) == 0.0


def test_question_set_is_consistent() -> None:
    """Каждая золотая фраза должна быть в своём документе, иначе вопрос нечестный."""
    from pathlib import Path

    questions = load_questions()
    ids = [x.id for x in questions]
    assert len(ids) == len(set(ids))
    assert sum(1 for x in questions if x.must_contain) >= 30
    for x in questions:
        if x.must_contain is None:
            assert x.expected_documents == []
            continue
        texts = [Path("data", d).read_text(encoding="utf-8") for d in x.expected_documents]
        assert any(x.must_contain in t for t in texts), x.id
