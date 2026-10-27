"""Этап 4: ответ на вопрос по найденным кускам с помощью Claude."""

from app.schemas import AskResponse, SearchHit


def answer(question: str, hits: list[SearchHit]) -> AskResponse:
    """TODO(этап 4): передать каждый кусок как content block типа `document`
    с включёнными цитатами, модель — settings.llm_model; сопоставить полученные
    цитаты с id кусков. Отвечать на языке вопроса; если в контексте нет ответа,
    честно сказать «не знаю»."""
    raise NotImplementedError
