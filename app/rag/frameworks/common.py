from app.rag.generation import NO_ANSWER
from app.schemas import AskResponse, Citation, SearchHit

SYSTEM = (
    "Ты отвечаешь на вопросы строго по приложенным документам. Отвечай на языке вопроса, "
    f"коротко и по делу. Если в документах нет ответа, ответь ровно: «{NO_ANSWER}»"
)


def format_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(f"[{i}] {h.document}\n{h.text}" for i, h in enumerate(hits, start=1))


def response_from(answer: str, hits: list[SearchHit]) -> AskResponse:
    """У фреймворков нет цитат уровня символов, как у Claude API: источники — это куски,
    которые ушли в контекст, цитата — начало куска."""
    answer = answer.strip()
    if not hits or NO_ANSWER in answer:
        return AskResponse(answer=NO_ANSWER, citations=[])
    citations = [
        Citation(document=h.document, chunk_id=h.chunk_id, quote=h.text[:120]) for h in hits
    ]
    return AskResponse(answer=answer, citations=citations)
