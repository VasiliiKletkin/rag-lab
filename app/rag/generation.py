"""Этап 4: ответ на вопрос по найденным кускам с помощью Claude."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Protocol

import anthropic

from app.config import get_settings
from app.schemas import AskResponse, Citation, SearchHit

logger = logging.getLogger(__name__)

NO_ANSWER = "В документах нет ответа на этот вопрос."

SYSTEM_PROMPT = f"""Ты отвечаешь на вопросы строго по приложенным документам.
Отвечай на языке вопроса, коротко и по делу, и опирайся только на приложенный текст.
Если в документах нет ответа, ответь ровно: «{NO_ANSWER}»"""


class GenerationError(Exception):
    """Модель недоступна или не смогла ответить; HTTP-код подбирает слой API."""

    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@contextmanager
def api_errors() -> Iterator[None]:
    """Ошибки SDK -> GenerationError: слой API не должен знать классы anthropic."""
    try:
        yield
    except anthropic.RateLimitError as exc:
        raise GenerationError("лимит запросов к модели", retryable=True) from exc
    except anthropic.APIStatusError as exc:
        retryable = exc.status_code >= 500
        raise GenerationError(f"ошибка модели: {exc.status_code}", retryable=retryable) from exc
    except anthropic.APIConnectionError as exc:
        raise GenerationError("модель недоступна", retryable=True) from exc


class Generator(Protocol):
    def answer(self, question: str, hits: list[SearchHit]) -> AskResponse: ...


class ClaudeGenerator:
    """Каждый кусок уходит отдельным блоком `document` с включёнными цитатами:
    модель цитирует дословно, а document_index в цитате сопоставляется с куском."""

    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def answer(self, question: str, hits: list[SearchHit]) -> AskResponse:
        if not hits:
            return AskResponse(answer=NO_ANSWER, citations=[])
        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": h.text},
                "title": f"{h.document} #{h.chunk_id}",
                "citations": {"enabled": True},
            }
            for h in hits
        ]
        with api_errors():
            message = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": [*documents, {"type": "text", "text": question}]}
                ],
            )

        logger.info(
            "claude %s: stop=%s in=%s out=%s",
            message.model,
            message.stop_reason,
            message.usage.input_tokens,
            message.usage.output_tokens,
        )
        if message.stop_reason == "refusal":
            raise GenerationError("модель отказалась отвечать", retryable=False)
        if message.stop_reason == "max_tokens":
            logger.warning("ответ обрезан по max_tokens")

        answer_parts: list[str] = []
        citations: list[Citation] = []
        for block in message.content:
            if block.type != "text":
                continue
            answer_parts.append(block.text)
            for citation in block.citations or []:
                hit = hits[citation.document_index]
                citations.append(
                    Citation(
                        document=hit.document, chunk_id=hit.chunk_id, quote=citation.cited_text
                    )
                )
        answer = "".join(answer_parts).strip()
        if NO_ANSWER in answer:
            # ответ «не знаю» не должен выглядеть как подтверждённый источниками
            return AskResponse(answer=NO_ANSWER, citations=[])
        return AskResponse(answer=answer, citations=citations)


@lru_cache
def get_client() -> anthropic.Anthropic:
    settings = get_settings()
    # в .env.example ключ пустой: пустая строка — это тоже «не задан»
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    if not key:
        raise GenerationError("ANTHROPIC_API_KEY не задан", retryable=False)
    return anthropic.Anthropic(api_key=key)


@lru_cache
def get_generator() -> Generator:
    return ClaudeGenerator(get_client(), get_settings().llm_model)
