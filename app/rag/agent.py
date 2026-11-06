"""Этап 8: цикл агента. Модель решает, что искать; мы выполняем и возвращаем результат.

    вопрос -> модель -> tool_use -> наш код -> tool_result -> модель -> ... -> ответ

Отличие от /ask: там один поиск по исходному вопросу, здесь модель сама формулирует
запросы (по-английски, по частям вопроса) и может дочитать соседние куски.
"""

import logging

import anthropic
import psycopg

from app.rag.embeddings import Embedder
from app.rag.generation import NO_ANSWER, GenerationError, api_errors
from app.rag.tools import TOOLS, ToolError, run_tool
from app.schemas import AgentResponse, ToolCall

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""Ты отвечаешь на вопросы по корпусу технической документации.
Своих знаний не используй: всё, что утверждаешь, должно быть найдено инструментами.
Вопрос из нескольких частей ищи по частям. Отвечай на языке вопроса, коротко,
в конце перечисли id кусков, на которые опирался.
Если поиск ничего не дал, ответь ровно: «{NO_ANSWER}»"""

PREVIEW_CHARS = 300


def run_agent(
    client: anthropic.Anthropic,
    model: str,
    conn: psycopg.Connection,
    embedder: Embedder,
    question: str,
    max_steps: int = 5,
) -> AgentResponse:
    """Шаг = один вызов модели. Лимит шагов — защита от зацикливания и от счёта за токены."""
    messages: list[dict] = [{"role": "user", "content": question}]
    trace: list[ToolCall] = []

    for step in range(1, max_steps + 1):
        with api_errors():
            message = client.messages.create(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                tool_choice={"type": "auto"},
                messages=messages,
            )
        logger.info(
            "agent step %s: stop=%s in=%s out=%s",
            step,
            message.stop_reason,
            message.usage.input_tokens,
            message.usage.output_tokens,
        )
        if message.stop_reason == "refusal":
            raise GenerationError("модель отказалась отвечать", retryable=False)
        if message.stop_reason != "tool_use":
            answer = "".join(b.text for b in message.content if b.type == "text").strip()
            return AgentResponse(answer=answer, trace=trace, steps=step, stopped="answer")

        # ответ модели возвращается в историю целиком: tool_result ссылается на id из tool_use
        messages.append({"role": "assistant", "content": message.content})
        results = []
        for block in message.content:
            if block.type != "tool_use":
                continue
            try:
                output, is_error = run_tool(conn, embedder, block.name, block.input), False
            except ToolError as exc:
                output, is_error = str(exc), True
            trace.append(
                ToolCall(
                    step=step,
                    name=block.name,
                    input=block.input,
                    output=output[:PREVIEW_CHARS],
                    is_error=is_error,
                )
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )
        # все результаты одного шага — одним сообщением user, иначе API вернёт 400
        messages.append({"role": "user", "content": results})

    # лимит исчерпан: честный отказ лучше ответа по недособранным данным
    return AgentResponse(answer=NO_ANSWER, trace=trace, steps=max_steps, stopped="max_steps")
