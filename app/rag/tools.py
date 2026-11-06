"""Этап 8: инструменты агента — то, что модель может попросить нас выполнить.

Модель сама ничего не вызывает: она возвращает блок tool_use с именем и JSON-аргументами,
а выполняет их наш код. Поэтому аргументы — недоверенный ввод и проверяются как любой запрос.
"""

import json
from collections.abc import Callable
from typing import Any, Literal

import psycopg
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.rag.embeddings import Embedder
from app.rag.retrieval import search

# Описание — это промпт: по нему модель решает, когда и какой инструмент звать.
# strict: API гарантирует, что аргументы соответствуют схеме (нужен additionalProperties: false).
TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_docs",
        "description": (
            "Найти в корпусе куски текста по смыслу запроса. Корпус на английском: "
            "формулируй query по-английски. Возвращает id куска, документ, оценку и текст. "
            "Для вопроса из нескольких частей вызывай отдельно по каждой части."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос на английском"},
                "k": {"type": "integer", "description": "Сколько кусков вернуть, от 1 до 10"},
                "mode": {
                    "type": "string",
                    "enum": ["vector", "hybrid", "rerank"],
                    "description": "rerank — точнее всего; hybrid — если важны точные слова",
                },
            },
            "required": ["query", "k", "mode"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_chunk",
        "description": (
            "Получить кусок по id вместе с соседними кусками того же документа. "
            "Используй, когда найденный кусок обрывается на середине мысли."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "id куска из search_docs"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_documents",
        "description": "Список документов корпуса: путь, заголовок, число кусков.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


class ToolError(Exception):
    """Ошибка, которую модель может исправить сама: уходит ей как tool_result с is_error."""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchDocsArgs(_Args):
    query: str = Field(min_length=1)
    k: int = Field(ge=1, le=10)
    mode: Literal["vector", "hybrid", "rerank"]


class GetChunkArgs(_Args):
    id: int


class NoArgs(_Args):
    pass


def search_docs(conn: psycopg.Connection, embedder: Embedder, args: SearchDocsArgs) -> list[dict]:
    hits = search(conn, embedder, args.query, args.k, args.mode)
    return [
        {"id": int(h.chunk_id), "document": h.document, "score": round(h.score, 3), "text": h.text}
        for h in hits
    ]


GET_CHUNK_SQL = """
SELECT n.id, d.path, n.chunk_index, n.content
FROM chunks c
JOIN chunks n ON n.document_id = c.document_id
             AND n.chunk_index BETWEEN c.chunk_index - 1 AND c.chunk_index + 1
JOIN documents d ON d.id = c.document_id
WHERE c.id = %(id)s
ORDER BY n.chunk_index
"""


def get_chunk(conn: psycopg.Connection, embedder: Embedder, args: GetChunkArgs) -> list[dict]:
    rows = conn.execute(GET_CHUNK_SQL, {"id": args.id}).fetchall()
    if not rows:
        raise ToolError(f"куска с id={args.id} нет")
    return [{"id": i, "document": p, "chunk_index": n, "text": t} for i, p, n, t in rows]


LIST_DOCUMENTS_SQL = """
SELECT d.path, d.title, count(c.id)
FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
GROUP BY d.id ORDER BY d.path
"""


def list_documents(conn: psycopg.Connection, embedder: Embedder, args: NoArgs) -> list[dict]:
    rows = conn.execute(LIST_DOCUMENTS_SQL).fetchall()
    return [{"path": p, "title": t, "chunks": n} for p, t, n in rows]


REGISTRY: dict[str, tuple[type[_Args], Callable[..., list[dict]]]] = {
    "search_docs": (SearchDocsArgs, search_docs),
    "get_chunk": (GetChunkArgs, get_chunk),
    "list_documents": (NoArgs, list_documents),
}


def run_tool(conn: psycopg.Connection, embedder: Embedder, name: str, raw: object) -> str:
    """Выполнить инструмент и вернуть JSON-строку для tool_result.

    ToolError — то, что модель может исправить (нет такого инструмента, плохие аргументы,
    нет такого id). Сбой базы сюда не относится: он пробрасывается и становится 503.
    """
    if name not in REGISTRY:
        raise ToolError(f"нет инструмента {name}; доступны: {', '.join(REGISTRY)}")
    args_model, func = REGISTRY[name]
    try:
        args = args_model.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ToolError(f"неверные аргументы: {problems}") from exc
    return json.dumps(func(conn, embedder, args), ensure_ascii=False)
